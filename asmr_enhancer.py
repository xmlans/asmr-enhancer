#!/usr/bin/env python3
"""ASMR Enhancer

一个尽量保留酥麻质感的批处理脚本：读取音频或视频文件，进行降噪、微动态增强、
低频塑形与响度校准，然后输出处理后的音频或重新封装好的视频。

依赖：Python 3.9+、ffmpeg 以及 ``requirements.txt`` 中列出的库。
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

import math
import time

import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def _detect_device(force: str | None = None) -> torch.device:
    if force:
        if force.lower() == "cpu":
            return torch.device("cpu")
        if force.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 设备不可用，无法选择 cuda。")
            return torch.device("cuda")
        raise ValueError(f"未知的设备选项: {force}")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------- 工具函数 ---------------------------

def run(cmd: list[str]):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc


class ProcessingMonitor:
    def __init__(self, total_steps: int, sr: int):
        self._bar = tqdm(total=total_steps, desc="Processing", leave=False)
        self._sr = sr

    def step(self, label: str, duration: float, processed_samples: int):
        if duration <= 0:
            realtime = float("inf")
            bitrate = float("inf")
        else:
            realtime = (processed_samples / self._sr) / duration
            bitrate = (processed_samples * 32 / duration) / 1000  # 32-bit float approximation
        postfix = {
            "stage": label,
            "speed": f"{realtime:.2f}x",
            "bitrate": f"{bitrate:.0f} kbps" if math.isfinite(bitrate) else "inf",
        }
        self._bar.set_postfix(postfix)
        self._bar.update(1)

    def close(self):
        self._bar.close()


def is_mp4(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {".mp4", ".mov", ".mkv"}


# --------------------------- 音频IO ---------------------------
@dataclass
class Audio:
    y: torch.Tensor  # shape [channels, samples]
    sr: int

    @staticmethod
    def load(path: str, target_sr: int = 48000, device: torch.device | None = None) -> "Audio":
        """通过 ffmpeg 解码为 float32 PCM, 保留声道数，重采样到 target_sr。"""
        if not shutil.which("ffmpeg"):
            raise EnvironmentError("ffmpeg 未安装或不可用。请先安装 ffmpeg。")
        with tempfile.TemporaryDirectory() as td:
            pcm = os.path.join(td, "tmp.wav")
            run([
                "ffmpeg", "-y", "-i", path,
                "-vn",  # 忽略视频
                "-acodec", "pcm_f32le",
                "-ac", "2",  # 强制双声道（若原为单声道，将复制到 L/R）
                "-ar", str(target_sr),
                pcm
            ])
            y_np, sr = sf.read(pcm, always_2d=True, dtype="float32")  # [N, C]
        y_np = y_np.T  # [C, N]
        tensor = torch.from_numpy(y_np)
        if device is not None:
            tensor = tensor.to(device)
        return Audio(y=tensor, sr=sr)

    def to(self, device: torch.device) -> "Audio":
        return Audio(y=self.y.to(device), sr=self.sr)

    def save(self, path: str, format_hint: str | None = None):
        """保存为 wav/mp3/m4a… 由扩展名决定；非常见格式走 ffmpeg。"""
        ext = (format_hint or os.path.splitext(path)[1].lower())
        ext = ext.lower()
        y_cpu = self.y.detach().cpu().numpy().T
        if ext == ".wav":
            sf.write(path, y_cpu, self.sr, subtype="PCM_24")
            return
        # 其它走 ffmpeg
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "tmp.wav")
            sf.write(wav, y_cpu, self.sr, subtype="PCM_24")
            run(["ffmpeg", "-y", "-i", wav, path])


# --------------------------- DSP 组件 ---------------------------
class TriggerAwareGate:
    """基于简单触发器感知的谱门限，支持 GPU 运算。"""

    def __init__(self, sr: int, hop_ms=10, win_ms=40, strength=1.0):
        self.sr = sr
        self.hop = int(sr * hop_ms / 1000)
        self.win = int(sr * win_ms / 1000)
        self.strength = strength

    def process(self, y: torch.Tensor) -> torch.Tensor:
        device = y.device
        window = torch.hann_window(self.win, device=device)
        n_fft = self._nfft()
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / self.sr, device=device)
        hi_band = (freqs >= 3000) & (freqs <= 8000)
        out = []
        for ch in y:
            S = torch.stft(
                ch,
                n_fft=n_fft,
                hop_length=self.hop,
                win_length=self.win,
                window=window,
                center=True,
                return_complex=True,
            )
            mag = torch.abs(S)
            frame_power = mag.mean(dim=0)
            th = torch.quantile(frame_power, 0.1)
            mask_noise = frame_power <= th
            if mask_noise.any():
                noise_profile = torch.median(mag[:, mask_noise], dim=1).values
            else:
                noise_profile = torch.median(mag, dim=1).values
            noise_profile = noise_profile + 1e-8

            power = mag.pow(2)
            centroid = (freqs[:, None] * power).sum(dim=0) / (power.sum(dim=0) + 1e-8)
            hi_energy_ratio = mag[hi_band, :].mean(dim=0) / (mag.mean(dim=0) + 1e-8)

            label = torch.zeros(mag.shape[1], dtype=torch.int64, device=device)
            label[(centroid < 2000) & (hi_energy_ratio < 0.25)] = 1
            label[(centroid > 2500) & (hi_energy_ratio > 0.35)] = 2

            alpha_base = 1.2 * self.strength
            alpha_whisper = 0.8 * self.strength
            alpha_sibilant = 0.6 * self.strength

            protect = torch.ones_like(noise_profile)
            protect[hi_band] = 0.65

            M = torch.empty_like(mag)
            norm_phase = S / (mag + 1e-8)
            nr = noise_profile * protect
            for idx in range(mag.shape[1]):
                a = alpha_base
                if label[idx] == 1:
                    a = alpha_whisper
                elif label[idx] == 2:
                    a = alpha_sibilant
                mask = 1.0 - torch.exp(-(mag[:, idx] / (nr * a + 1e-8)))
                M[:, idx] = torch.clamp(mask, 0.0, 1.0)

            S_hat = mag * M * norm_phase
            y_hat = torch.istft(
                S_hat,
                n_fft=n_fft,
                hop_length=self.hop,
                win_length=self.win,
                window=window,
                length=ch.shape[-1],
            )
            out.append(y_hat)
        return torch.stack(out, dim=0)

    def _nfft(self) -> int:
        target = max(1024, int(2 ** math.ceil(math.log2(self.win))))
        n = 1
        while n < target:
            n *= 2
        return n


class MicroDynamicsUpward:
    """极低电平 upward expansion：仅在 -50~-30 dBFS 之间轻抬，其他不动。"""

    def __init__(self, gain_db=4.0, knee_sigma=2.0):
        self.gain_db = gain_db
        self.knee_sigma = knee_sigma

    def process(self, y: torch.Tensor) -> torch.Tensor:
        sr = 48000
        win = int(sr * 0.05)
        if win <= 0:
            return y
        device = y.device
        eps = 1e-9
        C, N = y.shape
        frames = N // win
        if frames == 0:
            return y
        y_main = y[:, : frames * win]
        x_blocks = y_main.view(C, frames, win)
        rms = torch.sqrt(x_blocks.pow(2).mean(dim=2) + eps)
        db = 20 * torch.log10(rms + eps)
        lo, hi = -50.0, -30.0
        gain_db = torch.zeros_like(db)
        inside = (db > lo) & (db < hi)
        gain_db[inside] = (db[inside] - lo) / (hi - lo) * self.gain_db
        gain_db[db >= hi] = self.gain_db

        if frames > 1:
            kernel_size = max(3, int(6 * self.knee_sigma) | 1)
            coords = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
            kernel = torch.exp(-(coords ** 2) / (2 * self.knee_sigma ** 2))
            kernel = (kernel / kernel.sum()).view(1, 1, -1)
            pad = kernel_size // 2
            smoothed = F.conv1d(
                gain_db.view(C, 1, frames), kernel.expand(C, -1, -1), padding=pad, groups=C
            ).view(C, frames)
        else:
            smoothed = gain_db

        linear_gain = torch.pow(10.0, smoothed / 20.0)
        processed = (x_blocks * linear_gain.unsqueeze(-1)).view(C, frames * win)
        if frames * win < N:
            tail = y[:, frames * win :]
            processed = torch.cat([processed, tail], dim=1)
        return torch.clamp(processed, -1.0, 1.0)


class SubSoothingDynamicShelf:
    """低频动态低架 EQ：当检测到触发器(高频比↑)时，40–120 Hz 轻度收紧。"""

    def __init__(self, sr: int, min_gain_db=-3.0):
        self.sr = sr
        self.min_gain = min_gain_db

    def process(self, y: torch.Tensor) -> torch.Tensor:
        n_fft = 1024
        hop = 480
        win = 1920
        device = y.device
        window = torch.hann_window(win, device=device)
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / self.sr, device=device)
        lo = (freqs >= 40) & (freqs <= 120)
        hi = (freqs >= 3000) & (freqs <= 8000)
        out = []
        for ch in y:
            S = torch.stft(
                ch,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                center=True,
                return_complex=True,
            )
            mag = torch.abs(S)
            lo_e = mag[lo, :].mean(dim=0) + 1e-8
            hi_e = mag[hi, :].mean(dim=0) + 1e-8
            ratio = hi_e / (lo_e + 1e-8)
            p10, p90 = torch.quantile(ratio, torch.tensor([0.1, 0.9], device=device))
            denom = (p90 - p10).clamp(min=1e-6)
            r = torch.clamp((ratio - p10) / denom, 0.0, 1.0)
            gain_db = self.min_gain * r
            low_gain = torch.pow(10.0, gain_db / 20.0)
            gain = torch.ones_like(mag)
            gain[lo, :] *= low_gain
            norm_phase = S / (mag + 1e-8)
            S_hat = mag * gain * norm_phase
            y_hat = torch.istft(
                S_hat,
                n_fft=n_fft,
                hop_length=hop,
                win_length=win,
                window=window,
                length=ch.shape[-1],
            )
            out.append(y_hat)
        out_tensor = torch.stack(out, dim=0)
        return torch.clamp(out_tensor, -1.0, 1.0)


class LoudnessGlue:
    """目标 -26 LUFS, True Peak ceiling -1 dBTP (简化TP)。"""

    def __init__(self, target_lufs=-26.0, true_peak_ceiling_db=-1.0):
        self.target = target_lufs
        self.ceiling = true_peak_ceiling_db

    def process(self, y: torch.Tensor, sr: int) -> torch.Tensor:
        device = y.device
        mono = y.mean(dim=0).detach().cpu().numpy()
        meter = pyln.Meter(sr)
        loud = meter.integrated_loudness(mono)
        gain_db = self.target - loud
        g = torch.pow(torch.tensor(10.0, device=device), gain_db / 20.0)
        y = y * g
        up = F.interpolate(y.unsqueeze(0), scale_factor=4, mode="linear", align_corners=False).squeeze(0)
        tp = up.abs().max() + 1e-9
        target_lin = torch.pow(torch.tensor(10.0, device=device), self.ceiling / 20.0)
        if tp > target_lin:
            y = y * (target_lin / tp)
        return torch.clamp(y, -1.0, 1.0)


# --------------------------- 主流程 ---------------------------
@dataclass
class Settings:
    gate: bool = True
    microdyn: bool = True
    lowfreq: bool = True
    loudnorm: bool = True


def enhance_audio(ai: Audio, cfg: Settings, device: torch.device) -> Audio:
    y = ai.y.to(device)
    sr = ai.sr
    if sr != 48000:
        raise ValueError("内部采样率应为 48kHz")

    steps = sum([cfg.gate, cfg.microdyn, cfg.lowfreq, cfg.loudnorm])
    monitor = ProcessingMonitor(steps, sr) if steps else None
    sample_count = y.shape[0] * y.shape[1]

    if cfg.gate:
        start = time.time()
        gate = TriggerAwareGate(sr, strength=1.0)
        y = gate.process(y)
        if monitor:
            monitor.step("gate", time.time() - start, sample_count)

    if cfg.microdyn:
        start = time.time()
        md = MicroDynamicsUpward(gain_db=4.0)
        y = md.process(y)
        if monitor:
            monitor.step("microdyn", time.time() - start, sample_count)

    if cfg.lowfreq:
        start = time.time()
        lf = SubSoothingDynamicShelf(sr, min_gain_db=-3.0)
        y = lf.process(y)
        if monitor:
            monitor.step("lowfreq", time.time() - start, sample_count)

    if cfg.loudnorm:
        start = time.time()
        ln = LoudnessGlue(target_lufs=-26.0, true_peak_ceiling_db=-1.0)
        y = ln.process(y, sr)
        if monitor:
            monitor.step("loudnorm", time.time() - start, sample_count)

    if monitor:
        monitor.close()

    return Audio(y=y, sr=sr)


def process_file(input_path: str, output_path: str | None, cfg: Settings, device: torch.device):
    tmpdir = tempfile.mkdtemp(prefix="asmr_enh_")
    try:
        if is_mp4(input_path):
            # 抽出音频
            wav_in = os.path.join(tmpdir, "audio_in.wav")
            run(["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_f32le", "-ac", "2", "-ar", "48000", wav_in])
            ai = Audio.load(wav_in, target_sr=48000, device=device)
            ao = enhance_audio(ai, cfg, device)
            wav_out = os.path.join(tmpdir, "audio_out.wav")
            ao.save(wav_out, ".wav")
            # 复封装
            out = output_path or os.path.splitext(input_path)[0] + "_enh.mp4"
            run(["ffmpeg", "-y", "-i", input_path, "-i", wav_out, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", out])
            print(f"✅ 已输出: {out}")
        else:
            # 纯音频
            ai = Audio.load(input_path, target_sr=48000, device=device)
            ao = enhance_audio(ai, cfg, device)
            out = output_path or _auto_outname(input_path)
            ao.save(out)
            print(f"✅ 已输出: {out}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _auto_outname(p: str) -> str:
    stem, ext = os.path.splitext(p)
    if ext.lower() not in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
        # 默认输出 wav
        return stem + "_enh.wav"
    return stem + "_enh" + ext


# --------------------------- CLI ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="ASMR 一键优化 — 触发器友好降噪 / 微动态 / 低频塑形 / LUFS 校准")
    ap.add_argument("input", help="输入文件（mp3/wav/mp4…）")
    ap.add_argument("-o", "--output", help="输出文件路径（默认自动命名）")
    ap.add_argument("--no-gate", action="store_true", help="关闭触发器守护降噪")
    ap.add_argument("--no-microdyn", action="store_true", help="关闭微动态扩展")
    ap.add_argument("--no-lowfreq", action="store_true", help="关闭低频动态低架")
    ap.add_argument("--no-loudnorm", action="store_true", help="关闭 LUFS/峰值校准")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="选择运行设备（默认自动检测）")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Settings(
        gate=not args.no_gate,
        microdyn=not args.no_microdyn,
        lowfreq=not args.no_lowfreq,
        loudnorm=not args.no_loudnorm,
    )
    device = _detect_device(None if args.device == "auto" else args.device)
    print(f"🚀 计算设备: {device}")
    if device.type == "cuda":
        print(f"⚡️ CUDA 已启用，当前 GPU: {torch.cuda.get_device_name(device)}")
    process_file(args.input, args.output, cfg, device)


if __name__ == "__main__":
    main()
