#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm.auto import tqdm

try:  # 可选 GPU 支持
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - 动态依赖
    torch = None  # type: ignore
    F = None  # type: ignore

import librosa
from scipy.ndimage import gaussian_filter1d


# --------------------------- 运行环境 ---------------------------

def _detect_device(requested: str | None = None) -> str:
    """返回 ``cpu``/``cuda``/``auto``。若 Torch 不可用则退回 CPU。"""
    if requested:
        requested = requested.lower()
        if requested not in {"cpu", "cuda", "auto"}:
            raise ValueError(f"未知的设备选项: {requested}")
    else:
        requested = "auto"

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("CUDA 设备不可用或 PyTorch 未安装。")
        return "cuda"

    # auto
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _log_device(device: str):
    if device == "cuda":
        name = torch.cuda.get_device_name() if torch is not None else "Unknown"
        print(f"使用 GPU: {name}")
    else:
        print("使用 CPU 处理")


# --------------------------- 通用工具 ---------------------------

def run(cmd: Iterable[str]):
    proc = subprocess.run(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc


def has_audio_ext(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}


def is_mp4(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {".mp4", ".mov", ".mkv"}


class ProcessingMonitor:
    """简单进度监控：每个处理模块更新一次，显示速度和码率估计。"""

    def __init__(self, total_steps: int, sr: int):
        self._bar = tqdm(total=total_steps, desc="Processing", leave=False)
        self._sr = sr

    def step(self, label: str, duration: float, processed_samples: int):
        if duration <= 0:
            realtime = float("inf")
            bitrate = float("inf")
        else:
            realtime = (processed_samples / self._sr) / duration
            bitrate = (processed_samples * 32 / duration) / 1000  # 32bit float 估算
        postfix = {
            "stage": label,
            "speed": f"{realtime:.2f}x" if math.isfinite(realtime) else "inf",
            "bitrate": f"{bitrate:.0f} kbps" if math.isfinite(bitrate) else "inf",
        }
        self._bar.set_postfix(postfix)
        self._bar.update(1)

    def close(self):
        self._bar.close()


# --------------------------- 音频 IO ---------------------------

@dataclass
class Audio:
    y: np.ndarray  # shape [channels, samples]
    sr: int

    @staticmethod
    def load(path: str, target_sr: int = 48000) -> "Audio":
        if not shutil.which("ffmpeg"):
            raise EnvironmentError("ffmpeg 未安装或不可用。请先安装 ffmpeg。")
        with tempfile.TemporaryDirectory() as td:
            pcm = os.path.join(td, "tmp.wav")
            run([
                "ffmpeg", "-y", "-i", path,
                "-vn",
                "-acodec", "pcm_f32le",
                "-ac", "2",
                "-ar", str(target_sr),
                pcm,
            ])
            y, sr = sf.read(pcm, always_2d=True, dtype="float32")
        return Audio(y=y.T, sr=sr)

    def save(self, path: str, format_hint: str | None = None):
        ext = (format_hint or os.path.splitext(path)[1].lower()).lower()
        yT = self.y.T
        if ext == ".wav":
            sf.write(path, yT, self.sr, subtype="PCM_24")
            return
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "tmp.wav")
            sf.write(wav, yT, self.sr, subtype="PCM_24")
            run(["ffmpeg", "-y", "-i", wav, path])


# --------------------------- CPU 处理 ---------------------------

class TriggerAwareGateCPU:
    def __init__(self, sr: int, hop_ms=10, win_ms=40, strength=1.0):
        self.sr = sr
        self.hop = int(sr * hop_ms / 1000)
        self.win = int(sr * win_ms / 1000)
        self.strength = strength

    def process(self, y: np.ndarray) -> np.ndarray:
        C, _ = y.shape
        out = np.zeros_like(y)
        for c in range(C):
            out[c] = self._proc_ch(y[c])
        return out

    def _proc_ch(self, x: np.ndarray) -> np.ndarray:
        S = librosa.stft(x, n_fft=self._nfft(), hop_length=self.hop, win_length=self.win, window="hann")
        mag, ang = np.abs(S), np.angle(S)
        frame_power = mag.mean(axis=0)
        th = np.quantile(frame_power, 0.1)
        noise_frames = mag[:, frame_power <= th]
        noise_profile = np.median(noise_frames, axis=1) + 1e-8

        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self._nfft())
        hi_band = (freqs >= 3000) & (freqs <= 8000)
        centroid = librosa.feature.spectral_centroid(S=mag, sr=self.sr).squeeze()
        zcr = librosa.feature.zero_crossing_rate(y=librosa.istft(S, hop_length=self.hop, win_length=self.win))
        zcr = zcr.squeeze() if zcr.ndim > 1 else zcr
        hi_energy_ratio = (mag[hi_band, :].mean(axis=0) / (mag.mean(axis=0) + 1e-8))

        label = np.zeros(mag.shape[1], dtype=int)
        label[(centroid < 2000) & (hi_energy_ratio < 0.25)] = 1
        label[(centroid > 2500) & (hi_energy_ratio > 0.35)] = 2

        alpha_base = 1.2 * self.strength
        alpha_whisper = 0.8 * self.strength
        alpha_sibilant = 0.6 * self.strength

        protect = np.ones_like(noise_profile)
        protect[hi_band] = 0.65

        M = np.empty_like(mag)
        for t in range(mag.shape[1]):
            a = alpha_base
            if label[t] == 1:
                a = alpha_whisper
            elif label[t] == 2:
                a = alpha_sibilant
            nr = noise_profile * protect
            mask = 1.0 - np.exp(-(mag[:, t] / (nr * a + 1e-8)))
            M[:, t] = np.clip(mask, 0.0, 1.0)

        S_hat = (mag * M) * np.exp(1j * ang)
        y_hat = librosa.istft(S_hat, hop_length=self.hop, win_length=self.win)
        return self._match_length(y_hat, len(x))

    def _nfft(self):
        n = 1
        target = max(1024, int(2 ** np.ceil(np.log2(self.win))))
        while n < target:
            n *= 2
        return n

    @staticmethod
    def _match_length(x: np.ndarray, N: int) -> np.ndarray:
        if len(x) < N:
            return np.pad(x, (0, N - len(x)))
        return x[:N]


class MicroDynamicsUpwardCPU:
    def __init__(self, gain_db=4.0, knee_db=10.0):
        self.gain_db = gain_db
        self.knee_db = knee_db

    def process(self, y: np.ndarray) -> np.ndarray:
        sr = 48000
        win = int(sr * 0.05)
        eps = 1e-9
        out = np.zeros_like(y)
        for c in range(y.shape[0]):
            x = y[c]
            frames = len(x) // win
            x_pad = x[: frames * win].reshape(frames, win)
            rms = np.sqrt((x_pad ** 2).mean(axis=1) + eps)
            db = 20 * np.log10(rms + eps)
            g = np.zeros_like(db)
            lo, hi = -50, -30
            inside = (db > lo) & (db < hi)
            g[inside] = (db[inside] - lo) / (hi - lo) * self.gain_db
            g[db >= hi] = self.gain_db
            g = gaussian_filter1d(g, sigma=2)
            gain = 10 ** (g / 20)
            x_pad = x_pad * gain[:, None]
            y_out = x_pad.reshape(-1)
            out[c, : len(y_out)] = y_out
            out[c, len(y_out) :] = x[len(y_out) :]
        return np.clip(out, -1.0, 1.0)


class SubSoothingDynamicShelfCPU:
    def __init__(self, sr: int, min_gain_db=-3.0, freq=90.0):
        self.sr = sr
        self.min_gain = min_gain_db
        self.freq = freq

    def process(self, y: np.ndarray) -> np.ndarray:
        nfft = 2048
        hop = 480
        win = 1920
        out = np.zeros_like(y)
        for c in range(y.shape[0]):
            x = y[c]
            S = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=win, window="hann")
            mag = np.abs(S)
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=nfft)
            lo = (freqs >= 40) & (freqs <= 120)
            hi = (freqs >= 3000) & (freqs <= 8000)
            lo_e = mag[lo, :].mean(axis=0) + 1e-8
            hi_e = mag[hi, :].mean(axis=0) + 1e-8
            ratio = hi_e / (lo_e + 1e-8)
            r = (ratio - np.percentile(ratio, 10)) / (np.percentile(ratio, 90) - np.percentile(ratio, 10) + 1e-8)
            r = np.clip(r, 0, 1)
            gain_db = self.min_gain * r
            G = np.ones_like(mag)
            G[lo, :] *= (10 ** (gain_db[None, :] / 20))
            S_hat = (mag * G) * np.exp(1j * np.angle(S))
            y_hat = librosa.istft(S_hat, hop_length=hop, win_length=win)
            out[c, : len(y_hat)] = y_hat
            out[c, len(y_hat) :] = x[len(y_hat) :]
        return np.clip(out, -1.0, 1.0)


class LoudnessGlueCPU:
    def __init__(self, target_lufs=-26.0, true_peak_ceiling_db=-1.0):
        self.target = target_lufs
        self.ceiling = true_peak_ceiling_db

    def process(self, y: np.ndarray, sr: int) -> np.ndarray:
        meter = pyln.Meter(sr)
        mono = y.mean(axis=0)
        loud = meter.integrated_loudness(mono)
        gain_db = self.target - loud
        g = 10 ** (gain_db / 20)
        y = y * g
        up = librosa.resample(y, orig_sr=sr, target_sr=sr * 4, axis=1)
        tp = np.max(np.abs(up)) + 1e-9
        target_lin = 10 ** (self.ceiling / 20)
        if tp > target_lin:
            y *= (target_lin / tp)
        return np.clip(y, -1.0, 1.0)


# --------------------------- GPU 处理 ---------------------------

class _TorchBase:
    def __init__(self, device: torch.device, sr: int):
        if torch is None:
            raise RuntimeError("需要安装 PyTorch 才能使用 GPU 模式。")
        self.device = device
        self.sr = sr


class TriggerAwareGateTorch(_TorchBase):
    def __init__(self, sr: int, device: torch.device, hop_ms=10, win_ms=40, strength=1.0):
        super().__init__(device, sr)
        self.hop = int(sr * hop_ms / 1000)
        self.win = int(sr * win_ms / 1000)
        self.strength = strength

    def process(self, y: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.win, device=self.device)
        n_fft = self._nfft()
        freqs = torch.fft.rfftfreq(n_fft, 1.0 / self.sr, device=self.device)
        hi_band = (freqs >= 3000) & (freqs <= 8000)
        outputs = []
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

            power = mag.square()
            centroid = (freqs[:, None] * power).sum(dim=0) / (power.sum(dim=0) + 1e-8)
            hi_energy_ratio = mag[hi_band, :].mean(dim=0) / (mag.mean(dim=0) + 1e-8)

            label = torch.zeros(mag.shape[1], dtype=torch.int64, device=self.device)
            label[(centroid < 2000) & (hi_energy_ratio < 0.25)] = 1
            label[(centroid > 2500) & (hi_energy_ratio > 0.35)] = 2

            alpha_base = 1.2 * self.strength
            alpha_whisper = 0.8 * self.strength
            alpha_sibilant = 0.6 * self.strength

            protect = torch.ones_like(noise_profile)
            protect[hi_band] = 0.65
            nr = noise_profile * protect

            M = torch.empty_like(mag)
            for idx in range(mag.shape[1]):
                a = alpha_base
                if label[idx] == 1:
                    a = alpha_whisper
                elif label[idx] == 2:
                    a = alpha_sibilant
                mask = 1.0 - torch.exp(-(mag[:, idx] / (nr * a + 1e-8)))
                M[:, idx] = torch.clamp(mask, 0.0, 1.0)

            norm_phase = S / (mag + 1e-8)
            S_hat = mag * M * norm_phase
            y_hat = torch.istft(
                S_hat,
                n_fft=n_fft,
                hop_length=self.hop,
                win_length=self.win,
                window=window,
                length=ch.shape[-1],
            )
            outputs.append(y_hat)
        return torch.stack(outputs, dim=0)

    def _nfft(self) -> int:
        target = max(1024, int(2 ** math.ceil(math.log2(self.win))))
        n = 1
        while n < target:
            n *= 2
        return n


class MicroDynamicsUpwardTorch(_TorchBase):
    def __init__(self, device: torch.device, gain_db=4.0, knee_db=10.0):
        super().__init__(device=device, sr=48000)
        self.gain_db = gain_db
        self.knee_db = knee_db

    def process(self, y: torch.Tensor) -> torch.Tensor:
        win = int(self.sr * 0.05)
        eps = 1e-9
        channels = []
        for ch in y:
            frames = ch.shape[-1] // win
            trimmed = ch[: frames * win]
            if frames == 0:
                channels.append(ch)
                continue
            x_pad = trimmed.view(frames, win)
            rms = torch.sqrt((x_pad.pow(2).mean(dim=1) + eps))
            db = 20 * torch.log10(rms + eps)
            g = torch.zeros_like(db)
            lo, hi = -50.0, -30.0
            inside = (db > lo) & (db < hi)
            g[inside] = (db[inside] - lo) / (hi - lo) * self.gain_db
            g[db >= hi] = self.gain_db
            g = gaussian_filter1d(g.cpu().numpy(), sigma=2)
            gain = torch.from_numpy(10 ** (g / 20)).to(self.device, dtype=ch.dtype)
            x_pad = x_pad * gain[:, None]
            y_out = x_pad.reshape(-1)
            merged = torch.zeros_like(ch)
            merged[: y_out.shape[0]] = y_out
            merged[y_out.shape[0] :] = ch[y_out.shape[0] :]
            channels.append(torch.clamp(merged, -1.0, 1.0))
        return torch.stack(channels, dim=0)


class SubSoothingDynamicShelfTorch(_TorchBase):
    def __init__(self, sr: int, device: torch.device, min_gain_db=-3.0):
        super().__init__(device=device, sr=sr)
        self.min_gain = min_gain_db

    def process(self, y: torch.Tensor) -> torch.Tensor:
        nfft = 2048
        hop = 480
        win = 1920
        window = torch.hann_window(win, device=self.device)
        freqs = torch.fft.rfftfreq(nfft, 1.0 / self.sr, device=self.device)
        lo = (freqs >= 40) & (freqs <= 120)
        hi = (freqs >= 3000) & (freqs <= 8000)
        outputs = []
        for ch in y:
            S = torch.stft(
                ch,
                n_fft=nfft,
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
            p10 = torch.quantile(ratio, 0.1)
            p90 = torch.quantile(ratio, 0.9)
            r = (ratio - p10) / (p90 - p10 + 1e-8)
            r = torch.clamp(r, 0.0, 1.0)
            gain_db = self.min_gain * r
            G = torch.ones_like(mag)
            G[lo, :] *= 10 ** (gain_db[None, :] / 20)
            phase = S / (mag + 1e-8)
            S_hat = mag * G * phase
            y_hat = torch.istft(
                S_hat,
                n_fft=nfft,
                hop_length=hop,
                win_length=win,
                window=window,
                length=ch.shape[-1],
            )
            padded = torch.zeros_like(ch)
            padded[: y_hat.shape[0]] = y_hat
            padded[y_hat.shape[0] :] = ch[y_hat.shape[0] :]
            outputs.append(torch.clamp(padded, -1.0, 1.0))
        return torch.stack(outputs, dim=0)


class LoudnessGlueTorch(_TorchBase):
    def __init__(self, target_lufs=-26.0, true_peak_ceiling_db=-1.0, sr: int = 48000, device: torch.device | None = None):
        if device is None:
            device = torch.device("cpu")
        super().__init__(device=device, sr=sr)
        self.target = target_lufs
        self.ceiling = true_peak_ceiling_db

    def process(self, y: torch.Tensor) -> torch.Tensor:
        meter = pyln.Meter(self.sr)
        mono = y.mean(dim=0).detach().cpu().numpy()
        loud = meter.integrated_loudness(mono)
        gain_db = self.target - loud
        g = 10 ** (gain_db / 20)
        y = y * g
        up = F.interpolate(y.unsqueeze(0), scale_factor=4, mode="linear", align_corners=False)
        tp = up.abs().max() + 1e-9
        target_lin = 10 ** (self.ceiling / 20)
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
    device: str = "auto"


def enhance_audio(ai: Audio, cfg: Settings) -> Audio:
    device_choice = _detect_device(cfg.device)
    _log_device(device_choice)
    monitor = ProcessingMonitor(
        total_steps=sum([cfg.gate, cfg.microdyn, cfg.lowfreq, cfg.loudnorm]),
        sr=ai.sr,
    )

    y = ai.y.astype(np.float32, copy=True)
    sr = ai.sr
    assert sr == 48000, "内部采样率应为 48kHz"

    if device_choice == "cpu" or torch is None:
        if cfg.gate:
            start = time.perf_counter()
            gate = TriggerAwareGateCPU(sr, strength=1.0)
            y = gate.process(y)
            monitor.step("gate", time.perf_counter() - start, y.shape[1])
        if cfg.microdyn:
            start = time.perf_counter()
            md = MicroDynamicsUpwardCPU(gain_db=4.0)
            y = md.process(y)
            monitor.step("microdyn", time.perf_counter() - start, y.shape[1])
        if cfg.lowfreq:
            start = time.perf_counter()
            lf = SubSoothingDynamicShelfCPU(sr, min_gain_db=-3.0)
            y = lf.process(y)
            monitor.step("lowfreq", time.perf_counter() - start, y.shape[1])
        if cfg.loudnorm:
            start = time.perf_counter()
            ln = LoudnessGlueCPU(target_lufs=-26.0, true_peak_ceiling_db=-1.0)
            y = ln.process(y, sr)
            monitor.step("loudnorm", time.perf_counter() - start, y.shape[1])
    else:  # cuda
        torch_device = torch.device("cuda")
        tensor = torch.from_numpy(y).to(torch_device)
        if cfg.gate:
            start = time.perf_counter()
            gate = TriggerAwareGateTorch(sr, device=torch_device, strength=1.0)
            tensor = gate.process(tensor)
            monitor.step("gate", time.perf_counter() - start, tensor.shape[1])
        if cfg.microdyn:
            start = time.perf_counter()
            md = MicroDynamicsUpwardTorch(device=torch_device, gain_db=4.0)
            tensor = md.process(tensor)
            monitor.step("microdyn", time.perf_counter() - start, tensor.shape[1])
        if cfg.lowfreq:
            start = time.perf_counter()
            lf = SubSoothingDynamicShelfTorch(sr, device=torch_device, min_gain_db=-3.0)
            tensor = lf.process(tensor)
            monitor.step("lowfreq", time.perf_counter() - start, tensor.shape[1])
        if cfg.loudnorm:
            start = time.perf_counter()
            ln = LoudnessGlueTorch(target_lufs=-26.0, true_peak_ceiling_db=-1.0, sr=sr, device=torch_device)
            tensor = ln.process(tensor)
            monitor.step("loudnorm", time.perf_counter() - start, tensor.shape[1])
        y = tensor.detach().cpu().numpy()

    monitor.close()
    return Audio(y=y, sr=sr)


def process_file(input_path: str, output_path: str | None, cfg: Settings):
    tmpdir = tempfile.mkdtemp(prefix="asmr_enh_")
    try:
        if is_mp4(input_path):
            wav_in = os.path.join(tmpdir, "audio_in.wav")
            run(["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_f32le", "-ac", "2", "-ar", "48000", wav_in])
            ai = Audio.load(wav_in, target_sr=48000)
            ao = enhance_audio(ai, cfg)
            wav_out = os.path.join(tmpdir, "audio_out.wav")
            ao.save(wav_out, ".wav")
            out = output_path or os.path.splitext(input_path)[0] + "_enh.mp4"
            run([
                "ffmpeg", "-y", "-i", input_path,
                "-i", wav_out,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "320k",
                out,
            ])
            print(f"已输出: {out}")
        else:
            ai = Audio.load(input_path, target_sr=48000)
            ao = enhance_audio(ai, cfg)
            out = output_path or _auto_outname(input_path)
            ao.save(out)
            print(f"已输出: {out}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _auto_outname(p: str) -> str:
    stem, ext = os.path.splitext(p)
    if ext.lower() not in {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}:
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
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="选择运算设备（默认自动检测）")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Settings(
        gate=not args.no_gate,
        microdyn=not args.no_microdyn,
        lowfreq=not args.no_lowfreq,
        loudnorm=not args.no_loudnorm,
        device=args.device,
    )
    process_file(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
