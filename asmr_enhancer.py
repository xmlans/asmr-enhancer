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

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln


# --------------------------- 工具函数 ---------------------------

def run(cmd: list[str]):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{proc.stderr}")
    return proc


def is_mp4(path: str) -> bool:
    return os.path.splitext(path.lower())[1] in {".mp4", ".mov", ".mkv"}


# --------------------------- 音频IO ---------------------------
@dataclass
class Audio:
    y: np.ndarray  # shape [channels, samples] (float32)
    sr: int

    @staticmethod
    def load(path: str, target_sr: int = 48000) -> "Audio":
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
            y, sr = sf.read(pcm, always_2d=True, dtype="float32")  # [N, C]
        y = y.T  # [C, N]
        return Audio(y=y, sr=sr)

    def save(self, path: str, format_hint: str | None = None):
        """保存为 wav/mp3/m4a… 由扩展名决定；非常见格式走 ffmpeg。"""
        ext = (format_hint or os.path.splitext(path)[1].lower())
        ext = ext.lower()
        yT = self.y.T
        if ext == ".wav":
            sf.write(path, yT, self.sr, subtype="PCM_24")
            return
        # 其它走 ffmpeg
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "tmp.wav")
            sf.write(wav, yT, self.sr, subtype="PCM_24")
            run(["ffmpeg", "-y", "-i", wav, path])


# --------------------------- DSP 组件 ---------------------------
class TriggerAwareGate:
    """基于简单触发器感知的谱门限：
    - 估计静音/底噪的中位谱；
    - 帧级做“耳语/齿擦/一般”粗分类（高频能量与谱质心启发式）；
    - 对不同类使用不同软掩膜强度，保护 3–8 kHz 细节。
    """
    def __init__(self, sr: int, hop_ms=10, win_ms=40, strength=1.0):
        self.sr = sr
        self.hop = int(sr * hop_ms / 1000)
        self.win = int(sr * win_ms / 1000)
        self.strength = strength

    def process(self, y: np.ndarray) -> np.ndarray:
        C, N = y.shape
        out = np.zeros_like(y)
        for c in range(C):
            out[c] = self._proc_ch(y[c])
        return out

    def _proc_ch(self, x: np.ndarray) -> np.ndarray:
        S = librosa.stft(x, n_fft=self._nfft(), hop_length=self.hop, win_length=self.win, window="hann")
        mag, ang = np.abs(S), np.angle(S)
        # 估计噪声谱为能量最低 10% 帧的中位数
        frame_power = mag.mean(axis=0)
        th = np.quantile(frame_power, 0.1)
        noise_frames = mag[:, frame_power <= th]
        noise_profile = np.median(noise_frames, axis=1) + 1e-8

        # 触发器粗分类
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self._nfft())
        hi_band = (freqs >= 3000) & (freqs <= 8000)
        centroid = librosa.feature.spectral_centroid(S=mag, sr=self.sr).squeeze()
        zcr = librosa.feature.zero_crossing_rate(y=librosa.istft(S, hop_length=self.hop, win_length=self.win))
        zcr = zcr.squeeze() if zcr.ndim > 1 else zcr
        hi_energy_ratio = (mag[hi_band, :].mean(axis=0) / (mag.mean(axis=0) + 1e-8))

        # 标签：0=一般，1=耳语，2=齿擦/口腔
        label = np.zeros(mag.shape[1], dtype=int)
        label[(centroid < 2000) & (hi_energy_ratio < 0.25)] = 1  # 耳语偏低质心+低高频比
        label[(centroid > 2500) & (hi_energy_ratio > 0.35)] = 2  # 齿擦/口腔高频占比大

        # 构造软掩膜，不同标签不同攻强
        alpha_base = 1.2 * self.strength  # 普通
        alpha_whisper = 0.8 * self.strength  # 耳语：更温和
        alpha_sibilant = 0.6 * self.strength  # 齿擦：最温和

        # 保护 3–8kHz：减小该带的抑制
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
        # 保证足够频率分辨率
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


class MicroDynamicsUpward:
    """极低电平 upward expansion：仅在 -50~-30 dBFS 之间轻抬，其他不动。"""
    def __init__(self, gain_db=4.0, knee_db=10.0):
        self.gain_db = gain_db
        self.knee_db = knee_db

    def process(self, y: np.ndarray) -> np.ndarray:
        # RMS 窗 50ms
        sr = 48000  # 由外部保证
        win = int(sr * 0.05)
        eps = 1e-9
        out = np.zeros_like(y)
        for c in range(y.shape[0]):
            x = y[c]
            # 计算滑动 RMS（简化：块状）
            frames = len(x) // win
            x_pad = x[: frames * win].reshape(frames, win)
            rms = np.sqrt((x_pad ** 2).mean(axis=1) + eps)
            db = 20 * np.log10(rms + eps)
            # 目标在 [-50,-30] 之间线性抬升 gain_db，带柔和 knee
            g = np.zeros_like(db)
            lo, hi = -50, -30
            inside = (db > lo) & (db < hi)
            g[inside] = (db[inside] - lo) / (hi - lo) * self.gain_db
            g[db >= hi] = self.gain_db
            # knee 平滑
            from scipy.ndimage import gaussian_filter1d
            g = gaussian_filter1d(g, sigma=2)
            gain = 10 ** (g / 20)
            x_pad = x_pad * gain[:, None]
            y_out = x_pad.reshape(-1)
            out[c, : len(y_out)] = y_out
            out[c, len(y_out) :] = x[len(y_out) :]
        return np.clip(out, -1.0, 1.0)


class SubSoothingDynamicShelf:
    """低频动态低架 EQ：当检测到触发器(高频比↑)时，40–120 Hz 轻度收紧。"""
    def __init__(self, sr: int, min_gain_db=-3.0, freq=90.0):
        self.sr = sr
        self.min_gain = min_gain_db
        self.freq = freq
        # 简单双一阶滤波器实现低架，系数实时插值

    def process(self, y: np.ndarray) -> np.ndarray:
        nfft = 1024
        hop = 480  # 10 ms @ 48k
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
            # 归一化到 [0,1], 触发器强→接近1
            r = (ratio - np.percentile(ratio, 10)) / (np.percentile(ratio, 90) - np.percentile(ratio, 10) + 1e-8)
            r = np.clip(r, 0, 1)
            # 将 r 映射到增益 [min_gain, 0]
            gain_db = self.min_gain * r
            # 应用到低频 bin 上
            G = np.ones_like(mag)
            G[lo, :] *= (10 ** (gain_db[None, :] / 20))
            S_hat = (mag * G) * np.exp(1j * np.angle(S))
            y_hat = librosa.istft(S_hat, hop_length=hop, win_length=win)
            out[c, : len(y_hat)] = y_hat
            out[c, len(y_hat) :] = x[len(y_hat) :]
        return np.clip(out, -1.0, 1.0)


class LoudnessGlue:
    """目标 -26 LUFS, True Peak ceiling -1 dBTP (简化TP)。"""
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
        # 简化 True Peak 限幅（4x 上采样峰值估计）
        up = librosa.resample(y, orig_sr=sr, target_sr=sr * 4, axis=1)
        tp = np.max(np.abs(up)) + 1e-9
        target_lin = 10 ** (self.ceiling / 20)
        if tp > target_lin:
            y *= (target_lin / tp)
        return np.clip(y, -1.0, 1.0)


# --------------------------- 主流程 ---------------------------
@dataclass
class Settings:
    gate: bool = True
    microdyn: bool = True
    lowfreq: bool = True
    loudnorm: bool = True


def enhance_audio(ai: Audio, cfg: Settings) -> Audio:
    y = ai.y.copy()
    sr = ai.sr
    assert sr == 48000, "内部采样率应为 48kHz"

    if cfg.gate:
        gate = TriggerAwareGate(sr, strength=1.0)
        y = gate.process(y)

    if cfg.microdyn:
        md = MicroDynamicsUpward(gain_db=4.0)
        y = md.process(y)

    if cfg.lowfreq:
        lf = SubSoothingDynamicShelf(sr, min_gain_db=-3.0)
        y = lf.process(y)

    if cfg.loudnorm:
        ln = LoudnessGlue(target_lufs=-26.0, true_peak_ceiling_db=-1.0)
        y = ln.process(y, sr)

    return Audio(y=y, sr=sr)


def process_file(input_path: str, output_path: str | None, cfg: Settings):
    tmpdir = tempfile.mkdtemp(prefix="asmr_enh_")
    try:
        if is_mp4(input_path):
            # 抽出音频
            wav_in = os.path.join(tmpdir, "audio_in.wav")
            run(["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_f32le", "-ac", "2", "-ar", "48000", wav_in])
            ai = Audio.load(wav_in, target_sr=48000)
            ao = enhance_audio(ai, cfg)
            wav_out = os.path.join(tmpdir, "audio_out.wav")
            ao.save(wav_out, ".wav")
            # 复封装
            out = output_path or os.path.splitext(input_path)[0] + "_enh.mp4"
            run(["ffmpeg", "-y", "-i", input_path, "-i", wav_out, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", out])
            print(f"✅ 已输出: {out}")
        else:
            # 纯音频
            ai = Audio.load(input_path, target_sr=48000)
            ao = enhance_audio(ai, cfg)
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
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Settings(
        gate=not args.no_gate,
        microdyn=not args.no_microdyn,
        lowfreq=not args.no_lowfreq,
        loudnorm=not args.no_loudnorm,
    )
    process_file(args.input, args.output, cfg)


if __name__ == "__main__":
    main()
