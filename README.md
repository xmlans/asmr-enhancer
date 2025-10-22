# ASMR Enhancer / ASMR 声音优化器

## Overview / 项目概览
- **EN:** ASMR Enhancer cleans up whisper-heavy recordings while keeping the tingles intact. It decodes incoming audio/video files to 48 kHz stereo PCM, runs a small mastering chain, and writes the processed result back to disk.
- **ZH:** ASMR Enhancer 专注于在保留酥麻感的同时去除底噪、提升细节。脚本会将输入音频或视频解码成 48 kHz 双声道 PCM，经过一组定制化处理后再输出到目标文件。

## Processing chain / 处理流程
1. **Trigger-aware spectral gate / 触发器守护降噪**：识别齿擦、耳语等高频触发器并降低过度抑制。
2. **Microdynamics upward expansion / 微动态扩展**：温和拉起 -50~-30 dBFS 范围的细节。
3. **Dynamic low-shelf / 动态低架 EQ**：在触发器占比高时收紧 40–120 Hz 低频。
4. **Loudness glue / 响度校准**：对齐到约 -26 LUFS，并限制真峰值至 -1 dBTP。

## Requirements / 环境依赖
- **Python** 3.9 或更高版本
- **ffmpeg** (命令行可用)
- **CUDA (可选)**：若系统安装了兼容的 NVIDIA 驱动/Toolkit 和 PyTorch，脚本会自动切换到 GPU；否则自动回退到 CPU。
- **Python 包**：`pip install -r requirements.txt`（其中 PyTorch 可选，未安装时将始终使用 CPU）。

## Usage / 使用方法
```bash
python asmr_enhancer.py input.wav -o output.wav
python asmr_enhancer.py mix.mp3 --no-lowfreq -o mix_enh.mp3
python asmr_enhancer.py video.mp4 -o video_enh.mp4
python asmr_enhancer.py input.wav --device cuda  # 手动指定 GPU（若可用）
```
- **EN:** Omit `-o` to let the script auto-name the output. Flags like `--no-gate` allow you to skip individual modules.
- **ZH:** 如果不指定 `-o` 会自动生成带 `_enh` 后缀的文件名；可以用 `--no-*` 参数分别关闭某个处理模块；使用 `--device` 可强制选择 `cpu`/`cuda` 或自动检测。

## Notes / 备注
- **EN:** When processing video, the script only swaps the audio track and keeps the original video stream untouched.
- **ZH:** 处理视频文件时仅替换音轨，画面数据保持不变。

## Speed / 速度
- **EN:** The speed depends on your hardware performance. Using NVIDIA CUDA can usually speed up the process several times, while using a CPU is slower. The longer the video/audio, the longer the processing time. Typically, processing one hour of content takes 3-5 minutes (using CUDA).
- **ZH:** 速度取决于你的硬件性能，使用NVIDIA CUDA通常能够将速度加快数倍，而使用CPU则较慢。越长的视频/音频处理时间越长，通常1小时的内容处理时间在3-5分钟（使用CUDA）

## 测试
非常感谢星梦ASMR免费资源的所有用户参与本项目的测试，通过测试5组视频和2组音频，并收集一些用户反馈后此声音优化器处理后的asmr文件确实有更好的听觉效果。本测试于2025年10月22日结束

## Thanks / 感谢
[星梦ASMR免费资源站](https://www.asmrzy.top) <br>
[Telegram 星梦ASMR免费资源](https://t.me/asmrzytop) <br>
[FFmpeg](https://ffmpeg.org)
