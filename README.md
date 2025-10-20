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
- Python 依赖：`pip install -r requirements.txt`

## Usage / 使用方法
```bash
python asmr_enhancer.py input.wav -o output.wav
python asmr_enhancer.py mix.mp3 --no-lowfreq -o mix_enh.mp3
python asmr_enhancer.py video.mp4 -o video_enh.mp4
```
- **EN:** Omit `-o` to let the script auto-name the output. Flags like `--no-gate` allow you to skip individual modules.
- **ZH:** 如果不指定 `-o` 会自动生成带 `_enh` 后缀的文件名；可以用 `--no-*` 参数分别关闭某个处理模块。

## Notes / 备注
- **EN:** When processing video, the script only swaps the audio track and keeps the original video stream untouched.
- **ZH:** 处理视频文件时仅替换音轨，画面数据保持不变。

