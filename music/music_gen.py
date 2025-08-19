# -*- coding: utf-8 -*-
"""
生成 BGM：使用 audiocraft 的 MusicGen-small
运行：python src\music_gen.py
输出：outputs/bgm.wav  (32 kHz, mono)
"""

import os
import torch
import torchaudio
from audiocraft.models import MusicGen

OUT_WAV = "outputs/bgm.wav"
PROMPT  = "lofi chill background, warm, soft texture, light drums, no vocals"
DUR_SEC = 60
SR      = 32000

def try_move_to_device(mg, device: str):
    """
    在 audiocraft 1.3.x 中，MusicGen 实例没有 .to() / .cuda() 方法，
    但内部包含的语言模型与压缩模型通常暴露为属性，可单独迁移。
    这里做了“有就迁”的安全处理；如果没有这些属性就跳过。
    """
    try:
        if hasattr(mg, "lm") and hasattr(mg.lm, "to"):
            mg.lm.to(device)
        if hasattr(mg, "compression_model") and hasattr(mg.compression_model, "to"):
            mg.compression_model.to(device)
    except Exception:
        pass

def save_wav_tensor(wav_tensor, path: str, sr: int):
    """
    MusicGen.generate 可能返回 list[Tensor] 或 Tensor。
    统一转换为 (channels, samples) 再用 torchaudio 保存。
    """
    if isinstance(wav_tensor, (list, tuple)):
        wav = wav_tensor[0]
    else:
        wav = wav_tensor
    if wav.dim() == 1:            # (samples,)
        wav = wav.unsqueeze(0)    # -> (1, samples)
    elif wav.dim() == 2:
        pass                      # (channels, samples)
    else:
        wav = wav.squeeze()       # 尽量挤掉多余维度
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # torchaudio.save 支持 float32 [-1,1]
    torchaudio.save(path, wav.cpu().float(), sr)
    print("√ BGM 写入：", path)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 加载预训练模型
    mg = MusicGen.get_pretrained("facebook/musicgen-small")
    # 迁移内部子模块到 device（在你版本中 .to() 可能不可用，这里做兼容）
    try_move_to_device(mg, device)

    # 设置生成参数
    mg.set_generation_params(duration=DUR_SEC)  # 秒
    # 生成（不传 device 参数；在你的版本中 generate 不接受 device）
    wav = mg.generate([PROMPT], progress=True)
    save_wav_tensor(wav, OUT_WAV, SR)

if __name__ == "__main__":
    main()
