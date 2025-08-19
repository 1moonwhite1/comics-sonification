# -*- coding: utf-8 -*-
"""
对说话人 B 做 +1 半音的轻微变调（使用 torchaudio），并更新 JSON 指向
输入 JSON:  outputs/dialogs_demo_spk_clean.json   ← 建议先跑 clean_ocr_text + 重新 TTS
输出 JSON:  outputs/dialogs_demo_spk_pitch.json
输出音频:    outputs/wav_demo_pitch/*.wav
运行:       python src\pitch_shift_speaker_ta.py
"""
import os, json
import torch
import torchaudio
from tqdm import tqdm

INP_JSON = "outputs\dialogs_spk.json"
OUT_JSON = "outputs/dialogs_spk_pitch.json"
OUT_DIR  = "outputs/wav_demo_pitch"

# 说话人半音调整：A 不变、B +1
SEMITONES = {"A": 0.0, "B": 1.0}

# 可选：先小批量验证（None 表示处理全部）
MAX_ITEMS = None  # 例如想先试 60，就改为 60

os.makedirs(OUT_DIR, exist_ok=True)
dialogs = json.load(open(INP_JSON, encoding="utf-8"))

device = "cuda" if torch.cuda.is_available() else "cpu"

# 检测可用的变调途径
HAS_TA_PITCH = hasattr(torchaudio.functional, "pitch_shift")
HAS_SOX = hasattr(torchaudio.sox_effects, "apply_effects_tensor")
print(f"Using device: {device} | torchaudio.pitch_shift: {HAS_TA_PITCH} | sox_effects: {HAS_SOX}")

def shift_with_torchaudio(wav: torch.Tensor, sr: int, steps: float) -> torch.Tensor:
    """优先使用 torchaudio.functional.pitch_shift（CPU/GPU 均可）"""
    if steps == 0:
        return wav
    return torchaudio.functional.pitch_shift(wav.to(device), sr, n_steps=steps).cpu()

def shift_with_sox(wav: torch.Tensor, sr: int, steps: float) -> torch.Tensor:
    """回退：使用 SoX 的 pitch（单位：cents），记得随后恢复采样率"""
    if steps == 0:
        return wav
    cents = int(round(steps * 100))
    eff = [["pitch", str(cents)], ["rate", str(sr)]]
    out, out_sr = torchaudio.sox_effects.apply_effects_tensor(wav, sr, eff)
    # out_sr 应该等于 sr；以防万一做一次重采样修正
    if out_sr != sr:
        out = torchaudio.functional.resample(out, out_sr, sr)
    return out

updated = 0
skipped = 0
errors  = 0

it = dialogs if MAX_ITEMS is None else dialogs[:MAX_ITEMS]
for d in tqdm(it, total=len(it)):
    wav_path = d.get("wav")
    spk = d.get("speaker", "A")
    if not wav_path or not os.path.exists(wav_path):
        skipped += 1
        continue

    steps = SEMITONES.get(spk, 0.0)
    base = os.path.basename(wav_path)
    out_path = os.path.join(OUT_DIR, base)

    try:
        wav, sr = torchaudio.load(wav_path)      # (channels, samples), float32/-1..1
        # 统一到 float32
        wav = wav.to(torch.float32)

        if steps == 0.0:
            # 直接保存到目标目录
            torchaudio.save(out_path, wav, sr)
        else:
            if HAS_TA_PITCH:
                out = shift_with_torchaudio(wav, sr, steps)
            elif HAS_SOX:
                out = shift_with_sox(wav, sr, steps)
            else:
                # 最保守退路：不变调直接复制（保证流水线不中断）
                out = wav
            torchaudio.save(out_path, out, sr)

        d["wav"] = out_path
        updated += 1

    except Exception as e:
        errors += 1
        # 失败时，至少复制到新目录，保证后续混音可继续
        try:
            torchaudio.save(out_path, wav, sr)
            d["wav"] = out_path
        except Exception:
            pass
        print(f"[warn] 处理失败，已回退：{base} | 错误：{repr(e)}")

# 写回新的 JSON
json.dump(dialogs, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"✅ 完成：更新 {updated} 条，跳过 {skipped} 条，错误 {errors} 条 → {OUT_DIR}")
print(f"✅ 新 JSON：{OUT_JSON}")
