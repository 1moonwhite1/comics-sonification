import os, json, pathlib, re
from TTS.api import TTS
from tqdm import tqdm
import soundfile as sf
import numpy as np

INP = "outputs\dialogs_spk.json"
OUT_DIR = "outputs/wav_demo"
SPLIT_LEN = 160        # 长句切段上限
SIL_PAUSE = 0.20       # 分段间隙（秒）
LANG = "en"            # COMICS 文本为英文，如需中文改为 "zh"

pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
dialogs = json.load(open(INP, encoding="utf-8"))

# 加载 XTTS‑v2（保持与当前环境兼容的写法）
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# —— 使用你指定的两个说话人 —— #
spk_A = "Claribel Dervla"
spk_B = "Daisy Studious"
SPK_MAP = {"A": spk_A, "B": spk_B}
print("使用的说话人：", SPK_MAP)

def clean_text(x:str)->str:
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"\s+([,.;:!?])", r"\1", x)
    x = re.sub(r"[-–—]\s+", "- ", x)
    return x.strip()

def chunk_text(x:str, n=160):
    x = clean_text(x)
    if len(x) <= n: return [x]
    parts, cur = [], []
    for w in x.split():
        cur.append(w)
        if sum(len(t)+1 for t in cur) > n:
            parts.append(" ".join(cur)); cur=[]
    if cur: parts.append(" ".join(cur))
    return parts

def resample_to_32k(wav, sr):
    if sr == 32000: return wav
    import math
    scale = 32000 / sr
    idx = (np.arange(math.floor(len(wav)*scale)) / scale).astype(np.float64)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0+1, len(wav)-1)
    frac = idx - i0
    return (1-frac)*wav[i0] + frac*wav[i1]

for d in tqdm(dialogs):
    out_wav = os.path.join(OUT_DIR, f"{d['id']:05d}_{d['speaker']}.wav")
    if os.path.exists(out_wav):
        d["wav"] = out_wav
        continue

    voice = SPK_MAP.get(d["speaker"], spk_A)  # A/B 映射
    try:
        segs = []
        for i, ck in enumerate(chunk_text(d["text"], SPLIT_LEN)):
            tmp = f"{out_wav}.tmp_{i}.wav"
            # 关键：传入有效的 speaker 名称 + 语言
            tts.tts_to_file(text=ck, speaker=voice, language=LANG, file_path=tmp)
            w, sr = sf.read(tmp, dtype="float32"); os.remove(tmp)
            segs.append(w); segs.append(np.zeros(int(SIL_PAUSE*sr), dtype=np.float32))
        wav = np.concatenate(segs)
        wav = resample_to_32k(wav, sr=sr)
        sf.write(out_wav, wav, 32000, subtype="PCM_16")
        d["wav"] = out_wav
    except Exception as e:
        print("跳过 id", d.get("id"), ":", repr(e))

json.dump(dialogs, open(INP, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("✔ 语音输出目录:", OUT_DIR)
