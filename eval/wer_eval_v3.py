# -*- coding: utf-8 -*-
"""
稳健版 WER/CER 评测（本地对齐，不依赖 jiwer 高级 API）
- Whisper ASR 得到假设转写
- 从 JSON 或文本拼接参考转写
- 词级/字符级动态规划对齐，输出 WER、CER、S/D/I/H/N
- ★ 新增：数字归一（"three hundred" -> "300" 等），可用 --no_num_norm 关闭

用法（全段）：
  python src\wer_eval_v3.py ^
    --audio outputs\demo_mix.wav ^
    --ref_json outputs\dialogs_demo_spk_deess.json ^
    --model small.en

用法（分段 + SRT）：
  python src\wer_eval_v3.py ^
    --audio outputs\demo_mix.wav ^
    --ref_json outputs\dialogs_demo_spk_deess.json ^
    --srt outputs\demo_mix.srt ^
    --model small.en ^
    --segmented
"""
import os, re, json, argparse, math
from tqdm import tqdm

import numpy as np
import soundfile as sf
import torch
import whisper

# =========================
# 文本清洗与数字归一
# =========================

def clean_light(s: str) -> str:
    """与合成侧一致的轻清洗：统一引号/破折号，压空格"""
    if s is None: return ""
    s = s.strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_number_words(s: str) -> str:
    """
    将英文数字词组归一为阿拉伯数字：
    - 支持 a/an hundred, thousand, million
    - 支持 units/teens/tens + 可选 'hundred' + 可选 'thousand'/'million'
    - 支持 twenty-three（先把连字符视作空格再处理）
    例：'three hundred' -> '300', 'a thousand' -> '1000', 'twenty three' -> '23'
    """
    # 先把连字符切开，避免 twenty-three 被当作一个 token
    s = s.replace("-", " ")

    units = {
        "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9
    }
    teens = {
        "ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
        "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19
    }
    tens = {
        "twenty":20,"thirty":30,"forty":40,"fifty":50,
        "sixty":60,"seventy":70,"eighty":80,"ninety":90
    }
    scales = {"hundred":100, "thousand":1000, "million":1000000}

    toks = s.split()
    out = []
    i = 0
    N = len(toks)

    def is_num_word(tok):
        return tok in units or tok in teens or tok in tens or tok in scales or tok in ("a","an")

    while i < N:
        t = toks[i].lower()
        if not is_num_word(t):
            out.append(toks[i])
            i += 1
            continue

        # 尝试吃下一个“数字词片段”
        start = i
        total = 0
        current = 0
        matched = False

        while i < N:
            t = toks[i].lower()

            # 允许 a/an hundred/thousand/million
            if t in ("a","an"):
                if i+1 < N and toks[i+1].lower() in ("hundred","thousand","million"):
                    current += 1
                    matched = True
                    i += 1
                    t = toks[i].lower()
                else:
                    break  # 'a' 不接数量级时，结束数字解析

            if t in units:
                current += units[t]; matched = True; i += 1
            elif t in teens:
                current += teens[t]; matched = True; i += 1
            elif t in tens:
                # 允许 tens + unit（如 twenty three）
                val = tens[t]
                if i+1 < N and toks[i+1].lower() in units:
                    val += units[toks[i+1].lower()]
                    i += 1
                current += val; matched = True; i += 1
            elif t == "hundred":
                # 允许 lone 'hundred' 视为 100
                if current == 0: current = 1
                current *= 100; matched = True; i += 1
            elif t == "thousand":
                if current == 0: current = 1
                total += current * 1000
                current = 0
                matched = True
                i += 1
            elif t == "million":
                if current == 0: current = 1
                total += current * 1000000
                current = 0
                matched = True
                i += 1
            else:
                break

        if matched:
            total += current
            out.append(str(total))
        else:
            # 没有真正匹配到（孤立的 'a' 之类），按原样输出一个 token
            out.append(toks[start])
            i = start + 1

    return " ".join(out)

def norm_for_metric(s: str, do_num_norm: bool = True) -> str:
    s = clean_light(s).lower()
    if do_num_norm:
        s = normalize_number_words(s)

    # —— 新增：口语/拼写归一（可继续按你的语料扩充）——
    REPL = {
        "c'mon": "come on",
        "cmon": "come on",
        "come-on": "come on",
        "hi ya": "hiya",          # 或反过来都行，关键是统一
        "hi-ya": "hiya",
        "hiya": "hiya",
        "yep": "yep",             # 示例：保留
        "yeah": "yeah",
        "ya": "you",              # 口语 you/ya 统一（也可不做，看你偏好）
        "dont": "don't",          # 去标点前先补常见省略（利于后续去标点后统一）
        "wont": "won't",
        "im": "i'm",
        "ive": "i've",
        "youre": "you're",
        "hes": "he's",
        "shes": "she's",
        "theyre": "they're",
        "lets": "let's",
        "ok": "okay",
        "okay": "okay",
        "jerry": "jeri",          # 统一专名的一个写法（按你的数据具体调整）
        "jay": "j",               # “jay”→字母“j”
        "gee": "g",
        "cee": "c",
    }
    # 词级替换：只在完整词边界替换，避免误伤
    for a, b in REPL.items():
        s = re.sub(rf"\b{re.escape(a)}\b", b, s)

    # 删除数字里的千分位逗号（1,000 -> 1000）
    s = re.sub(r"(?<=\d),(?=\d)", "", s)

    # 之后再去标点并压空格
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ref_from_json(path: str) -> str:
    data = json.load(open(path, encoding="utf-8"))
    data = sorted(data, key=lambda d: d.get("id", 0))
    texts = [clean_light(d.get("text","")) for d in data if clean_light(d.get("text",""))]
    return " ".join(texts)

def ref_from_txt(path: str) -> str:
    return clean_light(open(path, encoding="utf-8").read())

# =========================
# SRT 解析（用于分段评测）
# =========================

def read_srt(path: str):
    pat_time = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)")
    segs = []
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    i=0
    while i < len(lines):
        if re.match(r"^\d+$", lines[i]): i+=1
        if i>=len(lines): break
        m = pat_time.search(lines[i]) if i < len(lines) else None
        if not m: i+=1; continue
        to_s = lambda h,mn,s,ms: int(h)*3600+int(mn)*60+int(s)+int(ms)/1000
        st = to_s(*m.groups()[:4]); ed = to_s(*m.groups()[4:])
        i+=1
        txt=[]
        while i<len(lines) and lines[i].strip()!="":
            txt.append(lines[i].strip()); i+=1
        segs.append({"start":st, "end":ed, "text":" ".join(txt)})
        while i<len(lines) and lines[i].strip()=="":
            i+=1
    return segs

# =========================
# 音频加载 / Whisper ASR
# =========================

def load_audio(path, sr=16000):
    wav, or_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim==2:
        wav = np.mean(wav, axis=1)
    if or_sr != sr:
        scale = sr / or_sr
        idx = (np.arange(math.floor(len(wav)*scale)) / scale).astype(np.float64)
        i0 = np.floor(idx).astype(int)
        i1 = np.minimum(i0+1, len(wav)-1)
        frac = idx - i0
        wav = (1-frac)*wav[i0] + frac*wav[i1]
    return wav, sr

def slice_audio(wav, sr, start_s, end_s):
    i0 = max(0, int(start_s*sr))
    i1 = min(len(wav), int(end_s*sr))
    return wav[i0:i1]

def asr_whisper(model, audio, language="en"):
    return model.transcribe(
        audio,
        language=language,
        task="transcribe",
        fp16=torch.cuda.is_available(),
        temperature=0.0,          # 贪心/低温度
        best_of=5,                # 取更优候选
        beam_size=5,              # 开 beam search
        condition_on_previous_text=False,   # 避免长音频串联时“带偏”
        no_speech_threshold=0.3,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4
    )["text"]


# =========================
# 对齐计数（S/D/I/H/N）
# =========================

def align_counts_words(ref_str: str, hyp_str: str, do_num_norm: bool = True):
    r = norm_for_metric(ref_str, do_num_norm=do_num_norm).split()
    h = norm_for_metric(hyp_str, do_num_norm=do_num_norm).split()
    n, m = len(r), len(h)
    if n == 0:
        return dict(N=0, H=0, S=0, D=0, I=m, WER=0.0 if m==0 else 1.0)

    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i; bt[i][0] = 'D'
    for j in range(1, m+1):
        dp[0][j] = j; bt[0][j] = 'I'
    for i in range(1, n+1):
        for j in range(1, m+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]; bt[i][j] = 'H'
            else:
                sub = dp[i-1][j-1] + 1
                dele = dp[i-1][j] + 1
                ins = dp[i][j-1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                bt[i][j] = 'S' if best==sub else ('D' if best==dele else 'I')

    i, j = n, m
    H=S=D=I=0
    while i>0 or j>0:
        op = bt[i][j]
        if op == 'H':
            H += 1; i-=1; j-=1
        elif op == 'S':
            S += 1; i-=1; j-=1
        elif op == 'D':
            D += 1; i-=1
        elif op == 'I':
            I += 1; j-=1
        else:
            break
    WER = (S + D + I) / n
    return dict(N=n, H=H, S=S, D=D, I=I, WER=WER)

def cer_counts(ref_str: str, hyp_str: str, do_num_norm: bool = True):
    r = list(norm_for_metric(ref_str, do_num_norm=do_num_norm))
    h = list(norm_for_metric(hyp_str, do_num_norm=do_num_norm))
    n, m = len(r), len(h)
    if n == 0:
        return dict(N=0, CER=0.0 if m==0 else 1.0)

    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): dp[i][0] = i
    for j in range(1, m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if r[i-1] == h[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1)
    dist = dp[n][m]
    return dict(N=n, CER=dist/n)

# =========================
# CLI 主流程
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref_json")
    ap.add_argument("--text")
    ap.add_argument("--srt")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--segmented", action="store_true")
    ap.add_argument("--no_num_norm", action="store_true",
                    help="关闭数字归一（默认开启）")
    args = ap.parse_args()

    # 参考文本
    if args.ref_json:
        ref = ref_from_json(args.ref_json)
    elif args.text:
        ref = ref_from_txt(args.text)
    else:
        print("❌ 需要 --ref_json 或 --text"); return
    if not ref.strip():
        print("❌ 参考文本为空"); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using Whisper model: {args.model} on {device}")
    model = whisper.load_model(args.model, device=device)

    do_num_norm = not args.no_num_norm

    if not args.segmented:
        hyp = asr_whisper(model, args.audio, language="en")
        word_meas = align_counts_words(ref, hyp, do_num_norm=do_num_norm)
        char_meas = cer_counts(ref, hyp, do_num_norm=do_num_norm)

        rn = norm_for_metric(ref, do_num_norm=do_num_norm)
        hn = norm_for_metric(hyp, do_num_norm=do_num_norm)

        print("\n=== Global ASR Result ===")
        print("Reference (norm):", rn[:200] + ("..." if len(rn)>200 else ""))
        print("Hypothesis (norm):", hn[:200] + ("..." if len(hn)>200 else ""))
        print(f"WER: {word_meas['WER']*100:.2f}%  |  CER: {char_meas['CER']*100:.2f}%")
        print(f"Counts: N={word_meas['N']}, H={word_meas['H']}, S={word_meas['S']}, D={word_meas['D']}, I={word_meas['I']}")

    else:
        if not args.srt:
            print("❌ --segmented 需要 --srt"); return
        segs = read_srt(args.srt)
        if not segs:
            print("❌ SRT 解析失败"); return
        wav, sr = load_audio(args.audio, sr=16000)

        hyps = []
        per_word = []
        for s in tqdm(segs, desc="segment ASR"):
            a = slice_audio(wav, sr, s["start"], s["end"])
            if len(a) < 1600:   # <0.1s 忽略
                hyps.append(""); continue
            hyp_i = asr_whisper(model, a, language="en")
            hyps.append(hyp_i)
            per_word.append(align_counts_words(s["text"], hyp_i, do_num_norm=do_num_norm))

        ref_concat = " ".join([clean_light(s["text"]) for s in segs])
        hyp_concat = " ".join(hyps)

        word_all = align_counts_words(ref_concat, hyp_concat, do_num_norm=do_num_norm)
        char_all = cer_counts(ref_concat, hyp_concat, do_num_norm=do_num_norm)

        print("\n=== Segmented ASR Result (concatenated) ===")
        print(f"WER: {word_all['WER']*100:.2f}%  |  CER: {char_all['CER']*100:.2f}%")

        if per_word:
            N = sum(x["N"] for x in per_word)
            S = sum(x["S"] for x in per_word)
            D = sum(x["D"] for x in per_word)
            I = sum(x["I"] for x in per_word)
            H = sum(x["H"] for x in per_word)
            w_seg = (S+D+I)/max(1,N)
            print(f"(Segments aggregated) N={N}, H={H}, S={S}, D={D}, I={I} | WER≈{w_seg*100:.2f}%")

if __name__ == "__main__":
    main()
