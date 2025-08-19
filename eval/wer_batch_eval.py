# -*- coding: utf-8 -*-
import os, re, json, argparse, csv
from tqdm import tqdm
import numpy as np, soundfile as sf, math
import torch, whisper

# ==== 轻清洗 + 数字与口语归一（与 wer_eval_v3 保持一致） ====
def clean_light(s:str)->str:
    if s is None: return ""
    s = s.strip().replace("“",'"').replace("”",'"').replace("’","'").replace("‘","'")
    s = s.replace("—","-").replace("–","-")
    return re.sub(r"\s+"," ",s)

def normalize_number_words(s:str)->str:
    s = s.replace("-", " ")
    units = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
    teens = {"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19}
    tens  = {"twenty":20,"thirty":30,"forty":30+10,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
    scales= {"hundred":100,"thousand":1000,"million":1000000}
    toks = s.split(); out=[]; i=0; N=len(toks)
    def is_num(tok): return tok in units or tok in teens or tok in tens or tok in scales or tok in ("a","an")
    while i<N:
        t=toks[i].lower()
        if not is_num(t): out.append(toks[i]); i+=1; continue
        start=i; total=0; cur=0; matched=False
        while i<N:
            t=toks[i].lower()
            if t in ("a","an"):
                if i+1<N and toks[i+1].lower() in ("hundred","thousand","million"):
                    cur+=1; matched=True; i+=1; t=toks[i].lower()
                else: break
            if t in units: cur+=units[t]; matched=True; i+=1
            elif t in teens: cur+=teens[t]; matched=True; i+=1
            elif t in tens:
                val=tens[t]
                if i+1<N and toks[i+1].lower() in units: val+=units[toks[i+1].lower()]; i+=1
                cur+=val; matched=True; i+=1
            elif t=="hundred": 
                if cur==0: cur=1
                cur*=100; matched=True; i+=1
            elif t=="thousand":
                if cur==0: cur=1
                total+=cur*1000; cur=0; matched=True; i+=1
            elif t=="million":
                if cur==0: cur=1
                total+=cur*1000000; cur=0; matched=True; i+=1
            else: break
        if matched: total+=cur; out.append(str(total))
        else: out.append(toks[start]); i=start+1
    return " ".join(out)

REPL = {
    "c'mon":"come on", "cmon":"come on", "come-on":"come on",
    "hi ya":"hiya", "hi-ya":"hiya",
    "youre":"you're","dont":"don't","wont":"won't","im":"i'm","ive":"i've","shes":"she's","hes":"he's","theyre":"they're","lets":"let's",
    "ok":"okay",
    "jerry":"jeri",  # 按语料可调整
    "jay":"j"
}

def norm_for_metric(s:str, num_norm=True)->str:
    s = clean_light(s).lower()
    if num_norm: s = normalize_number_words(s)
    for a,b in REPL.items():
        s = re.sub(rf"\b{re.escape(a)}\b", b, s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)           # 千分位逗号
    s = re.sub(r"[^\w\s]", " ", s)                 # 去标点
    return re.sub(r"\s+"," ",s).strip()

# ==== 参考文本 ====
def ref_from_json(path): 
    data = json.load(open(path, encoding="utf-8"))
    data = sorted(data, key=lambda d:d.get("id",0))
    return " ".join(clean_light(d.get("text","")) for d in data if clean_light(d.get("text","")))

# ==== WER/CER 计数 ====
def align_counts_words(ref_str, hyp_str, num_norm=True):
    r = norm_for_metric(ref_str, num_norm).split()
    h = norm_for_metric(hyp_str, num_norm).split()
    n, m = len(r), len(h)
    if n==0: return dict(N=0,H=0,S=0,D=0,I=m,WER=0.0 if m==0 else 1.0)
    dp=[[0]*(m+1) for _ in range(n+1)]
    bt=[[None]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0]=i; bt[i][0]='D'
    for j in range(1,m+1): dp[0][j]=j; bt[0][j]='I'
    for i in range(1,n+1):
        for j in range(1,m+1):
            if r[i-1]==h[j-1]: dp[i][j]=dp[i-1][j-1]; bt[i][j]='H'
            else:
                sub=dp[i-1][j-1]+1; dele=dp[i-1][j]+1; ins=dp[i][j-1]+1
                best=min(sub,dele,ins); dp[i][j]=best; bt[i][j]='S' if best==sub else ('D' if best==dele else 'I')
    i,j=n,m; H=S=D=I=0
    while i>0 or j>0:
        op=bt[i][j]
        if op=='H': H+=1; i-=1; j-=1
        elif op=='S': S+=1; i-=1; j-=1
        elif op=='D': D+=1; i-=1
        elif op=='I': I+=1; j-=1
        else: break
    return dict(N=n,H=H,S=S,D=D,I=I,WER=(S+D+I)/max(1,n))

def cer_counts(ref_str, hyp_str, num_norm=True):
    r=list(norm_for_metric(ref_str,num_norm)); h=list(norm_for_metric(hyp_str,num_norm))
    n,m=len(r),len(h)
    if n==0: return dict(N=0,CER=0.0 if m==0 else 1.0)
    dp=[[0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0]=i
    for j in range(1,m+1): dp[0][j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            if r[i-1]==h[j-1]: dp[i][j]=dp[i-1][j-1]
            else: dp[i][j]=min(dp[i-1][j-1]+1, dp[i-1][j]+1, dp[i][j-1]+1)
    return dict(N=n, CER=dp[n][m]/n)

# ==== Whisper 推理 ====
def asr_whisper(model, audio_path, lang="en"):
    return model.transcribe(
        audio_path, language=lang, task="transcribe",
        fp16=torch.cuda.is_available(),
        temperature=0.0, best_of=5, beam_size=5,
        condition_on_previous_text=False,
        no_speech_threshold=0.3, logprob_threshold=-1.0, compression_ratio_threshold=2.4
    )["text"]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_json", required=True)
    ap.add_argument("--audios", nargs="+", required=True, help="一组要评测的 .wav 路径")
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--out_csv", default=r"outputs\wer_batch.csv")
    ap.add_argument("--no_num_norm", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    ref = ref_from_json(args.ref_json)
    device="cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(args.model, device=device)
    num_norm = not args.no_num_norm

    rows=[]
    for wav in tqdm(args.audios):
        hyp = asr_whisper(model, wav, lang="en")
        w = align_counts_words(ref, hyp, num_norm)
        c = cer_counts(ref, hyp, num_norm)
        rows.append({
            "audio": os.path.basename(wav),
            "WER": round(w["WER"]*100,2),
            "CER": round(c["CER"]*100,2),
            "N": w["N"], "H": w["H"], "S": w["S"], "D": w["D"], "I": w["I"]
        })

    with open(args.out_csv,"w",newline="",encoding="utf-8") as f:
        wcsv=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader(); wcsv.writerows(rows)
    print(f"✅ 写出: {args.out_csv}")

if __name__ == "__main__":
    main()
