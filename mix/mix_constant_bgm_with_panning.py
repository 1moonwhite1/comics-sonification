# -*- coding: utf-8 -*-
"""
稳定 BGM +（可选）轻度 ducking + A/B 说话人立体声定位
运行：python src\mix_constant_bgm_with_panning.py
输入：
  - outputs/dialogs_demo_spk.json   ← tts_batch_demo.py 产生并带有 speaker/页码
  - outputs/wav_demo/*.wav          ← 每条对白的音频
  - outputs/bgm.wav                 ← 固定风格 BGM（music_gen.py 生成）
  - assets/sfx/page_turn.wav        ← 可选，翻页 SFX（没有则自动跳过）
输出：
  - outputs/demo_mix.wav
"""
import os, json, math
from pydub import AudioSegment

INP_JSON = "outputs/dialogs_spk_deess.json"
VOICE_DIR  = "outputs/wav_demo"
BGM_WAV    = "outputs/bgm.wav"
PAGE_SFX   = "assets/sfx/page_turn.wav"   # 可选
OUT_WAV    = "outputs/demo_mix.wav"

# === 可调参数 ===
PAUSE_MS     = 160   # 句间停顿，与 build_timeline_and_srt 保持一致
BGM_BASE_DB  = -9    # BGM 基础降低（建议 -10~-6）
DUCK_DB      = -3    # 白话期间对 BGM 额外压低（设 0 即“完全不 duck”）
FADE_MS      = 50    # BGM 在对白段内的淡入/淡出，避免突兀
PAN_MAP      = {"A": -0.18, "B": 0.18}  # A/B 轻度定位；新角色默认 0
# ==============

def to_stereo_32k(seg: AudioSegment) -> AudioSegment:
    if seg.frame_rate != 32000: seg = seg.set_frame_rate(32000)
    if seg.channels != 2: seg = seg.set_channels(2)
    return seg

def simple_pan(seg: AudioSegment, pan: float) -> AudioSegment:
    """
    pan ∈ [-1, 1]，负数偏左，正数偏右。
    使用简单声道增益法（非完美等响，但足够轻量稳定）。
    """
    seg = seg.set_channels(2)
    L, R = seg.split_to_mono()
    # pan>0（偏右）：左声道稍降；pan<0：右声道稍降
    # 采用 *6 dB* 最大修正，按绝对值插值
    att = 6.0 * abs(pan)
    if pan > 0:
        L = L.apply_gain(-att)
    elif pan < 0:
        R = R.apply_gain(-att)
    return AudioSegment.from_mono_audiosegments(L, R)

def main():
    assert os.path.exists(INP_JSON), f"Missing {INP_JSON}"
    assert os.path.exists(BGM_WAV),  f"Missing {BGM_WAV}"

    dialogs = json.load(open(INP_JSON, encoding="utf-8"))

    # 先构造“事件表”：每条对白的 (start_ms, end_ms, seg, page_no)
    events = []
    tcursor = 0
    used = 0
    for d in dialogs:
        wav_path = d.get("wav")
        if not wav_path or not os.path.exists(wav_path):
            continue
        seg = to_stereo_32k(AudioSegment.from_wav(wav_path))
        # 立体声定位
        pan = PAN_MAP.get(d.get("speaker"), 0.0)
        seg = simple_pan(seg, pan)
        start = tcursor
        end   = start + len(seg)
        events.append({
            "start": start,
            "end":   end,
            "seg":   seg,
            "page":  d.get("page_no")
        })
        tcursor = end + PAUSE_MS
        used += 1

    total_ms = tcursor if events else 0
    if used == 0 or total_ms < 2000:
        raise RuntimeError("对白太少或未找到生成的 wav。请先运行 tts_batch_demo.py。")

    # 基准 BGM：整段不变（只做基础增益），长度对齐
    bgm = to_stereo_32k(AudioSegment.from_wav(BGM_WAV))
    if len(bgm) < total_ms:
        rep = math.ceil(total_ms / len(bgm))
        bgm = (bgm * rep)[:total_ms]
    else:
        bgm = bgm[:total_ms]
    bgm = bgm + BGM_BASE_DB

    # 主混音：不改变 BGM 风格，仅在对白片段可选地轻度压低（DUCK_DB）
    cursor = 0
    mixed = AudioSegment.silent(duration=0, frame_rate=32000)

    # 准备翻页音效（可选）
    use_sfx = os.path.exists(PAGE_SFX)
    sfx = to_stereo_32k(AudioSegment.from_wav(PAGE_SFX)) if use_sfx else None

    for i, ev in enumerate(events):
        st, ed, seg = ev["start"], ev["end"], ev["seg"]
        # 先拼入非对白片段
        if st > cursor:
            mixed += bgm[cursor:st]
        # 白话片段：BGM 轻度降低（或不降），并淡入/淡出，再叠加人声
        B = bgm[st:ed] if DUCK_DB == 0 else (bgm[st:ed] + DUCK_DB)
        if FADE_MS > 0:
            B = B.fade_in(FADE_MS).fade_out(FADE_MS)
        mixed += B.overlay(seg)
        cursor = ed

        # （可选）页边界处插入短 SFX
        if use_sfx and i + 1 < len(events):
            p_now = ev.get("page")
            p_nxt = events[i+1].get("page")
            if p_now is not None and p_nxt is not None and p_nxt != p_now:
                # 在两页之间的停顿中间插入 SFX
                pad = bgm[cursor:cursor + PAUSE_MS] if cursor + PAUSE_MS <= len(bgm) else AudioSegment.silent(PAUSE_MS, frame_rate=32000)
                insert_pos = len(pad) // 2 - len(sfx) // 2
                if insert_pos < 0:
                    insert_pos = 0
                pad = pad.overlay(sfx - 5, position=insert_pos)  # SFX 稍微小一点
                mixed += pad
                cursor += len(pad)

    if cursor < total_ms:
        mixed += bgm[cursor:total_ms]

    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    mixed.export(OUT_WAV, format="wav")
    print("√ 导出：", OUT_WAV, "| 时长：", f"{len(mixed)/1000:.1f}s", "| 对白条数：", used)

if __name__ == "__main__":
    main()
