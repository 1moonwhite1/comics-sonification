# -*- coding: utf-8 -*-
"""
列出 XTTS‑v2 模型内置的 speaker 名单，并保存到 outputs/speakers.txt
在 Anaconda Prompt 中运行：
    python src\list_xtts_speakers.py
"""

import os
import sys
import json

# 尽量不用 pkg_resources，避免将来弃用警告
try:
    import importlib.metadata as ilmd
except Exception:
    ilmd = None

# --- 打印环境信息 ---
def env_info():
    info = {}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception as e:
        info["torch_error"] = str(e)

    if ilmd:
        for pkg in ["TTS", "torchaudio", "transformers"]:
            try:
                info[pkg] = ilmd.version(pkg)
            except Exception:
                pass
    return info

def main():
    print("≈≈≈ 环境信息 ≈≈≈")
    info = env_info()
    for k, v in info.items():
        print(f"{k}: {v}")
    print()

    # --- 加载模型 ---
    from TTS.api import TTS

    # 避免使用已弃用的 gpu= 参数：用 .to(device)
    device = "cuda" if info.get("cuda_available") else "cpu"
    print(f"Loading XTTS‑v2 on device: {device} ...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    try:
        tts.to(device)
    except Exception:
        # 某些版本未提供 .to()；忽略
        pass

    # --- 获取说话人列表（多条路径，保证兼容） ---
    speakers = []
    err = None
    try:
        # 1) 主路径（XTTS‑v2 常见）
        speakers = list(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
    except Exception as e1:
        err = e1
        try:
            # 2) 某些版本提供 tts.speakers 属性
            speakers = list(getattr(tts, "speakers", []))
        except Exception as e2:
            err = (e1, e2)

    # --- 输出结果 ---
    os.makedirs("outputs", exist_ok=True)
    out_txt = os.path.join("outputs", "speakers.txt")
    if speakers:
        print(f"可用说话人数：{len(speakers)}")
        for i, name in enumerate(speakers):
            # 终端打印前 50 个；完整清单写文件
            if i < 50:
                print(f"{i:02d}: {name}")
        with open(out_txt, "w", encoding="utf-8") as f:
            for i, name in enumerate(speakers):
                f.write(f"{i:04d}\t{name}\n")
        print(f"\n✅ 完整清单已写入：{out_txt}")
        print("提示：在 TTS 合成时，将 speaker 参数设为上述任意一个名字即可。")
    else:
        print("⚠️ 未能获取到内置说话人清单。")
        if err:
            print("可能原因及原始错误：", repr(err))
        print("解决思路：")
        print("  1) 确认模型已成功下载（首次调用会自动下载到缓存目录）。")
        print("  2) 重试加载：TTS('tts_models/multilingual/multi-dataset/xtts_v2')。")
        print("  3) 退路：使用 speaker_wav=<你的 3-5 秒参考音频> 直接克隆发声。")

if __name__ == "__main__":
    main()
