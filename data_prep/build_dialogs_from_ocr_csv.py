# -*- coding: utf-8 -*-
"""
用 COMICS ocr_file.csv + 干净图片目录，生成对白 JSON
运行：python src\build_dialogs_from_ocr_csv.py
"""
import os, csv, json, re

IMG_DIR  = r"C:\Users\MOON\outputs\panels_clean"     # 干净图目录
OCR_CSV  = r"C:\Users\MOON\data\COMICS\COMICS_ocr_file.csv"               # 你下载的 CSV 路径
OUT_JSON = r"outputs\dialogs_from_clean_images.json"
KEEP_NARRATION = False                                # True=保留旁白；False=只要对话

EXTS = {".png",".jpg",".jpeg",".webp"}
name_re = re.compile(r"(\d+)_(\d+)_(\d+)\.[a-z]+$", re.I)

def clean_text(s:str)->str:
    if s is None: return ""
    s = s.strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘","'")
    s = s.replace("—","-").replace("–","-")
    s = re.sub(r"\s+", " ", s)
    return s

def list_images(dirpath):
    m = {}
    for fn in os.listdir(dirpath):
        ext = os.path.splitext(fn)[1].lower()
        if ext in EXTS:
            m[fn.lower()] = os.path.join(dirpath, fn)
    return m

def main():
    os.makedirs("outputs", exist_ok=True)
    imgs = list_images(IMG_DIR)

    # 解析文件名 -> (comic_no,page_no,panel_no)
    key_from_img = {}
    for fn in imgs:
        m = name_re.search(fn)
        if m:
            c,p,a = map(int, m.groups())
            key_from_img[(c,p,a)] = imgs[fn]

    if not key_from_img:
        print("未在文件名中解析到 (comic_no_page_no_panel_no)。请检查文件命名。")
        return

    # 读 CSV 并按面板聚合
    by_panel = {}
    with open(OCR_CSV, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                c = int(row["comic_no"]); p = int(row["page_no"]); a = int(row["panel_no"])
            except Exception:
                continue
            if (c,p,a) not in key_from_img:
                continue  # 只保留在你的干净图目录里存在的面板

            if not KEEP_NARRATION:
                # 只要对白（dataset里对话常用 1 表示 dialogue）
                try:
                    if int(row.get("dialog_or_narration","1")) != 1:
                        continue
                except Exception:
                    pass

            text = clean_text(row.get("text",""))
            if not text:
                continue

            key = (c,p,a)
            by_panel.setdefault(key, []).append({
                "textbox_no": int(row.get("textbox_no", 0)),
                "text": text
            })

    # 构建 JSON
    out, idx = [], 0
    for key, items in sorted(by_panel.items()):
        # 按 textbox_no 排序（气泡顺序）
        items.sort(key=lambda x: x["textbox_no"])
        img_path = key_from_img[key]
        c,p,a = key
        for it in items:
            out.append({
                "id": idx,
                "comic_no": c, "page_no": p, "panel_no": a,
                "speaker": None,                # 稍后 assign_speakers 再填 A/B
                "text": it["text"],
                "img": img_path
            })
            idx += 1

    json.dump(out, open(OUT_JSON,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"图片面板数: {len(key_from_img)} | 生成对白条目: {len(out)}")
    print(f"写出: {OUT_JSON}")

if __name__ == "__main__":
    main()
