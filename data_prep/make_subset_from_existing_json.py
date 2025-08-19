# -*- coding: utf-8 -*-
import os, json, argparse, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp",  required=True, help="现有对白 JSON")
    ap.add_argument("--out",  default=r"outputs\dialogs_subset.json")
    ap.add_argument("--num",  type=int, default=6, help="抽取多少张图片")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)
    data = json.load(open(args.inp, encoding="utf-8"))

    # 找到每条记录对应的图片路径
    def get_img(d):
        for k in ("img","image","panel_img","panel_path","path"):
            if k in d and d[k]: return d[k]
        return None

    # 以图片聚合
    by_img = {}
    for d in data:
        img = get_img(d)
        if not img: continue
        by_img.setdefault(img, []).append(d)

    imgs = list(by_img.keys())
    if not imgs:
        print("❌ 在 JSON 里没找到 img 字段")
        return

    random.seed(args.seed)
    pick = set(random.sample(imgs, min(args.num, len(imgs))))

    out, idx = [], 0
    for img in imgs:
        if img not in pick: continue
        for d in by_img[img]:
            d2 = dict(d)
            d2["id"] = idx
            out.append(d2)
            idx += 1

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"✅ 抽取图片: {len(pick)} | 对白条目: {len(out)}")
    print(f"✅ 写出: {args.out}")

if __name__ == "__main__":
    main()
