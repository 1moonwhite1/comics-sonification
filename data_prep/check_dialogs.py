import json, os, random
PATH = "outputs/dialogs.json"
assert os.path.exists(PATH), "缺少 outputs/dialogs.json"
data = json.load(open(PATH, encoding="utf-8"))
print("条目数:", len(data))
for k in ["id","image","text"]:
    assert k in data[0], f"缺少字段 {k}"
# 打印 3 条样例
for r in random.sample(data, min(3, len(data))):
    print("id:", r["id"], "| img exists?", os.path.exists(r["image"]), "| text:", r["text"][:60])
