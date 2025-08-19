import json, itertools, os
INP  = "outputs/dialogs_subset.json"
OUTP = "outputs/dialogs_spk.json"

os.makedirs("outputs", exist_ok=True)
dialogs = json.load(open(INP, encoding="utf-8"))

cycle = itertools.cycle(["A","B"])  # Simple AB alternation
for d in dialogs:
    d["speaker"] = next(cycle)

json.dump(dialogs, open(OUTP, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("✔ 写出:", OUTP, "| 条目:", len(dialogs))
