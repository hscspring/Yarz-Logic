import pnlp
import numpy as np

np.random.seed(42)

inp_file = "c3-d-dev.json"
inp_file = "c3-d-test.json"
if "dev" in inp_file:
    N = 1600
    fn = "train"
elif "test" in inp_file:
    N = 100
    fn = "test"

data = pnlp.read_json(inp_file)
res = []
lens = []
for v in data:
    dialog = "\n".join(v[0])
    q = v[1][0]["question"]
    choice = v[1][0]["choice"]
    ans = v[1][0]["answer"]
    ans_idx = choice.index(ans) + 1
    choice = [f"{i}.{v}" for (i,v) in enumerate(choice, start=1)]
    choice_text = "；".join(choice)
    inp = f"{dialog}\n\n问题：{q}\n选项：{choice_text}。\n\n请回答哪个选项最能回答给出的问题。"
    oup = ans_idx
    im = {
        "quiz": inp,
        "solution": str(oup)
    }
    res.append(im)
    lens.append(len(inp))

print(len(res))
res = [v for v in res if len(v["quiz"]) < 512]
print(len(res))
if len(res) < N:
    out = res
else:
    out = np.random.choice(res, size=N, replace=False).tolist()
pnlp.write_list_dict_to_file(f"{fn}.jsonl", out)