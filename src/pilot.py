"""Pilot: yield of belief-contradicting answers for each candidate pressure prompt.

300 TriviaQA questions (disjoint from the main pool). For each system prompt, greedy answer with
template T1; judge grades correct / incorrect / evasive. Lie yield is measured on questions the
model answers correctly under the neutral prompt.
"""
import random
import sys

from datasets import load_from_disk

from common import RES, ROOT, SYSTEM, TEMPLATES, chat_prompt, generate, jdump, load_model, set_seed
from judge import grade_qa

set_seed()
ds = load_from_disk(f"{ROOT}/datasets/trivia_qa_nocontext")["validation"]
idx = list(range(len(ds)))
random.Random(0).shuffle(idx)
pilot_idx = idx[2000:2300]  # main pool is idx[:2000]
qs = [ds[i]["question"] for i in pilot_idx]
golds = [ds[i]["answer"]["aliases"] for i in pilot_idx]

tok, model = load_model()
out = {}
for name, sysm in SYSTEM.items():
    prompts = [chat_prompt(tok, sysm, TEMPLATES["T1"].format(q=q)) for q in qs]
    resp = [g[0][0] for g in generate(tok, model, prompts)]
    out[name] = {"resp": resp, "grade": grade_qa(qs, golds, resp)}
    print(name, {g: out[name]["grade"].count(g) for g in ["CORRECT", "INCORRECT", "EVASIVE"]}, flush=True)

known = [i for i, g in enumerate(out["neutral"]["grade"]) if g == "CORRECT"]
print("neutral-correct questions:", len(known))
summary = {}
for name in SYSTEM:
    g = [out[name]["grade"][i] for i in known]
    summary[name] = {k: g.count(k) / len(g) for k in ["CORRECT", "INCORRECT", "EVASIVE"]}
    print(f"{name:12s} on neutral-correct: " + "  ".join(f"{k}={v:.3f}" for k, v in summary[name].items()))
    ex = [(qs[i], out[name]["resp"][i]) for i in known if out[name]["grade"][i] != "CORRECT"][:5]
    for e in ex:
        print("    ", e)
jdump({"questions": qs, "golds": golds, "out": out, "summary": summary, "n_known": len(known)}, f"{RES}/pilot.json")
