"""Main data collection: belief elicitation + statements under 4 conditions + activations.

Outputs results/main.json (texts, grades) and results/acts_<cond>.npz (activations, logprobs).
"""
import random
import time

import numpy as np
from datasets import load_from_disk

from common import (RES, ROOT, SYSTEM, TEMPLATES, alias_match, chat_prompt, extract, followup,
                    generate, jdump, load_model, set_seed)
from judge import grade_qa

CONDS = ["neutral", "inc_ally", "inc_rival", "instructed"]
N_SAMPLES = 10

set_seed()
ds = load_from_disk(f"{ROOT}/datasets/trivia_qa_nocontext")["validation"]
idx = list(range(len(ds)))
random.Random(0).shuffle(idx)
pool = idx[:2000] + idx[2300:3300]  # idx[2000:2300] was used for the prompt pilot
qs = [ds[i]["question"] for i in pool]
golds = [ds[i]["answer"]["aliases"] for i in pool]
data = {"pool": pool, "questions": qs, "golds": golds, "belief": {}, "cond": {}}

tok, model = load_model()
t0 = time.time()

# ---- neutral belief elicitation with paraphrased templates (T1 is produced in the loop below)
for t in ["T2", "T3"]:
    prompts = [chat_prompt(tok, SYSTEM["neutral"], TEMPLATES[t].format(q=q)) for q in qs]
    resp = [g[0][0] for g in generate(tok, model, prompts)]
    data["belief"][t] = {"resp": resp, "grade": grade_qa(qs, golds, resp)}
    print(t, time.time() - t0, flush=True)

# ---- statement conditions
users = [TEMPLATES["T1"].format(q=q) for q in qs]
for c in CONDS:
    prompts = [chat_prompt(tok, SYSTEM[c], u) for u in users]
    gen = generate(tok, model, prompts)
    resp, ids = [g[0][0] for g in gen], [g[0][1] for g in gen]
    acts = extract(tok, model, prompts, ids)
    fu_act, fu_diff = followup(tok, model, [SYSTEM[c]] * len(qs), users, resp)
    np.savez(f"{RES}/acts_{c}.npz", followup=fu_act, fu_diff=fu_diff, **acts)
    samples = [[s[0] for s in g] for g in generate(tok, model, prompts, max_new_tokens=16, sample=True, n=N_SAMPLES, bs=32)]
    data["cond"][c] = {
        "resp": resp,
        "grade": grade_qa(qs, golds, resp),
        "alias": [alias_match(r, g) for r, g in zip(resp, golds)],
        "samples": samples,
        "samples_alias": [[alias_match(s, g) for s in ss] for ss, g in zip(samples, golds)],
    }
    g = data["cond"][c]["grade"]
    print(c, {k: g.count(k) for k in set(g)}, time.time() - t0, flush=True)
    jdump(data, f"{RES}/main.json")
print("done", time.time() - t0)
