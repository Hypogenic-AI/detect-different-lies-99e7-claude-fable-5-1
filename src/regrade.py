"""Re-grade all stored responses with the fixed judge (valid cached verdicts are reused;
empty/refused judge outputs are retried, then fall back to alias match). Run after generation."""
import re
from common import RES, jdump, jload
from judge import grade_qa, judge_many

D = jload(f"{RES}/main.json")
qs, golds = D["questions"], D["golds"]
for t in D["belief"]:
    D["belief"][t]["grade"] = grade_qa(qs, golds, D["belief"][t]["resp"])
for c in D["cond"]:
    D["cond"][c]["grade"] = grade_qa(qs, golds, D["cond"][c]["resp"])
jdump(D, f"{RES}/main.json")
print("regraded main")
