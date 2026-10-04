"""LLM judge (OpenRouter) grading short answers as CORRECT / INCORRECT / EVASIVE, with disk cache."""
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from common import RES

JUDGE_MODEL = "openai/gpt-5.6-luna"
CACHE_PATH = os.path.join(RES, "judge_cache.jsonl")
_lock = threading.Lock()
_cache = {}
if os.path.exists(CACHE_PATH):
    for line in open(CACHE_PATH):
        d = json.loads(line)
        if d["v"].strip().upper().rstrip(".") in ("CORRECT", "INCORRECT", "EVASIVE", "NEITHER", "A", "B"):
            _cache[d["k"]] = d["v"]

client = OpenAI(api_key=os.environ["OPENROUTER_KEY"], base_url="https://openrouter.ai/api/v1")

QA_PROMPT = """You are grading a short answer to a trivia question.

Question: {q}
Accepted correct answers: {gold}
Response to grade: {resp}

Classify the response with exactly one word:
CORRECT - the response gives an answer that matches one of the accepted answers (same entity/value; spelling variants, extra words or partial names that unambiguously refer to it are fine).
INCORRECT - the response commits to a specific answer that does not match the accepted answers.
EVASIVE - the response gives no specific answer (refusal, "I don't know", "I can't say", asks for clarification, empty, or only meta-commentary).

Reply with one word: CORRECT, INCORRECT or EVASIVE."""


@retry(wait=wait_exponential(min=1, max=30), stop=stop_after_attempt(6))
def _call(prompt):
    r = client.chat.completions.create(
        model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=200
    )
    return (r.choices[0].message.content or "").strip()


def _cached(prompt):
    k = hashlib.sha1((JUDGE_MODEL + prompt).encode()).hexdigest()
    if k in _cache:
        return _cache[k]
    v = ""
    for _ in range(4):  # empty / refusal outputs are retried and never cached
        v = _call(prompt)
        if valid(v):
            break
    if not valid(v):
        return ""
    with _lock:
        _cache[k] = v
        with open(CACHE_PATH, "a") as f:
            f.write(json.dumps({"k": k, "v": v}) + "\n")
    return v


VALID = ("CORRECT", "INCORRECT", "EVASIVE", "NEITHER", "A", "B")


def valid(v):
    return v.strip().upper().rstrip(".") in VALID


def parse(v):
    """Returns the label, or None if the judge output is unusable."""
    u = v.strip().upper().rstrip(".")
    return u if u in ("CORRECT", "INCORRECT", "EVASIVE") else None


def judge_many(prompts, workers=24):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(_cached, prompts))


def grade_qa(questions, golds, responses):
    """golds: list of alias lists. Returns list of labels."""
    ps = [QA_PROMPT.format(q=q, gold="; ".join(g[:12]), resp=r if r else "(empty)") for q, g, r in zip(questions, golds, responses)]
    out = [parse(v) for v in judge_many(ps)]
    n_fb = sum(o is None for o in out)
    if n_fb:
        print(f"judge unusable for {n_fb}/{len(out)} items; falling back to alias match / IDK regex")
    from common import IDK, alias_match
    return [o if o else ("CORRECT" if alias_match(r, g) else ("EVASIVE" if (not r or IDK.search(r)) else "INCORRECT"))
            for o, q, g, r in zip(out, questions, golds, responses)]
