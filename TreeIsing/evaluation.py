"""Evaluation: generation and grading with vLLM, benchmark suites, paired tests, edit footprints, and the multi-slot engine.

generate          one model on one problem file (agent protocol: thinking on, Math-Verify grading); JSON lines + .summary.json
evaluate_model    several models x several benchmarks (each run in its own process), paired tests against reference tags
compare           paired tests between generation files (McNemar for one completion per problem, otherwise paired z)
edit_footprint    changed weights, BF16 and state-tree bit flips of every edited down_proj (paper Table 4)
compare_edits     several edits served by ONE vLLM engine (weight slots) on the same problems and seeds
generalization_report   repair loops vs one-shot edits on problems no loop saw

The multi-slot engine (install, set_weight, run_gen, collect) patches Qwen3.5 so that each edited down_proj holds P weight copies;
requests tagged 'c<k>-...' use slot k. It needs VLLM_ENABLE_V1_MULTIPROCESSING=0 (set by compare_edits and repair_loop)."""

import json
import math
import os
import random
import re
import statistics as st
import subprocess
import sys
import time
from math import comb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    PROFILES,
    atomic_json_dump,
    atomic_output,
    chat_prompt,
    checkpoint_shards,
    dup8,
    dup_frac,
    ensure_parent,
    grade,
    is_corr,
    load_shards,
    log,
    tree_fields,
)


# ---------------------------------------------------------------------------------------- generation and grading
def lenient_pred(text):
    found = []
    for pat in (
        r"[Ff]inal [Aa]nswer\s*(?:is)?\s*[:=]?\s*\$*([^\n$]+)",
        r"[Tt]he (?:final )?answer is\s*[:=]?\s*\$*([^\n$]+)",
        r"\b[Aa]nswer\s*[:=]\s*\$*([^\n$]+)",
    ):
        found += [
            (m.start(1), m.group(1).strip().strip("*").rstrip(".").strip()) for m in re.finditer(pat, text)
        ]
    for _, t in sorted(found, reverse=True):
        if re.search(r"[0-9\\]", t):
            return t
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def grade_lenient(text, gold, good, pred):
    if good or pred is not None:
        return bool(good)
    lp = lenient_pred(text)
    if lp is None:
        return False
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", lp)
    return is_corr(gold, lp) or bool(nums and is_corr(gold, nums[-1].replace(",", "")))


def generate(
    model_path: str,
    data_path: str,
    out_path: str,
    n: int = 2,
    limit: int = 0,
    offset: int = 0,
    profile: str = "mit",
    max_tokens: int = 32768,
    gpu_util: float = 0.85,
    seed: int = 0,
    nothink: bool = False,
    lenient: bool = False,
    save_ids: bool = False,
    skip_existing: bool = True,
):
    """Answer every problem of data_path n times with the model at model_path; write graded completions to out_path."""
    if skip_existing and os.path.isfile(out_path) and os.path.isfile(out_path + ".summary.json"):
        log.info("skip %s (exists)", out_path)
        return json.load(open(out_path + ".summary.json"))
    if n < 1 or max_tokens < 1 or offset < 0 or limit < 0:
        raise ValueError("n and max_tokens must be positive, and offset and limit must be nonnegative")
    prof = PROFILES[profile]
    rows = [json.loads(l) for l in open(data_path)]
    rows = rows[offset:]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No problems selected from %s. Check offset and limit." % data_path)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_tokens + 2048,
        gpu_memory_utilization=gpu_util,
        seed=seed,
        **({} if nothink else {"limit_mm_per_prompt": {"image": 0, "video": 0}}),
        enable_prefix_caching=True,
    )
    tok = llm.get_tokenizer()
    prompts = [chat_prompt(tok, r["problem"], nothink) for r in rows]
    sp = SamplingParams(n=n, max_tokens=max_tokens, seed=seed, **prof)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = 0
    acc, cut = [], []
    ensure_parent(out_path)
    with atomic_output(out_path) as f:
        for r, o in zip(rows, outs):
            for c in o.outputs:
                ok, pred = grade(c.text, str(r["answer"]), nothink)
                ok_strict = ok
                if lenient:
                    ok = grade_lenient(c.text, str(r["answer"]), ok, pred)
                ids = list(c.token_ids)
                ntok += len(ids)
                rec = dict(
                    id=r["id"],
                    answer=r["answer"],
                    pred=pred,
                    correct=ok,
                    correct_strict=ok_strict,
                    finish=c.finish_reason,
                    ntok=len(ids),
                    closed="</think>" in c.text,
                    dup8=dup8(ids),
                    text=c.text,
                )
                if save_ids:
                    rec["ids"] = ids
                    rec["prompt_ids"] = list(o.prompt_token_ids)
                acc.append(ok)
                cut.append(c.finish_reason == "length")
                f.write(json.dumps(rec) + "\n")
    summ = dict(
        model=model_path,
        data=data_path,
        profile=profile,
        n=n,
        problems=len(rows),
        samples=len(acc),
        acc=sum(acc) / len(acc),
        cut_share=sum(cut) / len(cut),
        tokens=ntok,
        seconds=dt,
        tok_per_s=ntok / dt,
    )
    print("SUMMARY", json.dumps(summ), flush=True)
    atomic_json_dump(summ, out_path + ".summary.json")
    return summ


# ---------------------------------------------------------------------------------------------- benchmark suites
def _run_one(model_path, tag, b, out_dir, gpu_util):
    out = os.path.join(out_dir, "%s_%s.jsonl" % (tag, b["name"]))
    if os.path.isfile(out) and os.path.isfile(out + ".summary.json"):
        log.info("skip %s (exists)", out)
        return out
    cfg = dict(
        model_path=model_path,
        data_path=b["data"],
        out_path=out,
        n=b.get("n", 1),
        seed=b.get("seed", 0),
        profile=b.get("profile", "mit"),
        max_tokens=b.get("max_tokens", 32768),
        gpu_util=b.get("gpu_util", gpu_util),
        limit=b.get("limit", 0),
        nothink=b.get("nothink", False),
    )
    log.info("generate %s on %s", tag, b["name"])
    code = "import json, sys; from TreeIsing.evaluation import generate; generate(**json.loads(sys.argv[1]))"
    subprocess.run(
        [sys.executable, "-c", code, json.dumps(cfg)], check=True
    )  # own process: vLLM frees the GPU
    return out


def evaluate_model(
    benchmarks: list,
    model_path: str = None,
    tag: str = None,
    models: list = None,
    out_dir: str = "results/gen",
    gpu_util: float = 0.9,
    references: list = None,
    paired_out: str = None,
):
    """benchmarks: list of {name, data, n, seed, profile, max_tokens, limit}.
    One model (model_path, tag) or several (models: list of {path, tag}); references: tags of runs to compare every model with."""
    models = models or [dict(path=model_path, tag=tag)]
    os.makedirs(out_dir, exist_ok=True)
    for m in models:
        for b in benchmarks:
            _run_one(m["path"], m["tag"], b, out_dir, gpu_util)
    res = {}
    for m in models:
        res[m["tag"]] = {}
        for b in benchmarks:
            f = os.path.join(out_dir, "%s_%s.jsonl" % (m["tag"], b["name"]))
            s = json.load(open(f + ".summary.json"))
            r = dict(acc=s["acc"], cut_share=s["cut_share"], samples=s["samples"], paired={})
            for ref in references or []:
                g = os.path.join(out_dir, "%s_%s.jsonl" % (ref, b["name"]))
                if ref != m["tag"] and os.path.exists(g):
                    r["paired"][ref] = paired_test(f, g)
            res[m["tag"]][b["name"]] = r
            print(m["tag"], b["name"], json.dumps(r), flush=True)
    if paired_out:
        ensure_parent(paired_out)
        atomic_json_dump(res, paired_out)
    return res


# -------------------------------------------------------------------------------------------------- paired tests
def _load(path):
    d = {}
    for l in open(path):
        r = json.loads(l)
        d.setdefault(r["id"], []).append((bool(r["correct"]), r.get("finish") == "length"))
    return d


def _ids(path):
    return {json.loads(l)["id"] for l in open(path)}


def paired_test(file_a: str, file_b: str, keep: str = None, exclude: str = None):
    a, b = _load(file_a), _load(file_b)
    ks = [k for k in a if k in b]
    if keep:
        kk = _ids(keep)
        ks = [k for k in ks if k in kk]
    if exclude:
        ex = _ids(exclude)
        ks = [k for k in ks if k not in ex]
    if not ks:
        raise ValueError(
            "No shared problem IDs remain for the paired comparison of %s and %s" % (file_a, file_b)
        )
    acc = lambda d: 100 * sum(sum(c for c, _ in d[k]) / len(d[k]) for k in ks) / len(ks)
    cut = lambda d: 100 * sum(sum(x for _, x in d[k]) for k in ks) / sum(len(d[k]) for k in ks)
    out = dict(
        n=len(ks),
        acc_A=round(acc(a), 2),
        acc_B=round(acc(b), 2),
        cut_A=round(cut(a), 1),
        cut_B=round(cut(b), 1),
    )
    if all(len(a[k]) == 1 and len(b[k]) == 1 for k in ks):
        oa = sum(a[k][0][0] and not b[k][0][0] for k in ks)
        ob = sum(b[k][0][0] and not a[k][0][0] for k in ks)
        both = sum(a[k][0][0] and b[k][0][0] for k in ks)
        n = oa + ob
        m = min(oa, ob)
        out.update(
            both=both,
            only_A=oa,
            only_B=ob,
            mcnemar_p=min(1.0, 2 * sum(comb(n, i) for i in range(m + 1)) / 2**n) if n else 1.0,
        )
    else:
        df = [sum(c for c, _ in a[k]) / len(a[k]) - sum(c for c, _ in b[k]) / len(b[k]) for k in ks]
        md = st.mean(df)
        se = st.pstdev(df) / len(df) ** 0.5
        z = md / se if se > 0 else float("nan")
        out.update(
            diff=round(100 * md, 2),
            se=round(100 * se, 2),
            z=round(z, 2),
            p=round(math.erfc(abs(z) / math.sqrt(2)), 4) if se > 0 else float("nan"),
            better=sum(x > 0 for x in df),
            worse=sum(x < 0 for x in df),
        )
    return out


def compare(comparisons: list, out_path: str = None):
    """comparisons: list of {name, a, b, keep (optional), exclude (optional)}; prints one JSON line per comparison."""
    res = {}
    for c in comparisons:
        r = paired_test(c["a"], c["b"], c.get("keep"), c.get("exclude"))
        res[c["name"]] = r
        print(c["name"], json.dumps(r), flush=True)
    if out_path:
        ensure_parent(out_path)
        atomic_json_dump(res, out_path)
    return res


# ------------------------------------------------------------------------------------------------ edit footprint
def _pop(x, nb):
    return sum(((x >> i) & 1).sum().item() for i in range(nb))


def edit_footprint(base_model: str, edited_models: list, out_path: str = None):
    _, _, base = load_shards(base_model)
    rows = {}
    for d in edited_models:
        _, _, ed = load_shards(d)
        tot = dict(
            changed=0,
            bf16_bits=0,
            tree_bits=0,
            node=0,
            state_changes=0,
            state_D_changes=0,
            escape_changes=0,
            mant=0,
            sign=0,
        )
        for k in base:
            if (
                not k.endswith("mlp.down_proj.weight")
                or not torch.is_tensor(ed.get(k))
                or torch.equal(base[k], ed[k])
            ):
                continue
            W0, W1 = base[k].cuda(), ed[k].cuda()
            s0, e0, m0, E0, D0 = tree_fields(W0)
            s1, e1, m1, E1, D1 = tree_fields(W1)
            x = (W0.view(torch.int16).to(torch.int32) ^ W1.view(torch.int16).to(torch.int32)) & 0xFFFF
            tot["changed"] += int((x != 0).sum())
            tot["bf16_bits"] += _pop(x, 16)
            tot["node"] += int((E0 != E1).sum())
            tot["mant"] += int((m0 != m1).sum())
            tot["sign"] += int((s0 != s1).sum())
            tot["state_changes"] += int((D0.clamp(max=15) != D1.clamp(max=15)).sum())
            tot["state_D_changes"] += int((D0 != D1).sum())
            tot["escape_changes"] += int(((D0 > 14) != (D1 > 14)).sum())
            tot["tree_bits"] += (
                _pop(E0 ^ E1, 8)
                + _pop((D0.clamp(max=15) ^ D1.clamp(max=15)) & 0xF, 4)
                + int((s0 != s1).sum())
                + _pop(m0 ^ m1, 7)
            )
        tot["tree_bits_with_escape"] = tot["tree_bits"] + 8 * tot["escape_changes"]
        rows[os.path.basename(d.rstrip("/"))] = tot
        print(os.path.basename(d.rstrip("/")), json.dumps(tot), flush=True)
    if out_path:
        ensure_parent(out_path)
        atomic_json_dump(rows, out_path)
    return rows


# --------------------------------------------------------------------------- held-out report of the repair loops
def _per_problem(path, ids):
    d = {}
    for l in open(path):
        r = json.loads(l)
        if r["id"] in ids:
            d.setdefault(r["id"], []).append(bool(r["correct"]))
    return {k: sum(v) / len(v) for k, v in d.items()}


def _paired(a, b):
    common = [k for k in a if k in b]
    diffs = [a[k] - b[k] for k in common]
    n = len(diffs)
    if not n:
        raise ValueError("No shared problem IDs remain for the paired comparison")
    md = sum(diffs) / n
    se = (sum((x - md) ** 2 for x in diffs) / n) ** 0.5 / n**0.5
    z = md / se if se > 0 else float("nan")
    p = math.erfc(abs(z) / 2**0.5) if se > 0 else float("nan")
    return dict(n=n, diff=md, se=se, z=z, p=p)


def generalization_report(sets: list, pairs: list, out_path: str = None):
    out = {}
    for s in sets:
        ids = {json.loads(l)["id"] for l in open(s["ids"])}
        acc = {m: _per_problem(f, ids) for m, f in s["files"].items() if os.path.exists(f)}
        rec = dict(
            problems=len(ids), acc={m: sum(v.values()) / len(v) for m, v in acc.items() if v}, paired={}
        )
        print("==", s["name"], "| problems", len(ids))
        print(
            "   accuracy: "
            + ", ".join("%s %.1f%% (n=%d)" % (m, 100 * a, len(acc[m])) for m, a in rec["acc"].items())
        )
        for a, b in pairs:
            if a in acc and b in acc:
                r = _paired(acc[a], acc[b])
                rec["paired"]["%s-%s" % (a, b)] = r
                print(
                    "   %s - %s: %+.1f +- %.1f points (z %.2f, p %.3g, %d problems)"
                    % (a, b, 100 * r["diff"], 100 * r["se"], r["z"], r["p"], r["n"])
                )
        out[s["name"]] = rec
    if out_path:
        ensure_parent(out_path)
        atomic_json_dump(out, out_path)
    return out


# ---------------------------------------------------------------------------------------- multi-slot vLLM engine
LAYERS = (15, 19)
STATE = dict(
    P=1,
    mods={},
    tok=None,
    lab=None,
    collect=False,
    labels={},
    acc={},
    steps=0,
    mismatch=0,
    clipped=0,
    collected_tokens=0,
)


def cand_of(req_id):
    if req_id.startswith("c"):
        j = req_id.find("-")
        if j > 1:
            try:
                return int(req_id[1:j])
            except ValueError:
                pass
    return 0


def norm_id(req_id):
    return "-".join(req_id.split("-")[:3])


def reset_acc(dim):
    STATE["acc"] = {
        L: dict(
            aR=torch.zeros(dim, dtype=torch.float64, device="cuda"),
            aP=torch.zeros(dim, dtype=torch.float64, device="cuda"),
            CP=torch.zeros(dim, dim, dtype=torch.float64, device="cuda"),
            nR=0,
            nP=0,
        )
        for L in LAYERS
    }


def _accumulate(x, layer):
    lab = STATE["lab"][: x.shape[0]]
    acc = STATE["acc"][layer]
    for code, key in ((1, "R"), (2, "P")):
        m = lab == code
        a = x[m].double()
        n = a.shape[0]
        if n:
            acc["a" + key] += a.sum(0)
            acc["n" + key] += n
            if key == "P":
                acc["CP"] += a.T @ a
            if layer == LAYERS[0]:
                STATE["collected_tokens"] += n


@torch.library.custom_op("mcand::down", mutates_args=())
def mc_down(x: torch.Tensor, w: torch.Tensor, tok: torch.Tensor, P: int, layer: int) -> torch.Tensor:
    y = F.linear(x, w)
    if P > 1:
        T = x.shape[0]
        o = w.shape[0] // P
        y = torch.gather(y.view(T, P, o), 1, tok[:T].view(T, 1, 1).expand(T, 1, o)).reshape(T, o)
    if STATE["collect"]:
        _accumulate(x, layer)
    return y


@mc_down.register_fake
def _(x, w, tok, P, layer):
    return x.new_empty((x.shape[0], w.shape[0] // P))


class MultiDown(nn.Module):
    def __init__(self, W, P, tok, layer):
        super().__init__()
        out_f, in_f = W.shape
        self.P, self.out_f, self.layer = P, out_f, layer
        self.weight_cat = nn.Parameter(W.detach().repeat(P, 1).contiguous(), requires_grad=False)
        self.register_buffer("tok_cand", tok, persistent=False)

    def forward(self, x):
        return torch.ops.mcand.down(x, self.weight_cat, self.tok_cand, self.P, self.layer), None


def install(P, layers=LAYERS, max_tokens=65536):
    STATE["P"] = P
    from vllm.config.compilation import CompilationConfig

    if "mcand::down" not in CompilationConfig._attention_ops:
        CompilationConfig._attention_ops.append("mcand::down")
    from vllm.model_executor.models import qwen3_5

    cls = qwen3_5.Qwen3_5ForConditionalGeneration
    orig_load = cls.load_weights

    def load_weights(self, weights):
        r = orig_load(self, weights)
        dev = next(self.parameters()).device
        tok = torch.zeros(max_tokens, dtype=torch.int64, device=dev)
        STATE["tok"] = tok
        STATE["lab"] = torch.zeros(max_tokens, dtype=torch.int8, device=dev)
        for name, mod in list(self.named_modules()):
            m = re.search(r"language_model\.(?:model\.)?layers\.(\d+)\.mlp$", name)
            if m and int(m.group(1)) in layers and hasattr(mod, "down_proj"):
                L = int(m.group(1))
                new = MultiDown(mod.down_proj.weight.data, P, tok, L)
                mod.down_proj = new
                STATE["mods"][L] = new
        print("MCAND2 installed layers", sorted(STATE["mods"]), "P", P, flush=True)
        assert sorted(STATE["mods"]) == sorted(layers), "did not find all layers"
        if r is not None:
            r = set(r) | {
                n + ".down_proj.weight_cat"
                for n, mm in self.named_modules()
                if isinstance(getattr(mm, "down_proj", None), MultiDown)
            }
        return r

    cls.load_weights = load_weights

    from vllm.utils.torch_utils import async_tensor_h2d
    from vllm.v1.worker.gpu import model_runner as mr

    orig_prep = mr.GPUModelRunner.prepare_inputs

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc, num_active_loras):
        ib = orig_prep(self, scheduler_output, batch_req_state, batch_desc, num_active_loras)
        ns = np.asarray(ib.num_scheduled_tokens, dtype=np.int64)[: ib.num_reqs]
        if STATE["P"] > 1 and STATE["tok"] is not None:
            c = np.fromiter((cand_of(r) for r in ib.req_ids), dtype=np.int64, count=ib.num_reqs)
            if (c >= STATE["P"]).any():
                STATE["clipped"] += 1
                c = np.minimum(c, STATE["P"] - 1)
            t = np.repeat(c, ns)
            if int(t.shape[0]) != ib.num_tokens:
                STATE["mismatch"] += 1
            if t.shape[0]:
                async_tensor_h2d(t, out=STATE["tok"][: t.shape[0]])
            STATE["steps"] += 1
        if STATE["collect"] and STATE["lab"] is not None:
            nc = np.asarray(ib.num_computed_tokens_np, dtype=np.int64)
            parts = []
            for j, r in enumerate(ib.req_ids):
                lab = STATE["labels"].get(norm_id(r))
                if lab is None:
                    parts.append(np.zeros(ns[j], np.int8))
                else:
                    pos = np.minimum(nc[j] + np.arange(ns[j]), len(lab) - 1)
                    parts.append(lab[pos])
            t = np.concatenate(parts) if parts else np.zeros(0, np.int8)
            STATE["lab"].zero_()
            if t.shape[0]:
                async_tensor_h2d(t, out=STATE["lab"][: t.shape[0]])
        return ib

    mr.GPUModelRunner.prepare_inputs = prepare_inputs


def _shards(model_dir):
    return [os.path.join(model_dir, name) for name in checkpoint_shards(model_dir)]


def load_layers(model_dir, layers=LAYERS):
    from safetensors import safe_open

    out = {}
    for f in _shards(model_dir):
        with safe_open(f, "pt") as h:
            for k in h.keys():
                m = re.search(r"language_model\.(?:model\.)?layers\.(\d+)\.mlp\.down_proj\.weight$", k)
                if m and "mtp" not in k and int(m.group(1)) in layers:
                    out[int(m.group(1))] = h.get_tensor(k)
    return out


@torch.no_grad()
def set_weight(layer, k, W):
    m = STATE["mods"][layer]
    o = m.out_f
    m.weight_cat.data[k * o : (k + 1) * o].copy_(W.to(m.weight_cat.device, m.weight_cat.dtype))


@torch.no_grad()
def get_weight(layer, k):
    m = STATE["mods"][layer]
    o = m.out_f
    return m.weight_cat.data[k * o : (k + 1) * o].clone()


PROF = PROFILES["mit"]


def run_gen(llm, items, cap, win=256, thr=0.8, min_len=1024, every=128, sustain=1024):
    """Generation with a sustained-loop abort: stop a sample once its last 256 tokens have stayed >= 80% repeated
    8-grams for 1024 tokens (checked every 128 tokens after the first 1024)."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    eng = llm.llm_engine
    toks, fin, nxt, onset = {}, {}, {}, {}
    t0 = time.time()
    for rid, prompt, seed in items:
        eng.add_request(
            rid,
            prompt,
            SamplingParams(max_tokens=cap, seed=seed, output_kind=RequestOutputKind.DELTA, **PROF),
        )
        toks[rid], nxt[rid], onset[rid] = [], min_len, None
    while eng.has_unfinished_requests():
        ab = []
        for o in eng.step():
            rid = norm_id(o.request_id)
            if rid in fin or rid not in toks:
                continue
            toks[rid].extend(list(o.outputs[0].token_ids) if o.outputs else [])
            if o.finished:
                fin[rid] = (o.outputs[0].finish_reason, time.time() - t0)
            elif len(toks[rid]) >= nxt[rid]:
                n = len(toks[rid])
                nxt[rid] = n + every
                if dup_frac(toks[rid][-win:]) >= thr:
                    if onset[rid] is None:
                        onset[rid] = n
                    elif n - onset[rid] >= sustain:
                        ab.append(rid)
                        fin[rid] = ("loop", time.time() - t0)
                else:
                    onset[rid] = None
        if ab:
            eng.abort_request(ab)
    return toks, fin, time.time() - t0


def collect(llm, reqs):
    eng = llm.llm_engine
    for rid, p, sp in reqs:
        eng.add_request(rid, p, sp)
    res = {}
    while eng.has_unfinished_requests():
        for o in eng.step():
            if o.finished:
                res[norm_id(o.request_id)] = o
    return res


# ----------------------------------------------------------------------------------- several edits in one engine
def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def compare_edits(
    slots: list,
    out_dir: str,
    data: str = "data/practice_rs.jsonl",
    n: int = 192,
    cap: int = 8192,
    engine_model: str = "models/Qwen3.5-0.8B",
    gpu_util: float = 0.85,
):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
    from vllm import LLM

    os.makedirs(out_dir, exist_ok=True)
    loaded = []
    for s in slots:
        w = torch.load(s["path"]) if s["path"].endswith(".pt") else load_layers(s["path"])
        loaded.append((s["name"], {int(k): v for k, v in w.items()}))
    rows = [json.loads(l) for l in open(data)]
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    install(len(loaded))
    t = time.time()
    llm = LLM(
        model=engine_model,
        dtype="bfloat16",
        max_model_len=cap + 2048,
        gpu_memory_utilization=gpu_util,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_prefix_caching=True,
        max_num_seqs=512,
    )
    print("INIT %.1fs" % (time.time() - t), flush=True)
    for k, (name, w) in enumerate(loaded):
        for L in LAYERS:
            set_weight(L, k, w[L])
    torch.cuda.synchronize()
    tok = llm.get_tokenizer()
    items, meta = [], {}
    for i, r in enumerate(rows):
        p = chat_prompt(tok, r["problem"])
        for k in range(len(loaded)):
            rid = "c%d-ev-%d" % (k, i)
            items.append((rid, {"prompt": p, "cache_salt": "ev%d" % k}, 5000 + i))
            meta[rid] = (i, k)
    toks, fin, wall = run_gen(llm, items, cap)
    res = {}
    with atomic_output(os.path.join(out_dir, "samples.jsonl")) as f:
        for rid, (i, k) in meta.items():
            reason = fin.get(rid, ("missing", 0))[0]
            text = tok.decode(toks[rid], skip_special_tokens=False)
            ok = False if reason in ("loop", "missing") else grade(text, str(rows[i]["answer"]))[0]
            res[(i, k)] = dict(correct=bool(ok), finish=str(reason), ntok=len(toks[rid]))
            f.write(json.dumps(dict(id=rows[i]["id"], slot=loaded[k][0], **res[(i, k)])) + "\n")
    summ = dict(data=data, n=len(rows), cap=cap, wall_s=round(wall, 1), slots={})
    for k, (name, w) in enumerate(loaded):
        rk = [res[(i, k)] for i in range(len(rows))]
        math_k = [res[(i, k)]["correct"] for i in range(len(rows)) if rows[i]["id"].startswith("math")]
        gsm_k = [res[(i, k)]["correct"] for i in range(len(rows)) if rows[i]["id"].startswith("gsm8k")]
        summ["slots"][name] = dict(
            correct=sum(r["correct"] for r in rk),
            loop=sum(r["finish"] == "loop" for r in rk),
            length=sum(r["finish"] == "length" for r in rk),
            stop=sum(r["finish"] == "stop" for r in rk),
            mean_ntok=round(float(np.mean([r["ntok"] for r in rk])), 1),
            math_correct="%d/%d" % (sum(math_k), len(math_k)),
            gsm8k_correct="%d/%d" % (sum(gsm_k), len(gsm_k)),
        )
        if k:
            b = sum(res[(i, 0)]["correct"] and not res[(i, k)]["correct"] for i in range(len(rows)))
            c = sum(res[(i, k)]["correct"] and not res[(i, 0)]["correct"] for i in range(len(rows)))
            summ["slots"][name]["vs_" + loaded[0][0]] = dict(
                only_first=b, only_this=c, mcnemar_p=round(mcnemar_p(b, c), 4)
            )
    print("SUMMARY", json.dumps(summ), flush=True)
    atomic_json_dump(summ, os.path.join(out_dir, "summary.json"))
    return summ
