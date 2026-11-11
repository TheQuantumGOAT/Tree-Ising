"""run_treeising: run the experiments of one joint YAML file, in order, each in its own process.

    run_treeising --config configs/qwen35_0.8b.yaml                    # every experiment of the file
    run_treeising --config configs/qwen35_0.8b.yaml --only edit_sa     # one experiment (comma-separated names or wildcards)
    run_treeising --config configs/qwen35_0.8b.yaml --list             # what the file contains

A config file has two sections. `vars` holds strings that any value can use as {key}; {name} is the experiment's own name.
`experiments` maps a name to {step: <step>, from: <another experiment>, ...keyword arguments of the step's function}; `from`
copies the other experiment's keys and the new keys override them. Every step skips work whose output already exists, so a file
can be re-run after an interruption. The repair loop (repair_loop) is the one multi-stage step defined here."""

import argparse
import ast
import fnmatch
import importlib
import inspect
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

log = logging.getLogger("TreeIsing")

STEPS = {
    "download": ("utils", "download_models"),  # Hugging Face checkpoints
    "data": ("utils", "prepare_data"),  # benchmark and practice problem files
    "splits": ("utils", "make_splits"),  # repair / held-out halves
    "md5": ("utils", "verify_md5"),  # check problem files against data/MD5SUMS
    "generate": ("evaluation", "generate"),  # answers of one model on one problem file
    "statistics": ("calibration", "collect_statistics"),
    "scan": ("calibration", "layer_scan"),
    "edit": ("solver", "tree_ising_edit"),
    "continuous": ("solver", "continuous_edit"),
    "evaluate": ("evaluation", "evaluate_model"),  # benchmark suite + paired tests
    "paired": ("evaluation", "compare"),
    "footprint": ("evaluation", "edit_footprint"),
    "repair": ("pipeline", "repair_loop"),
    "build": ("utils", "build_model"),  # model directory from repair-loop layers
    "compare_edits": ("evaluation", "compare_edits"),
    "generalization": ("evaluation", "generalization_report"),
}


# ----------------------------------------------------------------------------------------------- joint YAML runner
def read_config(config_path):
    """Read experiment metadata without importing the optional GPU runtime."""
    with open(config_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("experiments"), dict):
        raise ValueError("Configuration must contain an 'experiments' mapping.")
    if not isinstance(config.get("vars", {}), dict):
        raise ValueError("Configuration 'vars' must be a mapping.")
    return config


def _subst(v, ctx):
    if isinstance(v, str):
        for k, x in ctx.items():
            v = v.replace("{%s}" % k, str(x))
        return v
    if isinstance(v, list):
        return [_subst(x, ctx) for x in v]
    if isinstance(v, dict):
        return {k: _subst(x, ctx) for k, x in v.items()}
    return v


def _raw(experiments, name, seen=()):
    if name not in experiments:
        raise KeyError("no experiment %r" % name)
    if name in seen:
        raise ValueError('circular "from": %s' % " -> ".join(seen + (name,)))
    if not isinstance(experiments[name], dict):
        raise ValueError("%s: experiment must be a mapping" % name)
    e = dict(experiments[name])
    parent = e.pop("from", None)
    cfg = _raw(experiments, parent, seen + (name,)) if parent else {}
    cfg.update(e)
    return cfg


def resolve(config, name):
    """Keyword arguments of one experiment: inherited keys, then {vars} and {name} substituted."""
    ctx = {}
    for k, v in (config.get("vars") or {}).items():
        ctx[k] = _subst(v, ctx)
    ctx["name"] = name
    cfg = _subst(_raw(config["experiments"], name), ctx)
    if cfg.get("step") not in STEPS:
        raise ValueError("%s: unknown step %r (steps: %s)" % (name, cfg.get("step"), ", ".join(STEPS)))
    return cfg


def _bind(config, name, load=True):
    """The step function of one experiment and its keyword arguments (unknown keys are an error)."""
    cfg = resolve(config, name)
    module, fn_name = STEPS[cfg.pop("step")]
    if load:
        fn = getattr(importlib.import_module("TreeIsing." + module), fn_name)
        params = inspect.signature(fn).parameters
        required = {key for key, param in params.items() if param.default is inspect.Parameter.empty}
    else:
        # Listing checks the source signatures so it also works on a laptop without CUDA dependencies.
        source = Path(__file__).with_name(module + ".py").read_text(encoding="utf-8")
        definition = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == fn_name
        )
        positional = definition.args.posonlyargs + definition.args.args
        params = {arg.arg for arg in positional + definition.args.kwonlyargs}
        required = {arg.arg for arg in positional[: len(positional) - len(definition.args.defaults)]}
        required.update(
            arg.arg
            for arg, default in zip(definition.args.kwonlyargs, definition.args.kw_defaults)
            if default is None
        )
        fn = fn_name
    unknown = sorted(k for k in cfg if k not in params)
    missing = sorted(k for k in required if k not in cfg)
    if unknown or missing:
        raise ValueError(
            "%s: %s() %s"
            % (
                name,
                fn_name,
                "; ".join(
                    x
                    for x in (
                        "unknown " + ", ".join(unknown) if unknown else "",
                        "missing " + ", ".join(missing) if missing else "",
                    )
                    if x
                ),
            )
        )
    return fn, cfg


def run_experiment(config_path, name):
    fn, cfg = _bind(read_config(config_path), name)
    log.info("%s: %s(%s)", name, fn.__name__, json.dumps(cfg)[:400])
    return fn(**cfg)


def run_treeising(config_path, only=None, list_only=False):
    config = read_config(config_path)
    names = list(config["experiments"])
    if only:
        pats = [part.strip() for part in only.split(",")] if isinstance(only, str) else list(only)
        names = [n for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
        if not names:
            raise SystemExit("no experiment matches %s" % only)
    if list_only:  # also checks every experiment's arguments
        for n in names:
            fn, cfg = _bind(config, n, load=False)
            print(
                "%-22s %-12s %s"
                % (
                    n,
                    resolve(config, n)["step"],
                    cfg.get("out_path") or cfg.get("out_dir") or cfg.get("paired_out") or "",
                )
            )
        return
    for n in names:
        t = time.time()
        log.info("=== %s (%s)", n, resolve(config, n)["step"])
        subprocess.run(
            [sys.executable, "-m", "TreeIsing.pipeline", "--config", config_path, "--worker", n], check=True
        )
        log.info("=== %s done in %.0f s", n, time.time() - t)


def run_from_config():
    parser = argparse.ArgumentParser(
        description="Run the experiments of a Tree-Ising YAML configuration file."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument(
        "--only", type=str, default=None, help="Comma-separated experiment names or wildcards."
    )
    parser.add_argument("--list", action="store_true", help="List the experiments and their steps.")
    parser.add_argument("--worker", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker:
            run_experiment(args.config, args.worker)
        else:
            run_treeising(args.config, args.only, args.list)
    except (FileNotFoundError, KeyError, ValueError) as error:
        parser.error(str(error))


# ------------------------------------------------------------------------------------------------------ repair loop
def repair_loop(
    start_model: str,
    out_dir: str,
    pool: list,
    u_stats_path: str,
    u_end_stats_path: str,
    base_model: str = "models/Qwen3.5-0.8B",
    reference_model: str = "models/Qwen3.5-0.8B-weight-space-programming",
    rounds: int = 3,
    batch: int = 128,
    cap: int = 8192,
    tscales: list = (0.3, 0.6),
    budget: int = 16384,
    beta: float = 0.5,
    min_loop_tokens: int = 1000,
    gpu_util: float = 0.80,
):
    """Evolving repair loop: one vLLM engine with weight slots, in-engine statistics, warm-started GPU solve.

    Round 0: generate with the current model on a fresh batch, label R/P tokens, in-engine statistics, solve candidates.
    Round r >= 1: swap the candidates into slots 1.., generate current + candidates on a fresh batch (same seed per problem),
    accept the best candidate if it has at least as many correct answers and fewer unfinished ones (loop abort or cap),
    then statistics on the accepted model's samples and new candidates around it (target = tscale x the reference edit's dose).
    The loop direction u stays the paper's (base-model statistics, beta 0.5).
    Outputs in out_dir: rounds.json, round<r>_samples.jsonl, current_round<r>.pt and final.pt
    ({layer: bf16 down_proj}). final.pt records the accepted model even after an early stop."""
    import numpy as np
    import torch

    from . import evaluation as mc
    from . import solver as qs
    from .calibration import loop_direction, make_labels
    from .utils import atomic_json_dump, atomic_output, atomic_torch_save, chat_prompt, grade, log

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault(
        "VLLM_DISABLE_COMPILE_CACHE", "1"
    )  # the patched graph must not reuse a cached unpatched one
    LAYERS = mc.LAYERS
    TS = [float(x) for x in (tscales.split(",") if isinstance(tscales, str) else tscales)]
    P = 1 + len(TS)
    os.makedirs(out_dir, exist_ok=True)
    done = os.path.join(out_dir, "rounds.json")
    final_path = os.path.join(out_dir, "final.pt")
    if os.path.exists(done):
        try:
            with open(done) as stream:
                completed = json.load(stream)
        except json.JSONDecodeError:
            log.warning("Incomplete repair report at %s; restarting the repair", done)
        else:
            if "total_s" in completed:
                if not os.path.exists(final_path):
                    last_round = completed["rounds"][-1]["round"]
                    snapshot = os.path.join(out_dir, "current_round%d.pt" % last_round)
                    atomic_torch_save(torch.load(snapshot, map_location="cpu"), final_path)
                log.info("skip %s (finished)", out_dir)
                return completed

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    T0 = time.time()
    U = loop_direction(u_stats_path, u_end_stats_path, LAYERS, beta)
    Wb0 = mc.load_layers(base_model)
    Wref = mc.load_layers(reference_model)
    DM = {L: (Wref[L].float() - Wb0[L].float()).cuda() for L in LAYERS}
    rows_pool = []
    for f in pool.split(",") if isinstance(pool, str) else pool:
        rows_pool += [json.loads(l) for l in open(f)]
    print(
        "POOL",
        len(rows_pool),
        dict(Counter(r["id"].split("/")[0].split("_")[0] for r in rows_pool)),
        flush=True,
    )

    mc.install(P)
    t = time.time()
    llm = LLM(
        model=start_model,
        dtype="bfloat16",
        max_model_len=cap + 2048,
        gpu_memory_utilization=gpu_util,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
        enable_prefix_caching=True,
        max_num_seqs=512,
    )
    t_init = time.time() - t
    tok = llm.get_tokenizer()
    cur = mc.load_layers(start_model)
    W_start = {L: w.clone() for L, w in cur.items()}
    for k in range(P):
        for L in LAYERS:
            mc.set_weight(L, k, cur[L])
    torch.cuda.synchronize()
    print("INIT engine %.1fs, slots %d, pool %d problems" % (t_init, P, len(rows_pool)), flush=True)
    log_ = dict(
        args=dict(
            start_model=start_model,
            pool=pool,
            rounds=rounds,
            batch=batch,
            cap=cap,
            tscales=TS,
            budget=budget,
            beta=beta,
            gpu_util=gpu_util,
        ),
        engine_init_s=round(t_init, 1),
        rounds=[],
    )
    cands, gprev = {}, {}
    for r in range(rounds + 1):
        T = {}
        rec = dict(round=r)
        tr = time.time()
        bt = [rows_pool[(r * batch + i) % len(rows_pool)] for i in range(batch)]
        prompts = [chat_prompt(tok, row["problem"]) for row in bt]
        slots = [0] + sorted(cands)
        # 1. swap candidates in
        t = time.time()
        for k, ws in cands.items():
            for L, w in ws.items():
                mc.set_weight(L, k, w)
        torch.cuda.synchronize()
        llm.llm_engine.reset_prefix_cache()
        T["swap"] = time.time() - t
        # 2. generate current + candidates
        items, meta = [], {}
        for i, p in enumerate(prompts):
            for k in slots:
                rid = "c%d-g%d-%d" % (k, r, i)
                items.append((rid, {"prompt": p, "cache_salt": "g%dc%d" % (r, k)}, 1000 * r + i))
                meta[rid] = (i, k)
        toks, fin, T["generate"] = mc.run_gen(llm, items, cap)
        # 3. grade and accept
        t = time.time()
        res = {}
        for rid, (i, k) in meta.items():
            reason = fin.get(rid, ("missing", 0))[0]
            text = tok.decode(toks[rid], skip_special_tokens=False)
            ok = False if reason in ("loop", "missing") else grade(text, str(bt[i]["answer"]))[0]
            res[rid] = dict(
                i=i,
                slot=k,
                finish=str(reason),
                ntok=len(toks[rid]),
                correct=bool(ok),
                closed="</think>" in text,
            )
        per = {}
        for k in slots:
            rk = [v for v in res.values() if v["slot"] == k]
            per[k] = dict(
                correct=sum(v["correct"] for v in rk),
                unfinished=sum(v["finish"] in ("loop", "length") for v in rk),
                loop=sum(v["finish"] == "loop" for v in rk),
                length=sum(v["finish"] == "length" for v in rk),
                mean_ntok=round(float(np.mean([v["ntok"] for v in rk])), 1),
            )
        winner = 0
        for k in sorted(cands):
            if per[k]["correct"] >= per[0]["correct"] and per[k]["unfinished"] < per[0]["unfinished"]:
                if winner == 0 or (per[k]["correct"], -per[k]["unfinished"]) > (
                    per[winner]["correct"],
                    -per[winner]["unfinished"],
                ):
                    winner = k
        if winner:
            cur = {L: w.clone() for L, w in cands[winner].items()}
            for L in LAYERS:
                mc.set_weight(L, 0, cur[L])
            torch.cuda.synchronize()
        T["grade_accept"] = time.time() - t
        rec.update(per_slot=per, winner=winner, winner_tscale=(TS[winner - 1] if winner else None))
        with atomic_output(os.path.join(out_dir, "round%d_samples.jsonl" % r)) as f:
            for rid, v in res.items():
                f.write(json.dumps(dict(rid=rid, id=bt[v["i"]]["id"], **v)) + "\n")
        # 4. labels from the accepted model's samples, then in-engine statistics on slot 0
        t = time.time()
        seqs, kinds = [], Counter()
        for i, p in enumerate(prompts):
            rid = "c%d-g%d-%d" % (winner, r, i)
            v = res[rid]
            lab, kind = make_labels(toks[rid], v["finish"], v["closed"], v["correct"])
            if lab is None:
                continue
            pid = tok.encode(p, add_special_tokens=False)
            seqs.append((pid + list(toks[rid]), np.concatenate([np.zeros(len(pid), np.int8), lab])))
            kinds[kind] += 1
        T["label"] = time.time() - t
        t = time.time()
        mc.reset_acc(DM[LAYERS[0]].shape[1])
        mc.STATE["labels"] = {}
        reqs = []
        sp = SamplingParams(max_tokens=1, temperature=0.0, output_kind=RequestOutputKind.FINAL_ONLY)
        for i, (ids, lab) in enumerate(seqs):
            rid = "c0-s%d-%d" % (r, i)
            mc.STATE["labels"][rid] = lab
            reqs.append((rid, {"prompt_token_ids": ids, "cache_salt": "st%d" % r}, sp))
        llm.llm_engine.reset_prefix_cache()
        mc.STATE["collected_tokens"] = 0
        mc.STATE["collect"] = True
        mc.collect(llm, reqs)
        torch.cuda.synchronize()
        mc.STATE["collect"] = False
        acc = mc.STATE["acc"]
        n_lab = int(sum(int((lab > 0).sum()) for _, lab in seqs))
        stats = {
            L: dict(
                muR=(acc[L]["aR"] / max(acc[L]["nR"], 1)),
                CP=(acc[L]["CP"] / max(acc[L]["nP"], 1)),
                nR=acc[L]["nR"],
                nP=acc[L]["nP"],
            )
            for L in LAYERS
        }
        T["stats"] = time.time() - t
        rec.update(
            stat_sequences=dict(kinds),
            stat_tokens=sum(len(s) for s, _ in seqs),
            labelled_tokens=n_lab,
            captured=dict((L, (stats[L]["nR"], stats[L]["nP"])) for L in LAYERS),
        )
        # 5. new candidates around the current model
        t = time.time()
        cands, solves = {}, []
        if r < rounds and min(stats[L]["nR"] for L in LAYERS) >= min_loop_tokens:
            for j, ts in enumerate(TS):
                new = {}
                for L in LAYERS:
                    prob = qs.LayerProblem(cur[L], stats[L]["muR"], stats[L]["CP"], U[L])
                    target = ts * prob.dose(DM[L])
                    g0 = gprev.get((j, L), gprev.get((0, L)))
                    out = qs.calibrate(prob, target, mu_rel=2.0, budget=budget, g0=g0)
                    gprev[(j, L)] = out["gamma"]
                    new[L] = (prob.W + out["D"]).to(torch.bfloat16).cpu()
                    solves.append(
                        dict(
                            slot=1 + j,
                            tscale=ts,
                            layer=L,
                            warm=g0 is not None,
                            **{k: v for k, v in out.items() if k != "D"},
                        )
                    )
                    del prob, out
                cands[1 + j] = new
            torch.cuda.empty_cache()
        elif r < rounds:
            rec["note"] = "too few loop tokens for a new candidate; stopping"
        T["solve"] = time.time() - t
        T["round_total"] = time.time() - tr
        rec["solves"] = solves
        rec["changed_vs_start"] = {L: int((cur[L] != W_start[L]).sum()) for L in LAYERS}
        rec["changed_vs_base"] = {L: int((cur[L] != Wb0[L]).sum()) for L in LAYERS}
        rec["times_s"] = {k: round(v, 2) for k, v in T.items()}
        log_["rounds"].append(rec)
        print("ROUND", json.dumps(rec), flush=True)
        atomic_json_dump(log_, done, indent=1)
        atomic_torch_save({L: w for L, w in cur.items()}, os.path.join(out_dir, "current_round%d.pt" % r))
        if r < rounds and not cands:
            break
    atomic_torch_save({L: w for L, w in cur.items()}, final_path)
    log_["total_s"] = round(time.time() - T0, 1)
    atomic_json_dump(log_, done, indent=1)
    print("DONE total %.1fs" % log_["total_s"], flush=True)
    return log_


if __name__ == "__main__":
    run_from_config()
