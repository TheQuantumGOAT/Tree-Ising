"""Calibration: forward-only statistics of the base model on its own answers to the practice (calibration) problems.

collect_statistics   teacher-forced statistics of the down_proj layers to edit. Onset file: per layer the residual means on loop
                     tokens R and normal tokens P, the down_proj input mean on R, C_P = E_P[a a^T], C_R and the counts.
                     Conclude file: the residual means on conclude tokens Z and the tokens M before them.
layer_scan           forward write score of every layer along its loop direction (paper Fig. 3), optional gradient check
loop_direction       u = unit(mu_R - mu_P) - beta unit(mu_Z - mu_M), normalised (the paper's direction)
make_labels          R/P token labels of one generated answer (the same rule as collect_statistics), for in-engine statistics"""

import json
import os
import re

import numpy as np
import torch

from .utils import INSTR, atomic_json_dump, atomic_torch_save, dup_frac, log, resolve_layers


def require_token_counts(stats, kinds, context):
    """Reject incomplete calibration data before computing token means."""
    missing = [kind for kind in kinds if stats.get("n" + kind, 0) <= 0]
    if missing:
        raise ValueError(
            "%s requires positive token counts for %s. Collect more looping and correctly completed practice answers."
            % (context, ", ".join(missing))
        )


# ----------------------------------------------------------------------------------------- activation statistics
def _load_model(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    cfg = model.config.get_text_config()
    NL, H, I = cfg.num_hidden_layers, cfg.hidden_size, cfg.intermediate_size
    layers = [m for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == NL][0]
    return tok, model, layers, H, I


def _prompt(tok, problem):
    return tok.apply_chat_template(
        [{"role": "user", "content": problem + "\n" + INSTR}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def collect_onset_stats(tok, model, layers, H, I, gen_path, probs, out_path, LAYERS=(15, 19), MAXT=8192):
    """R/P statistics (collect_onset2.py)."""
    st = {
        L: dict(
            resR=torch.zeros(H, dtype=torch.float64, device="cuda"),
            resP=torch.zeros(H, dtype=torch.float64, device="cuda"),
            aR=torch.zeros(I, dtype=torch.float64, device="cuda"),
            aP=torch.zeros(I, dtype=torch.float64, device="cuda"),
            CP=torch.zeros(I, I, dtype=torch.float64, device="cuda"),
            CR=torch.zeros(I, I, dtype=torch.float64, device="cuda"),
            nR=0,
            nP=0,
        )
        for L in LAYERS
    }
    cur = {}

    def mk_pre(L):
        def f(mod, inp):
            cur[("a", L)] = inp[0][0].float()

        return f

    def mk_post(L):
        def f(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            cur[("h", L)] = h[0].float()

        return f

    hooks = []
    for L in LAYERS:
        hooks.append(layers[L].mlp.down_proj.register_forward_pre_hook(mk_pre(L)))
        hooks.append(layers[L].register_forward_hook(mk_post(L)))
    nseq = dict(R=0, P=0)
    with torch.no_grad():
        for line in open(gen_path):
            g = json.loads(line)
            rep = g["finish"] == "length" and g["dup8"] > 0.8
            prod = g["closed"] and g["correct"] and g["finish"] == "stop"
            if not (rep or prod):
                continue
            pid = tok(_prompt(tok, probs[g["id"]]["problem"]), add_special_tokens=False).input_ids
            oid = tok(g["text"], add_special_tokens=False).input_ids
            if rep:
                oid = oid[:MAXT]
            ids = (pid + oid)[: MAXT + len(pid)]
            T, n0 = len(ids), len(pid)
            o = ids[n0:]
            if prod:
                maskP = torch.zeros(T, dtype=torch.bool)
                maskP[n0:] = True
                maskR = torch.zeros(T, dtype=torch.bool)
            else:
                seen = set()
                repv = []
                for i in range(len(o) - 7):
                    k = tuple(o[i : i + 8])
                    repv.append(1 if k in seen else 0)
                    seen.add(k)
                rv = torch.tensor(repv, dtype=torch.float)
                if rv.numel() < 600:
                    continue
                win = torch.nn.functional.avg_pool1d(rv[None, None], 256, 1)[0, 0]
                hit = (win >= 0.8).nonzero()
                if hit.numel() == 0:
                    continue
                on = int(hit[0])
                if on < 300:
                    continue
                maskR = torch.zeros(T, dtype=torch.bool)
                maskP = torch.zeros(T, dtype=torch.bool)
                r_idx = torch.arange(on, min(on + 1000, len(repv)))
                r_idx = r_idx[rv[r_idx] > 0]
                maskR[n0 + r_idx + 7] = True
                maskP[n0 + max(0, on - 2000) : n0 + on - 50] = True
            x = torch.tensor([ids], device="cuda")
            model(input_ids=x, use_cache=False)
            nseq["P" if prod else "R"] += 1
            for key, msk in (("R", maskR), ("P", maskP)):
                if msk.sum() == 0:
                    continue
                m = msk.cuda()
                for L in LAYERS:
                    a = cur[("a", L)][m].double()
                    h = cur[("h", L)][m].double()
                    s = st[L]
                    s["res" + key] += h.sum(0)
                    s["a" + key] += a.sum(0)
                    s["C" + key] += a.T @ a
                    s["n" + key] += int(m.sum())
            print(nseq, T, int(maskR.sum()), int(maskP.sum()), flush=True)
    for h in hooks:
        h.remove()
    out = {L: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in s.items()} for L, s in st.items()}
    out["nseq"] = nseq
    for L in LAYERS:
        require_token_counts(out[L], "RP", "Layer %s onset statistics" % L)
    atomic_torch_save(out, out_path)
    print("DONE", nseq, {L: (st[L]["nR"], st[L]["nP"]) for L in LAYERS}, flush=True)
    return out


def collect_end_stats(tok, model, layers, H, gen_path, probs, out_path, LAYERS=(15, 19)):
    """Z/M statistics (collect_end2.py)."""
    end_id = tok.convert_tokens_to_ids("</think>")
    st = {
        L: dict(
            resZ=torch.zeros(H, dtype=torch.float64), resM=torch.zeros(H, dtype=torch.float64), nZ=0, nM=0
        )
        for L in LAYERS
    }
    cur = {}
    hooks = [
        layers[L].register_forward_hook(
            lambda mod, inp, out, L=L: cur.__setitem__(
                L, (out[0] if isinstance(out, tuple) else out)[0].float().cpu()
            )
        )
        for L in LAYERS
    ]
    n = 0
    with torch.no_grad():
        for line in open(gen_path):
            g = json.loads(line)
            if not (g["closed"] and g["correct"] and g["finish"] == "stop"):
                continue
            pid = tok(_prompt(tok, probs[g["id"]]["problem"]), add_special_tokens=False).input_ids
            oid = tok(g["text"], add_special_tokens=False).input_ids
            if end_id not in oid:
                continue
            e = oid.index(end_id)
            if e < 200:
                continue
            ids = pid + oid[: e + 1]
            n0 = len(pid)
            model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            for L in LAYERS:
                h = cur[L].double()
                st[L]["resZ"] += h[n0 + e - 64 : n0 + e].sum(0)
                st[L]["nZ"] += 64
                st[L]["resM"] += h[n0 : n0 + e - 64].sum(0)
                st[L]["nM"] += e - 64
            n += 1
    for h in hooks:
        h.remove()
    for L in LAYERS:
        require_token_counts(st[L], "ZM", "Layer %s concluding statistics" % L)
    st["n"] = n
    atomic_torch_save(st, out_path)
    print("DONE", n, {L: (st[L]["nZ"], st[L]["nM"]) for L in LAYERS}, flush=True)
    return st


def collect_statistics(
    model_path: str,
    gen_path: str,
    problems_path: str,
    stats_path: str,
    end_stats_path: str = None,
    layers: list = (15, 19),
    max_tokens: int = 8192,
    skip_existing: bool = True,
):
    """Onset (R/P) statistics to stats_path and, if end_stats_path is given, conclude (Z/M) statistics to end_stats_path."""
    layers = tuple(resolve_layers(layers))
    todo_onset = not (skip_existing and os.path.exists(stats_path))
    todo_end = end_stats_path is not None and not (skip_existing and os.path.exists(end_stats_path))
    if not (todo_onset or todo_end):
        log.info("skip statistics (exist): %s %s", stats_path, end_stats_path)
        return
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(problems_path)}
    tok, model, mlayers, H, I = _load_model(model_path)
    if todo_onset:
        collect_onset_stats(tok, model, mlayers, H, I, gen_path, probs, stats_path, layers, max_tokens)
    if todo_end:
        collect_end_stats(tok, model, mlayers, H, gen_path, probs, end_stats_path, layers)


# ---------------------------------------------------------------------------------------------------- layer scan
def layer_scan(
    model_path: str,
    gen_path: str,
    problems_path: str,
    out_path: str,
    beta: float = 0.5,
    max_tokens: int = 8192,
    nothink: bool = False,
    grad: bool = False,
    ngrad: int = 24,
    gmax: int = 4096,
    skip_existing: bool = True,
):
    """Write scores of every layer; res['selected'] = the two top write layers, final layer excluded."""
    if skip_existing and os.path.exists(out_path):
        log.info("skip %s (exists)", out_path)
        return json.load(open(out_path))
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MAXT, BETA, NGRAD, GMAX, NOTHINK = max_tokens, beta, ngrad, gmax, nothink
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    nl = model.config.get_text_config().num_hidden_layers
    layers = [m for n, m in model.named_modules() if isinstance(m, torch.nn.ModuleList) and len(m) == nl][0]
    probs = {json.loads(l)["id"]: json.loads(l) for l in open(problems_path)}
    cur = {}
    for L in range(nl):
        layers[L].mlp.down_proj.register_forward_hook(lambda m, i, o, L=L: cur.__setitem__(("o", L), o))
        layers[L].register_forward_hook(
            lambda m, i, o, L=L: cur.__setitem__(("h", L), (o[0] if isinstance(o, tuple) else o))
        )
    emb = model.get_input_embeddings()
    emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True) if cur.get("grad") else None)
    ANS = re.compile(r"\\boxed|[Ff]inal [Aa]nswer|[Tt]he (?:final )?answer is")

    def masks_for(g):
        kw = {} if NOTHINK else {"enable_thinking": True}
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": probs[g["id"]]["problem"] + "\n" + INSTR}],
            tokenize=False,
            add_generation_prompt=True,
            **kw,
        )
        pid = tok(prompt, add_special_tokens=False).input_ids
        oid = tok(g["text"], add_special_tokens=False).input_ids[:MAXT]
        ids = pid + oid
        n0 = len(pid)
        T = len(ids)
        closed = g["closed"] or NOTHINK
        rep = g["finish"] == "length" and g["dup8"] > 0.8
        prod = closed and g["correct"] and g["finish"] == "stop"
        if rep:
            seen = set()
            repv = []
            for i in range(len(oid) - 7):
                k = tuple(oid[i : i + 8])
                repv.append(1 if k in seen else 0)
                seen.add(k)
            rv = torch.tensor(repv, dtype=torch.float)
            if rv.numel() < 600:
                return None
            win = torch.nn.functional.avg_pool1d(rv[None, None], 256, 1)[0, 0]
            hit = (win >= 0.8).nonzero()
            if hit.numel() == 0 or int(hit[0]) < 300:
                return None
            on = int(hit[0])
            mR = torch.zeros(T, dtype=torch.bool)
            r_idx = torch.arange(on, min(on + 1000, len(repv)))
            r_idx = r_idx[rv[r_idx] > 0]
            mR[n0 + r_idx + 7] = True
            mP = torch.zeros(T, dtype=torch.bool)
            mP[n0 + max(0, on - 2000) : n0 + on - 50] = True
            return "rep", ids, {"R": mR, "P": mP}, n0 + min(on + 1000, len(oid))
        if prod:
            if NOTHINK:
                ms = list(ANS.finditer(g["text"]))
                if not ms:
                    return None
                e = len(tok(g["text"][: ms[-1].start()], add_special_tokens=False).input_ids)
            else:
                end_id = tok.convert_tokens_to_ids("</think>")
                if end_id not in oid:
                    return None
                e = oid.index(end_id)
            mP = torch.zeros(T, dtype=torch.bool)
            mP[n0:] = True
            m = {"P": mP}
            if e >= 100:
                mZ = torch.zeros(T, dtype=torch.bool)
                mZ[n0 + e - 64 : n0 + e] = True
                mM = torch.zeros(T, dtype=torch.bool)
                mM[n0 : n0 + e - 64] = True
                m.update({"Z": mZ, "M": mM})
            return "prod", ids, m, T
        return None

    items = [x for x in (masks_for(json.loads(l)) for l in open(gen_path)) if x is not None]
    print(
        "items",
        sum(1 for x in items if x[0] == "rep"),
        "looping,",
        sum(1 for x in items if x[0] == "prod"),
        "correct",
        flush=True,
    )
    acc = {}

    def add(key, L, kind, v):
        k = (key, L, kind)
        if k not in acc:
            acc[k] = [0, 0]
        acc[k][0] = acc[k][0] + v.double().sum(0)
        acc[k][1] += v.shape[0]

    with torch.no_grad():
        for kind, ids, m, T in items:
            model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
            for key, mk in m.items():
                if mk.sum() == 0:
                    continue
                mk = mk.cuda()
                for L in range(nl):
                    add(key, L, "h", cur[("h", L)][0][mk].float())
                    add(key, L, "o", cur[("o", L)][0][mk].float())
    mean = lambda key, L, kind: (acc[(key, L, kind)][0] / acc[(key, L, kind)][1]).float()
    res = {"model": model_path, "gen": gen_path, "beta": BETA, "layers": {}}
    for L in range(nl):
        required = "RPZM" if BETA != 0 else "RP"
        counts = {"n" + key: acc.get((key, L, "h"), (0, 0))[1] for key in required}
        require_token_counts(counts, required, "Layer %s scan statistics" % L)
        dR = mean("R", L, "h") - mean("P", L, "h")
        u0 = dR / dR.norm()
        if BETA != 0:
            ue = mean("Z", L, "h") - mean("M", L, "h")
            ue = ue / ue.norm()
            u = u0 - BETA * ue
        else:
            u = u0
        u = u / u.norm()
        w = float(u @ (mean("R", L, "o") - mean("P", L, "o")))
        res["layers"][L] = dict(
            write=w,
            share=w / float(dR.norm()),
            loop_write=float(u0 @ (mean("R", L, "o") - mean("P", L, "o"))),
            write_R=float(u @ mean("R", L, "o")),
            dR_norm=float(dR.norm()),
        )
        if BETA != 0:
            res["layers"][L]["cos_uloop_uend"] = float(u0 @ ue)
    if grad:
        sup = {(k, L): [0.0, 0] for k in ("R", "P") for L in range(nl)}
        reps = [x for x in items if x[0] == "rep"][:NGRAD]
        for kind, ids, m, T in reps:
            end = min(T, m["R"].nonzero().max().item() + 2)
            start = max(0, end - GMAX)
            ids_s = torch.tensor([ids[start:end]], device="cuda")
            cur["grad"] = True
            out = model(input_ids=ids_s, use_cache=False)
            os_ = [cur[("o", L)] for L in range(nl)]
            for o in os_:
                o.retain_grad()
            lg = out.logits[0, :-1]
            tgt = ids_s[0, 1:]
            for key in ("R", "P"):
                mk = m[key][start:end][
                    1:
                ].cuda()  # logits row t predicts token t+1; mask marks predicted tokens
                if mk.sum() == 0:
                    continue
                for o in os_:
                    o.grad = None
                idx = mk.nonzero()[:, 0]
                lp = torch.log_softmax(lg[idx].float(), -1).gather(-1, tgt[idx, None])[:, 0]
                lp.mean().backward(retain_graph=(key == "R"))
                for L in range(nl):
                    o = os_[L]
                    s_ = (o.grad[0, :-1][mk].float() * o[0, :-1][mk].float()).sum(-1)
                    sup[(key, L)][0] += float(s_.sum())
                    sup[(key, L)][1] += int(mk.sum())
            cur["grad"] = False
            del out, lg, os_
            torch.cuda.empty_cache()
        for L in range(nl):
            res["layers"][L]["support_R"] = sup[("R", L)][0] / max(1, sup[("R", L)][1])
            res["layers"][L]["support_P"] = sup[("P", L)][0] / max(1, sup[("P", L)][1])
    rank_w = sorted(range(nl), key=lambda L: -res["layers"][L]["write"])
    res["top_write"] = rank_w[:4]
    res["selected"] = sorted(
        [L for L in rank_w if L != nl - 1][:2]
    )  # the paper's rule: top two, final layer excluded
    if grad:
        res["top_support"] = sorted(range(nl), key=lambda L: -res["layers"][L]["support_R"])[:4]
        res["top_selective"] = sorted(
            range(nl), key=lambda L: -(res["layers"][L]["support_R"] - res["layers"][L]["support_P"])
        )[:4]
    atomic_json_dump(res, out_path, indent=1)
    print("LAYERS", nl)
    for L in range(nl):
        r = res["layers"][L]
        print(L, " ".join(f"{k}={v:.4g}" for k, v in r.items()))
    print(
        "TOP_WRITE",
        res["top_write"],
        "SELECTED",
        res["selected"],
        "TOP_SUPPORT",
        res.get("top_support"),
        "TOP_SELECTIVE",
        res.get("top_selective"),
        flush=True,
    )
    return res


# ------------------------------------------------------------ loop direction and labels for in-engine statistics
def make_labels(gen, finish, closed, correct):
    """R/P labels over generated tokens, as in activations.py."""
    n = len(gen)
    if closed and correct and finish == "stop":
        return np.full(n, 2, np.int8), "P"
    rep = finish == "loop" or (finish == "length" and dup_frac(gen[-2000:]) > 0.8)
    if not rep:
        return None, None
    seen, repv = set(), []
    for i in range(n - 7):
        k = tuple(gen[i : i + 8])
        repv.append(1 if k in seen else 0)
        seen.add(k)
    if len(repv) < 600:
        return None, None
    rv = np.array(repv, dtype=np.float64)
    cs = np.concatenate([[0.0], np.cumsum(rv)])
    win = (cs[256:] - cs[:-256]) / 256.0
    hit = np.nonzero(win >= 0.8)[0]
    if len(hit) == 0 or int(hit[0]) < 300:
        return None, None
    on = int(hit[0])
    lab = np.zeros(n, np.int8)
    r_idx = np.arange(on, min(on + 1000, len(repv)))
    r_idx = r_idx[rv[r_idx] > 0]
    lab[max(0, on - 2000) : on - 50] = 2
    lab[r_idx + 7] = 1
    return lab, "R"


def loop_direction(u_stats_path, u_end_stats_path, layers, beta=0.5):
    S = torch.load(u_stats_path)
    if beta != 0 and u_end_stats_path is None:
        raise ValueError("A nonzero beta requires concluding statistics in u_end_stats_path.")
    END = torch.load(u_end_stats_path) if beta != 0 else None
    U = {}
    for L in layers:
        s = S[L]
        require_token_counts(s, "RP", "Layer %s loop direction" % L)
        ul = (s["resR"] / s["nR"]) - (s["resP"] / s["nP"])
        ul = ul / ul.norm()
        if beta != 0:
            e = END[L]
            require_token_counts(e, "ZM", "Layer %s concluding direction" % L)
            ue = (e["resZ"] / e["nZ"]) - (e["resM"] / e["nM"])
            ue = ue / ue.norm()
            u = ul - beta * ue
        else:
            u = ul
        U[L] = (u / u.norm()).float()
    return U
