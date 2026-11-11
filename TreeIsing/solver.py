"""Solvers: the Tree-Ising QUBO edit and its continuous baselines.

Each output row of an edited down_proj is a QUBO over K candidate columns and the tree-local moves f (w -> f w changes only sign and
state bits of the BF16 state tree). dose = u^T Delta mu_R, collateral = E_P ||Delta a||^2; gamma is bisected so the dose meets the
target and mu (sparsity) doubles while the edit is over budget.

sa, sb_rows                      simulated annealing / simulated bifurcation (+ float64 single-flip descent) of a batch of rows
tree_ising_edit                  the paper's edit: candidates, moves, dose calibration, solve, report (reference or own-write dose)
LayerProblem, calibrate          the warm-started GPU solver of the repair loop (same QUBO, 256-row batches)
continuous_edit                  dense baselines of the same objective (closed-form rank one, ridge, directional ablation)"""

import json
import math
import os
import time

import torch

from .calibration import require_token_counts
from .utils import (
    atomic_json_dump,
    down_proj_key,
    load_shards,
    load_stats,
    log,
    model_complete,
    resolve_layers,
    save_like,
    seed_all,
    tree_cost,
    tree_fields,
)

# ------------------------------------------------------------------------------------- annealing and bifurcation
FACT_EXT = [-1.0, 2.0, 0.5, 4.0, -2.0, -4.0, -0.5, 0.25]
FACT_BASIC = [-1.0, 2.0, 0.5]


def sa(c, Qoff, R=16, sweeps=300, greedy=False, dev="cuda"):
    """min c.x + x^T Qoff x over x in {0,1}^K for each of B rows; c [B,K], Qoff [B,K,K] symmetric, zero diagonal."""
    if c.numel() == 0 or R < 1 or sweeps < 2:
        raise ValueError("Use nonempty problems, positive replicas and at least two sweeps.")
    B, K_ = c.shape
    x = torch.zeros(B, R, K_, device=c.device, dtype=c.dtype)
    F = torch.zeros_like(x)
    scale = c.abs().median(dim=1).values.clamp(min=1e-30)
    T0, T1 = (1e-12 * scale, 1e-12 * scale) if greedy else (2.0 * scale, 1e-4 * scale)
    for sw in range(sweeps):
        T = (T0 * (T1 / T0) ** (sw / (sweeps - 1)))[:, None]
        for a in torch.randperm(K_).tolist():
            dE = (1 - 2 * x[..., a]) * (c[:, None, a] + 2 * F[..., a])
            acc = (dE < 0) | (torch.rand_like(dE) < torch.exp(-dE.clamp(min=0) / T))
            delta = acc.float() * (1 - 2 * x[..., a])
            x[..., a] += delta
            F += delta[..., None] * Qoff[:, None, a, :]
    E = (c[:, None, :] * x).sum(-1) + (x * F).sum(-1)
    best = E.argmin(1)
    return x[torch.arange(B, device=c.device), best], E.min(1).values


SB_CONFIG = {
    "variant": "dsb",
    "preconditioning": "global",
    "steps": 1200,
    "time_step": 0.5,
    "coupling_scale": 1.0,
    "initial_amplitude": 0.05,
    "pump_start": 0.0,
    "ramp_fraction": 0.25,
    "heating": 0.0,
    "sample_interval": 20,
}


def _validate(c, coupling, replicas, iterations):
    if c.ndim != 2 or coupling.shape != (c.shape[0], c.shape[1], c.shape[1]):
        raise ValueError("Expected c[B,K] and coupling[B,K,K].")
    if c.numel() == 0 or replicas < 1 or iterations < 2:
        raise ValueError("Use nonempty problems, positive replicas and at least two iterations.")
    if c.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use float32 or float64 coefficients.")
    if c.device != coupling.device or c.dtype != coupling.dtype:
        raise ValueError("Coefficients must share a device and dtype.")
    if not torch.isfinite(c).all() or not torch.isfinite(coupling).all():
        raise ValueError("Coefficients must be finite.")
    if not torch.equal(coupling, coupling.transpose(-1, -2)):
        raise ValueError("The coupling matrix must be symmetric.")
    if torch.count_nonzero(torch.diagonal(coupling, dim1=-2, dim2=-1)):
        raise ValueError("Fold the coupling diagonal into c first.")


def qubo_energies(c, coupling, candidates):
    fields = torch.bmm(candidates, coupling)
    return (c[:, None, :] * candidates).sum(-1) + (candidates * fields).sum(-1)


@torch.no_grad()
def sb_candidates(c, coupling, config, replicas=16):
    steps = config.get("steps", 1000)
    _validate(c, coupling, replicas, steps)
    dt = config.get("time_step", 0.5)
    strength = config.get("coupling_scale", 1.0)
    heating = config.get("heating", 0.0)
    variant = config.get("variant", "dsb")
    amplitude = config.get("initial_amplitude", 0.05)
    pump_start = config.get("pump_start", 0.0)
    ramp_fraction = config.get("ramp_fraction", 1.0)
    stride = config.get("sample_interval", 20)
    if variant not in ("dsb", "bsb"):
        raise ValueError("variant must be dsb or bsb")
    if config.get("preconditioning", "global") not in ("global", "diagonal_mass"):
        raise ValueError("preconditioning must be global or diagonal_mass")
    if stride < 1:
        raise ValueError("sample_interval must be positive")
    fields = -0.5 * (c + coupling.sum(-1))
    bound = torch.maximum(0.5 * coupling.abs().sum(-1).amax(-1), fields.abs().amax(-1))
    scale = strength / torch.where(bound > 0, bound, torch.ones_like(bound))
    if config.get("preconditioning", "global") == "diagonal_mass":
        mass = 0.5 * coupling.abs().sum(-1) + fields.abs()
        mass = torch.maximum(mass, bound[:, None] * 1e-6)
        mass = torch.where(mass > 0, mass, torch.ones_like(mass))
        force_scale = strength / mass[:, None]
    else:
        force_scale = scale[:, None, None]
    shape = (len(c), replicas, c.shape[1])
    position = torch.empty(shape, device=c.device, dtype=c.dtype)
    if config.get("initialization", "zero_edit") == "zero_edit":
        position.uniform_(0, amplitude).sub_(1)
    else:
        position.uniform_(-amplitude, amplitude)
    momentum = torch.empty_like(position).uniform_(-amplitude, amplitude)
    best = torch.zeros_like(position)
    best_energy = torch.zeros(shape[:2], device=c.device, dtype=c.dtype)
    for step in range(steps):
        fraction = min(1.0, step / max(1, int((steps - 1) * ramp_fraction)))
        pump = pump_start + (1 - pump_start) * fraction
        interaction = position
        if variant == "dsb":
            interaction = (position >= 0).to(position.dtype).mul_(2).sub_(1)
        force = -0.5 * (c[:, None] + torch.bmm(interaction + 1, coupling)) * force_scale
        previous_momentum = momentum.clone() if heating else None
        momentum += dt * ((pump - 1) * position + force)
        position += dt * momentum
        outside = position.abs() > 1
        position.clamp_(-1, 1)
        momentum.masked_fill_(outside, 0)
        if heating:
            momentum += heating * dt * previous_momentum
        if step % stride == 0 or step == steps - 1:
            bits = (position >= 0).to(position.dtype)
            values = qubo_energies(c, coupling, bits)
            improved = values < best_energy
            best = torch.where(improved[..., None], bits, best)
            best_energy = torch.minimum(values, best_energy)
    return best, best_energy


@torch.no_grad()
def single_flip_descent(c, coupling, candidates, max_flips=1024):
    x = candidates.clone()
    fields = torch.bmm(x, coupling)
    row_indices = torch.arange(len(c), device=c.device)[:, None]
    tolerance = torch.finfo(c.dtype).eps * 64 * (c.abs().amax(-1) + coupling.abs().sum(-1).amax(-1))
    for step in range(max_flips):
        differences = (1 - 2 * x) * (c[:, None] + 2 * fields)
        improvement, index = differences.min(-1)
        accepted = improvement < -tolerance[:, None]
        if not accepted.any():
            break
        current = x.gather(-1, index[..., None]).squeeze(-1)
        change = (1 - 2 * current) * accepted
        x.scatter_add_(-1, index[..., None], change[..., None])
        fields += change[..., None] * coupling[row_indices, index]
        if step % 32 == 31:
            fields = torch.bmm(x, coupling)
    return x, qubo_energies(c, coupling, x)


def best_candidates(c, coupling, candidates):
    values = qubo_energies(c, coupling, candidates)
    index = values.argmin(-1)
    rows = torch.arange(len(c), device=c.device)
    return candidates[rows, index], values[rows, index]


def sb_rows(c, Qoff, R=16, chunk=32):
    """SB on the GPU, then float64 single-flip descent on the CPU, rows in chunks of 32 (as in the paper's SB run)."""
    sel = []
    for st in range(0, c.shape[0], chunk):
        cc, qq = c[st : st + chunk], Qoff[st : st + chunk]
        cnd, _ = sb_candidates(cc, qq, SB_CONFIG, replicas=R)
        c64, q64, k64 = cc.cpu().double(), qq.cpu().double(), cnd.cpu().double()
        k64, _ = single_flip_descent(c64, q64, k64, max_flips=1024)
        s_, _ = best_candidates(c64, q64, k64)
        sel.append(s_.to(device=c.device, dtype=c.dtype))
    return torch.cat(sel)


# ------------------------------------------------------------------------------------------- the Tree-Ising edit
def solve_layer(
    W,
    Wb,
    u,
    dvec,
    CP,
    target,
    K,
    FACT,
    mu,
    budget,
    solver="sa",
    greedy=False,
    tree_local=True,
    dev="cuda",
    tag="",
):
    """Candidate neurons, tree-local move costs, then the gamma / mu calibration of the paper. Returns D, x, mi, gamma, mu_rel, cand."""
    dose = lambda D: float(u @ D @ dvec)
    cj = u @ W
    score = cj * dvec
    cand = score.topk(K).indices
    Wc = W[:, cand]  # [out, K]
    h_unit = u[:, None] * dvec[cand][None, :]  # dose per unit change of each weight
    moves = torch.stack(
        [(f - 1.0) * Wc for f in FACT]
    )  # w -> f*w : sign flip and/or exponent shift   [M, out, K]
    Cdiag = torch.diagonal(CP)[cand]
    _, e_all, _, E_all, _ = tree_fields(Wb.to(dev))
    gap = (E_all.repeat_interleave(64, 1) - e_all)[:, cand]
    bads = []
    for f in FACT:
        k = int(round(math.log2(abs(f))))
        if k > 0:
            bads.append(gap < k)
        elif k < 0:
            bads.append((gap < 1) | (gap - k > 14))
        else:
            bads.append(torch.zeros_like(gap, dtype=torch.bool))
    bad = torch.stack(bads)
    if not tree_local:
        bad = torch.zeros_like(bad)
    Cc = CP[cand][:, cand]

    def solve(gamma, mu_, R=4, sweeps=60):
        lin = gamma * moves * h_unit[None] + moves**2 * Cdiag[None, None, :] + mu_
        lin = torch.where(bad, torch.full_like(lin, 1e30), lin)
        mi = lin.argmin(0)  # best single move per weight
        d = moves.gather(0, mi[None]).squeeze(0)  # [out, K]
        c = lin.gather(0, mi[None]).squeeze(0)
        Qoff = d[:, :, None] * d[:, None, :] * Cc[None]
        Qoff = Qoff - torch.diag_embed(torch.diagonal(Qoff, dim1=1, dim2=2))
        if solver == "sb":
            x = sb_rows(c, Qoff, R=R)
        else:
            x, _ = sa(c, Qoff, R=R, sweeps=sweeps, greedy=greedy, dev=dev)
        D = torch.zeros_like(W)
        D[:, cand] = x * d
        return D, x, mi

    mu_rel = mu
    for attempt in range(6):
        mu_abs = mu_rel * float((moves[:3] ** 2 * Cdiag[None, None, :]).median())
        lo, hi = 1e-6, 1e6
        best = None
        for it in range(28):
            g = math.sqrt(lo * hi)
            D, x, mi = solve(g, mu_abs)
            dz = dose(D)
            if dz <= target:
                hi = g
                best = (g, D, x, mi)
            else:
                lo = g
            if hi / lo < 1.05:
                break
        if best is None:
            best = (hi,) + solve(hi, mu_abs)
        g = best[0]
        D, x, mi = solve(g, mu_abs, R=16, sweeps=300)
        if dose(D) > target:
            D, x, mi = best[1], best[2], best[3]
        nch = int((D != 0).sum())
        print(
            tag,
            "attempt",
            attempt,
            "mu_rel",
            mu_rel,
            "changed",
            nch,
            "dose",
            dose(D),
            "target",
            target,
            flush=True,
        )
        if nch <= budget:
            break
        mu_rel *= 2.0
    return D, x, mi, g, mu_rel, cand


def tree_ising_edit(
    model_path: str,
    stats_path: str,
    out_path: str,
    report_path: str = None,
    end_stats_path: str = None,
    layers: list = (15, 19),
    dose_rule: str = "reference",
    reference_model: str = None,
    dose_scale: float = 1.6,
    dose_mult: float = 6.5,
    K: int = 512,
    beta: float = 0.5,
    mu: float = 2.0,
    budget: int = 65536,
    moves: str = "ext",
    objective: str = "R",
    u_source: str = "ours",
    solver: str = "sa",
    greedy: bool = False,
    tree_local: bool = True,
    seed: int = 0,
    save_model: bool = True,
    skip_existing: bool = True,
):
    """Build the edited checkpoint out_path from model_path and write a JSON report (doses, collateral, moves, tree cost).

    reference rule (paper Sec. 4): stats_path = onset statistics, end_stats_path = conclude statistics (omit for beta = 0),
        reference_model = the agent's checkpoint, target = dose_scale x its dose along u; objective R or contrast.
    own_write rule (paper Sec. 5): target = -dose_mult x |u^T W mu_R|."""
    choices = {
        "dose_rule": (dose_rule, ("reference", "own_write")),
        "moves": (moves, ("ext", "basic")),
        "solver": (solver, ("sa", "sb")),
        "objective": (objective, ("R", "contrast")),
        "u_source": (u_source, ("ours", "mit")),
    }
    for name, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError("%s must be one of %s" % (name, ", ".join(allowed)))
    if dose_rule == "reference" and reference_model is None:
        raise ValueError("dose_rule reference needs reference_model")
    if beta != 0 and end_stats_path is None:
        raise ValueError("A nonzero beta requires end_stats_path.")
    report_path = report_path or os.path.join(out_path, "tree_ising_report.json")
    if skip_existing and os.path.exists(report_path) and (not save_model or model_complete(out_path)):
        log.info("skip %s (exists)", out_path)
        return json.load(open(report_path))
    seed_all(seed)
    dev = "cuda"
    layers = resolve_layers(layers)
    FACT = FACT_EXT if moves == "ext" else FACT_BASIC
    sf, shards, base = load_shards(model_path)
    new = dict(base)
    report = dict(
        model_path=model_path,
        stats_path=stats_path,
        end_stats_path=end_stats_path,
        layers_edited=layers,
        dose_rule=dose_rule,
        reference_model=reference_model,
        dose_scale=dose_scale,
        dose_mult=dose_mult,
        K=K,
        beta=beta,
        mu=mu,
        budget=budget,
        moves=moves,
        objective=objective,
        u_source=u_source,
        solver=solver,
        greedy=greedy,
        tree_local=tree_local,
        seed=seed,
        SB_CONFIG=SB_CONFIG if solver == "sb" else None,
        layers={},
    )
    if dose_rule == "reference":
        S = torch.load(stats_path)
        END = torch.load(end_stats_path) if end_stats_path else None
        _, _, ref = load_shards(reference_model)
        report["nseq"] = S.get("nseq")
    elif dose_rule == "own_write":
        S = load_stats([p for p in (stats_path, end_stats_path) if p])
        report["nseq"] = S["nseq"]
    else:
        raise ValueError("dose_rule must be reference or own_write")

    for L in layers:
        key = down_proj_key(base, L)
        Wb = base[key]
        W = Wb.float().to(dev)
        s = S[L]
        require_token_counts(s, "RP", "Layer %s repair statistics" % L)
        if dose_rule == "reference":  # qubo.py / qubo_sb.py order of operations
            Wm = ref[key].float().to(dev)
            muR, muP = (s["aR"] / s["nR"]).float().to(dev), (s["aP"] / s["nP"]).float().to(dev)
            CP = (s["CP"] / s["nP"]).float().to(dev)
            u_ours = ((s["resR"] / s["nR"]) - (s["resP"] / s["nP"])).float().to(dev)
            u_ours /= u_ours.norm()
            u_loop = u_ours.clone()
            if END is not None and beta != 0.0:
                require_token_counts(END[L], "ZM", "Layer %s concluding statistics" % L)
                u_end = ((END[L]["resZ"] / END[L]["nZ"]) - (END[L]["resM"] / END[L]["nM"])).float().to(dev)
                u_end /= u_end.norm()
                u_ours = u_loop - beta * u_end
                u_ours /= u_ours.norm()
            dM = Wm - W
            cols_m = dM.abs().sum(0).nonzero().flatten()
            U_, S_, V_ = torch.linalg.svd(dM[:, cols_m], full_matrices=False)
            u_mit = U_[:, 0] * torch.sign(U_[:, 0] @ u_ours)
            u = u_ours if u_source == "ours" else u_mit
            dvec = (muR - muP) if objective == "contrast" else muR
            dose = lambda D: float(u @ D @ dvec)
            doseR = lambda D: float(u @ D @ muR)
            coll = lambda D: float(((D @ CP) * D).sum())
            rec = dict(
                key=key,
                beta=beta,
                cos_uloop_ref=float(u_loop @ u_mit),
                cos_u_ref=float(u_ours @ u_mit),
                nR=s["nR"],
                nP=s["nP"],
                base_write_R=float(u @ W @ muR),
                base_write_P=float(u @ W @ muP),
                reference=dict(
                    dose=dose(dM),
                    doseR=doseR(dM),
                    coll=coll(dM),
                    changed=int((dM != 0).sum()),
                    n_cols=int(cols_m.numel()),
                ),
            )
            target = dose_scale * rec["reference"]["dose"]
        else:  # qubo_gen2.py order of operations
            mean = lambda k: (s["res" + k] / s["n" + k]).float().to(dev)
            if beta != 0:
                require_token_counts(s, "ZM", "Layer %s concluding statistics" % L)
                u_end = mean("Z") - mean("M")
                u_end /= u_end.norm()
            u0 = mean("R") - mean("P")
            dvec = (s["aR"] / s["nR"]).float().to(dev)
            u0 /= u0.norm()
            u = u0 - beta * u_end if beta != 0 else u0
            u /= u.norm()
            CP = (s["CP"] / s["nP"]).float().to(dev)
            base_write = float(u @ W @ dvec)
            target = -dose_mult * abs(base_write)
            dose = lambda D: float(u @ D @ dvec)
            doseR = dose
            coll = lambda D: float(((D @ CP) * D).sum())
            rec = dict(
                key=key,
                beta=beta,
                cos_u0_uend=float(u0 @ u_end) if beta != 0 else None,
                base_write=base_write,
            )
        D, x, mi, g, mu_rel, cand = solve_layer(
            W,
            Wb,
            u,
            dvec,
            CP,
            target,
            K,
            FACT,
            mu,
            budget,
            solver=solver,
            greedy=greedy,
            tree_local=tree_local,
            dev=dev,
            tag=str(L),
        )
        Wn = (W + D).to(torch.bfloat16)
        Dn = Wn.float() - W
        rec["target"] = target
        rec["ours"] = dict(
            mu_rel=mu_rel,
            gamma=g,
            dose=dose(Dn),
            doseR=doseR(Dn),
            coll=coll(Dn),
            changed=int((Dn != 0).sum()),
            moves={str(f): int(((mi == i) & (x > 0)).sum()) for i, f in enumerate(FACT)},
            tree=tree_cost(Wb.to(dev), Wn),
        )
        if dose_rule == "reference":
            rec["overlap_with_reference_cols"] = int(len(set(cand.tolist()) & set(cols_m.tolist())))
            rec["reference"]["tree"] = tree_cost(Wb.to(dev), ref[key].to(dev))
            rec["coll_ratio_ours_over_reference"] = rec["ours"]["coll"] / rec["reference"]["coll"]
        report["layers"][str(L)] = rec
        new[key] = Wn.cpu().contiguous()
        print(L, json.dumps(rec), flush=True)

    if save_model:
        save_like(model_path, out_path, sf, shards, new)
        log.info("saved %s", out_path)
    atomic_json_dump(report, report_path, indent=1)
    return report


# -------------------------------------------------------------------- warm-started GPU solver of the repair loop
FACT = FACT_EXT


def _exp_fields(Wb):
    b = Wb.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (b >> 7) & 0xFF
    M, N = Wb.shape
    E = e.view(M, N // 64, 64).amax(-1)
    return e, E.repeat_interleave(64, 1)


class LayerProblem:
    """One layer: current bf16 weights Wb [out, in], muR [in], CP [in, in] (mean over P tokens), unit direction u [out]."""

    def __init__(self, Wb, muR, CP, u, K=512, dev="cuda"):
        self.Wb = Wb.to(dev)
        self.W = self.Wb.float()
        self.u, self.muR = u.float().to(dev), muR.float().to(dev)
        CP = CP.float().to(dev)
        CP = 0.5 * (CP + CP.T)
        cj = self.u @ self.W
        self.cand = (cj * self.muR).topk(K).indices
        Wc = self.W[:, self.cand]
        self.h_unit = self.u[:, None] * self.muR[self.cand][None, :]
        self.moves = torch.stack([(f - 1.0) * Wc for f in FACT])
        self.Cdiag = torch.diagonal(CP)[self.cand].contiguous()
        self.Cc = CP[self.cand][:, self.cand].contiguous()
        e, Erep = _exp_fields(self.Wb)
        gap = (Erep - e)[:, self.cand]
        bads = []
        for f in FACT:
            k = int(round(math.log2(abs(f))))
            if k > 0:
                bads.append(gap < k)
            elif k < 0:
                bads.append((gap < 1) | (gap - k > 14))
            else:
                bads.append(torch.zeros_like(gap, dtype=torch.bool))
        self.bad = torch.stack(bads)
        self.mu_unit = float((self.moves[:3] ** 2 * self.Cdiag[None, None, :]).median())
        self.n_solves = 0

    def dose(self, D):
        return float(self.u @ D @ self.muR)

    @torch.no_grad()
    def solve(self, gamma, mu_abs, R=4, chunk=256):
        self.n_solves += 1
        lin = gamma * self.moves * self.h_unit[None] + self.moves**2 * self.Cdiag[None, None, :] + mu_abs
        lin = torch.where(self.bad, torch.full_like(lin, 1e30), lin)
        mi = lin.argmin(0)
        d = self.moves.gather(0, mi[None]).squeeze(0)
        c = lin.gather(0, mi[None]).squeeze(0)
        xs = []
        for st in range(0, c.shape[0], chunk):
            dd, cc = d[st : st + chunk], c[st : st + chunk]
            q = dd[:, :, None] * dd[:, None, :] * self.Cc[None]
            q.diagonal(dim1=1, dim2=2).zero_()
            cnd, _ = sb_candidates(cc, q, SB_CONFIG, replicas=R)
            c64, q64 = cc.double(), q.double()
            k64, _ = single_flip_descent(c64, q64, cnd.double(), max_flips=1024)
            s_, _ = best_candidates(c64, q64, k64)
            xs.append(s_.to(c.dtype))
            del q, q64, k64
        x = torch.cat(xs)
        D = torch.zeros_like(self.W)
        D[:, self.cand] = x * d
        return D


def calibrate(prob, target, mu_rel=2.0, budget=65536, g0=None, tol=1.10, max_attempts=6):
    """gamma search so that dose(D) <= target (target < 0), then a 16-replica solve; mu doubles while over budget."""
    t0 = time.time()
    n0 = prob.n_solves
    for attempt in range(max_attempts):
        mu_abs = mu_rel * prob.mu_unit
        best = None
        if g0 is None:
            lo, hi = 1e-6, 1e6
        else:
            lo, hi = g0 / 4, g0 * 4
            for _ in range(8):
                D = prob.solve(hi, mu_abs)
                if prob.dose(D) <= target:
                    best = (hi, D)
                    break
                lo, hi = hi, hi * 16
            for _ in range(8):
                D = prob.solve(lo, mu_abs)
                if prob.dose(D) > target:
                    break
                hi, best, lo = lo, (lo, D), lo / 16
        for it in range(28):
            if hi / lo < tol:
                break
            g = math.sqrt(lo * hi)
            D = prob.solve(g, mu_abs)
            if prob.dose(D) <= target:
                hi, best = g, (g, D)
            else:
                lo = g
        if best is None:
            best = (hi, prob.solve(hi, mu_abs))
        g = best[0]
        D = prob.solve(g, mu_abs, R=16)
        if prob.dose(D) > target:
            D = best[1]
        nch = int((D != 0).sum())
        if nch <= budget:
            break
        mu_rel *= 2.0
        g0 = g
    return dict(
        D=D,
        gamma=g,
        mu_rel=mu_rel,
        changed=nch,
        dose=prob.dose(D),
        target=target,
        solves=prob.n_solves - n0,
        seconds=round(time.time() - t0, 2),
    )


# ------------------------------------------------------------------------------------------ continuous baselines
def _ref_dose(ref, L):
    if "layers" in ref and str(L) in ref["layers"]:
        return float(ref["layers"][str(L)]["ours"]["dose"])
    return float(ref["layers_out"][str(L)]["dose"])


def continuous_edit(
    model_path: str,
    stats_path: str,
    end_stats_path: str,
    reference_report: str,
    mode: str,
    out_path: str,
    report_path: str = None,
    layers: list = (15, 19),
    beta: float = 0.5,
    eps: float = 1e-3,
    ncols: int = 64,
    save_model: bool = True,
    skip_existing: bool = True,
):
    report_path = report_path or os.path.join(out_path, "continuous_report.json")
    if beta != 0 and end_stats_path is None:
        raise ValueError("A nonzero beta requires end_stats_path.")
    if skip_existing and os.path.exists(report_path) and (not save_model or model_complete(out_path)):
        log.info("skip %s (exists)", out_path)
        return json.load(open(report_path))
    dev = "cuda" if torch.cuda.is_available() and os.environ.get("CPU") is None else "cpu"
    layers = resolve_layers(layers)
    S = load_stats([path for path in (stats_path, end_stats_path) if path])
    ref = json.load(open(reference_report))
    sf, shards, base = load_shards(model_path)
    new = dict(base)
    report = dict(
        model_path=model_path,
        stats_path=stats_path,
        end_stats_path=end_stats_path,
        reference_report=reference_report,
        mode=mode,
        layers_edited=layers,
        beta=beta,
        eps=eps,
        ncols=ncols,
        layers={},
    )
    for L in layers:
        key = down_proj_key(base, L)
        Wb = base[key].to(dev)
        W = Wb.float()
        s = S[L]
        require_token_counts(s, "RP", "Layer %s continuous-edit statistics" % L)
        mean = lambda k: (s["res" + k] / s["n" + k]).float().to(dev)
        if beta != 0:
            require_token_counts(s, "ZM", "Layer %s concluding statistics" % L)
            u_end = mean("Z") - mean("M")
            u_end /= u_end.norm()
        u0 = mean("R") - mean("P")
        u0 /= u0.norm()
        u = u0 - beta * u_end if beta != 0 else u0
        u /= u.norm()
        mu = (s["aR"] / s["nR"]).float().to(dev)
        CP = (s["CP"] / s["nP"]).float().to(dev)
        target = _ref_dose(ref, L)
        if mode == "cont_opt":
            A = CP + eps * torch.diagonal(CP).mean() * torch.eye(CP.shape[0], device=dev)
            v = torch.linalg.solve(A.double(), mu.double()).float()
            shape = torch.outer(u, v)
        elif mode == "cont_abl":
            shape = -torch.outer(u, u @ W)
        elif mode == "mitform":
            cols = ((u @ W) * mu).topk(ncols).indices
            shape = torch.zeros_like(W)
            shape[:, cols] = -torch.outer(u, u @ W[:, cols])
        else:
            raise ValueError("unknown mode " + mode)
        shape = shape * (-1.0 / float(u @ shape @ mu))  # unit negative dose before rounding
        rdose = lambda c: float(u @ ((W + c * shape).to(torch.bfloat16).float() - W) @ mu)
        lo, hi = 0.0, abs(target)
        while rdose(hi) > target and hi < 1e6 * abs(target):
            hi *= 2.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if rdose(mid) > target:
                lo = mid
            else:
                hi = mid
        c = hi
        Wn = (W + c * shape).to(torch.bfloat16)
        Dn = Wn.float() - W
        rec = dict(
            key=key,
            target=target,
            scale=c,
            dose=float(u @ Dn @ mu),
            dose_before_rounding=float(u @ (c * shape) @ mu),
            coll=float(((Dn @ CP) * Dn).sum()),
            fro=float(Dn.norm()),
            tree=tree_cost(Wb, Wn),
        )
        report["layers"][str(L)] = rec
        new[key] = Wn.cpu().contiguous()
        print(L, json.dumps(rec), flush=True)
    if save_model:
        save_like(model_path, out_path, sf, shards, new)
        log.info("saved %s", out_path)
    atomic_json_dump(report, report_path, indent=1)
    return report
