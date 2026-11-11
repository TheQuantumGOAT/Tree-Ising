"""General helpers: logging, configs, seeding, prompts and grading, checkpoints and statistics I/O, state-tree fields,
model download, problem files and splits."""

import hashlib
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from contextlib import contextmanager

import numpy as np
import torch
import yaml

try:
    from colorama import Fore, init

    init(autoreset=True)
    _FMT = f"{Fore.CYAN}%(asctime)s {Fore.YELLOW}[%(levelname)s] {Fore.RESET}%(message)s"
except ImportError:  # colorama only colours the log
    _FMT = "%(asctime)s [%(levelname)s] %(message)s"

logging.basicConfig(format=_FMT, level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("TreeIsing")


# The prompt and the sampling profiles of the agent's evaluation protocol (MIT exp8, code/evaluate.py, mathqa.py)
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
PROFILES = {
    "mit": dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5),
    "card": dict(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5),
    "gemma": dict(temperature=1.0, top_p=0.95, top_k=64),
    "qwen3": dict(temperature=0.6, top_p=0.95, top_k=20),
    "r1": dict(temperature=0.6, top_p=0.95),
    "llama": dict(temperature=0.6, top_p=0.9),
    "greedy": dict(temperature=0.0),
}


# ----------------------------------------------------------------------------------------------------- configs
def seed_all(seed=0):
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def ensure_parent(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


@contextmanager
def atomic_output(path, mode="w"):
    """Publish an output file only after its writer has finished successfully."""
    ensure_parent(path)
    directory = os.path.dirname(os.path.abspath(path))
    kwargs = {} if "b" in mode else {"encoding": "utf-8"}
    handle = tempfile.NamedTemporaryFile(mode=mode, dir=directory, delete=False, **kwargs)
    try:
        with handle:
            yield handle
        os.replace(handle.name, path)
    finally:
        if os.path.exists(handle.name):
            os.unlink(handle.name)


def atomic_json_dump(value, path, indent=1):
    with atomic_output(path) as handle:
        json.dump(value, handle, indent=indent)


def atomic_torch_save(value, path):
    with atomic_output(path, mode="wb") as handle:
        torch.save(value, handle)


def chat_prompt(tok, problem, nothink=False):
    return tok.apply_chat_template(
        [{"role": "user", "content": problem + "\n" + INSTR}],
        tokenize=False,
        add_generation_prompt=True,
        **({} if nothink else {"enable_thinking": True}),
    )


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        i = s.rfind("\\fbox")
        if i < 0:
            return None
    j = s.find("{", i)
    if j < 0:
        return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0:
                return s[j + 1 : k]
    return None


def is_corr(gold, p):
    from math_verify import parse, verify

    try:
        return bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + p + "}")))
    except Exception:
        return False


def grade(text, gold, nothink=False):
    """Last boxed answer after </think> (thinking mode) graded with Math-Verify; returns (correct, prediction)."""
    from math_verify import parse, verify

    ans_part = text if nothink else (text.split("</think>")[-1] if "</think>" in text else None)
    if ans_part is None:
        return False, None
    p = last_boxed(ans_part)
    if p is None:
        return False, None
    try:
        return bool(verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + p + "}"))), p
    except Exception:
        return False, p


def dup8(tok_ids, tail=2000):
    """Share of repeated 8-grams in the last `tail` tokens (loop detector used for the statistics)."""
    t = tok_ids[-tail:]
    if len(t) < 16:
        return 0.0
    grams = [tuple(t[i : i + 8]) for i in range(len(t) - 7)]
    c = Counter(grams)
    return sum(1 for g in grams if c[g] > 1) / len(grams)


def checkpoint_shards(model_dir):
    """Select the checkpoint's declared shards, excluding stale files from older revisions."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as handle:
            weight_map = json.load(handle)["weight_map"]
        if not isinstance(weight_map, dict):
            raise ValueError("The safetensors weight_map must be a dictionary")
        names = sorted(set(weight_map.values()))
    elif os.path.isfile(os.path.join(model_dir, "model.safetensors")):
        names = ["model.safetensors"]
    else:
        names = sorted(name for name in os.listdir(model_dir) if name.endswith(".safetensors"))
    if any(
        not isinstance(name, str) or os.path.basename(name) != name or not name.endswith(".safetensors")
        for name in names
    ):
        raise ValueError("Invalid safetensors shard name in checkpoint index")
    return names


def load_shards(model_dir):
    """All safetensors shards of a checkpoint: (shard names, {shard: state}, merged state)."""
    from safetensors.torch import load_file

    sf = checkpoint_shards(model_dir)
    shards = {f: load_file(os.path.join(model_dir, f)) for f in sf}
    merged = {k: v for f in sf for k, v in shards[f].items()}
    return sf, shards, merged


def model_complete(model_dir):
    """Check the configuration and every expected safetensors shard without loading weights."""
    from safetensors import SafetensorError, safe_open

    try:
        with open(os.path.join(model_dir, "config.json")) as handle:
            if not isinstance(json.load(handle), dict):
                return False
        shard_names = checkpoint_shards(model_dir)
        if not shard_names:
            return False
        for name in shard_names:
            if os.path.basename(name) != name or not name.endswith(".safetensors"):
                return False
            with safe_open(os.path.join(model_dir, name), framework="pt") as handle:
                if not list(handle.keys()):
                    return False
    except (OSError, ValueError, KeyError, TypeError, SafetensorError):
        return False
    return True


def save_like(model_dir, out_dir, sf, shards, new):
    """Stage a complete checkpoint and publish it, restoring an existing checkpoint on failure."""
    import fcntl

    from safetensors.torch import save_file

    destination = os.path.abspath(out_dir)
    if os.path.realpath(model_dir) == os.path.realpath(destination):
        raise ValueError("out_dir must differ from the source checkpoint")
    ensure_parent(destination)
    with open(destination + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(destination) and os.listdir(destination) and not model_complete(destination):
            raise FileExistsError(
                "Incomplete checkpoint at %s. Move it aside or choose a fresh output path." % destination
            )
        stage = tempfile.mkdtemp(
            prefix="." + os.path.basename(destination) + "-", dir=os.path.dirname(destination)
        )
        backup = None
        try:
            for name in os.listdir(model_dir):
                source = os.path.join(model_dir, name)
                if os.path.isfile(source) and not name.endswith(".safetensors"):
                    shutil.copy(source, stage)
            for name in sf:
                save_file(
                    {key: new[key] for key in shards[name]},
                    os.path.join(stage, name),
                    metadata={"format": "pt"},
                )
            if not model_complete(stage):
                raise RuntimeError("The staged checkpoint is incomplete")
            if os.path.exists(destination):
                backup = tempfile.mkdtemp(
                    prefix="." + os.path.basename(destination) + "-backup-", dir=os.path.dirname(destination)
                )
                os.replace(destination, backup)
            try:
                os.replace(stage, destination)
            except OSError:
                if backup is not None:
                    os.replace(backup, destination)
                raise
            if backup is not None:
                shutil.rmtree(backup)
        finally:
            shutil.rmtree(stage, ignore_errors=True)


def down_proj_key(state, layer):
    """Name of the down_proj weight of a decoder layer (language model, not the MTP head)."""
    exact = "model.language_model.layers.%d.mlp.down_proj.weight" % layer
    if exact in state:
        return exact
    keys = [k for k in state if k.endswith("layers.%d.mlp.down_proj.weight" % layer) and "mtp" not in k]
    if not keys:
        raise KeyError("no down_proj weight for layer %d" % layer)
    return keys[0]


def load_stats(paths):
    """Merge statistics files (per-layer dicts are merged key by key, as qubo_gen2.py / baselines.py do)."""
    if isinstance(paths, str):
        paths = paths.split(",")
    S = {}
    for p in paths:
        for k, v in torch.load(p).items():
            if isinstance(k, int):
                S.setdefault(k, {}).update(v)
            else:
                S[k] = v
    S.setdefault("nseq", None)
    return S


def resolve_layers(layers):
    """[15, 19], '15,19' or a layer-scan JSON (its 'selected' layers, ascending)."""
    if isinstance(layers, str) and layers.endswith(".json"):
        return sorted(int(x) for x in json.load(open(layers))["selected"])
    return [int(x) for x in (layers.split(",") if isinstance(layers, str) else layers)]


def download_models(repos: list, out_dir: str = "models", revisions: dict = None):
    """Download complete checkpoints at optional per-repository revisions, resuming cached files."""
    from huggingface_hub import snapshot_download

    revisions = revisions or {}
    unknown = set(revisions) - set(repos)
    if unknown:
        raise ValueError("Revisions supplied for repositories outside repos: " + ", ".join(sorted(unknown)))
    for repo in repos:
        path = os.path.join(out_dir, repo.split("/")[-1])
        revision = revisions.get(repo)
        if revision is None and model_complete(path):
            log.info("skip %s (complete)", path)
            continue
        snapshot_download(repo_id=repo, local_dir=path, revision=revision)
        if not model_complete(path):
            raise RuntimeError("Downloaded checkpoint is incomplete: " + path)
        log.info("%s -> %s", repo, path)


def tree_fields(Wb):
    """BF16 matrix -> state-tree fields per 64-weight group: sign, exponent, mantissa, node E, state D = E - exponent."""
    b = Wb.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    s, e = (b >> 15) & 1, (b >> 7) & 0xFF
    M, N = Wb.shape
    E = e.view(M, N // 64, 64).amax(-1)
    return s, e, b & 0x7F, E, E.repeat_interleave(64, 1) - e


def tree_cost(W0b, W1b):
    s0, e0, m0, E0, D0 = tree_fields(W0b)
    s1, e1, m1, E1, D1 = tree_fields(W1b)
    ch = W0b.view(torch.int16) != W1b.view(torch.int16)
    x = (W0b.view(torch.int16).to(torch.int32) ^ W1b.view(torch.int16).to(torch.int32)) & 0xFFFF
    return dict(
        changed_weights=int(ch.sum()),
        sign_flips=int((s0 != s1).sum()),
        mantissa_changes=int((m0 != m1).sum()),
        node_E_changes=int((E0 != E1).sum()),
        state_D_changes=int((D0 != D1).sum()),
        groups_touched=int(ch.view(ch.shape[0], -1, 64).any(-1).sum()),
        bit_flips=int(sum(((x >> i) & 1).sum().item() for i in range(16))),
    )


# ------------------------------------------------------------------------------------------------- problem files
def _dump(out_dir, name, rows, ensure_ascii=False):
    path = os.path.join(out_dir, name + ".jsonl")
    with atomic_output(path) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=ensure_ascii) + "\n")
    log.info("%s: %d problems -> %s", name, len(rows), path)


def _math500(out_dir, revision=None):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        "HuggingFaceH4/MATH-500",
        "test.jsonl",
        repo_type="dataset",
        local_dir=os.path.join(out_dir, "raw"),
        revision=revision,
    )
    rows = [json.loads(l) for l in open(p)]
    _dump(
        out_dir,
        "math500",
        [dict(id=r["unique_id"], problem=r["problem"], answer=r["answer"]) for r in rows],
        ensure_ascii=True,
    )
    random.seed(0)
    sub = random.sample(rows, 100)
    _dump(
        out_dir,
        "math500_sub100",
        [dict(id=r["unique_id"], problem=r["problem"], answer=r["answer"]) for r in sub],
        ensure_ascii=True,
    )


def _practice(out_dir, practice_dir):
    m = [json.loads(l) for l in open(os.path.join(practice_dir, "math_train.jsonl"))]
    g = [json.loads(l) for l in open(os.path.join(practice_dir, "gsm8k_train.jsonl"))]
    row = lambda r: dict(id=r["id"], problem=r["problem"], answer=str(r["answer"]))
    random.seed(1)
    col = random.sample(m, 192) + random.sample(g, 192)
    _dump(out_dir, "practice_collect", [row(r) for r in col], ensure_ascii=True)
    used = {r["id"] for r in col}
    m1, g1 = [r for r in m if r["id"] not in used], [r for r in g if r["id"] not in used]
    random.seed(7)
    scr = random.sample(m1, 96) + random.sample(g1, 32)
    _dump(out_dir, "practice_screen", [row(r) for r in scr], ensure_ascii=True)
    used |= {r["id"] for r in scr}
    m2, g2 = [r for r in m if r["id"] not in used], [r for r in g if r["id"] not in used]
    random.seed(11)
    random.shuffle(m2)
    random.shuffle(g2)
    _dump(out_dir, "practice_rs", [row(r) for r in m2[:1000] + g2[:1000]], ensure_ascii=True)
    _dump(out_dir, "practice_screen512", [row(r) for r in m2[1000:1384] + g2[1000:1128]], ensure_ascii=True)


PRACTICE_REPO = "https://github.com/RikiyaT/lm_understand-logs.git"


def prepare_data(
    out_dir: str = "data",
    benchmarks: list = ("math500", "gsm8k", "amc23", "minerva", "olympiad"),
    practice_dir: str = None,
    overwrite: bool = False,
    practice_revision: str = None,
    dataset_revisions: dict = None,
):
    """Writes the benchmark files and, with practice_dir (.../lm_understand-logs/exp8_math/practice; fetched by a sparse git
    clone when missing), the practice sets."""
    dataset_revisions = dataset_revisions or {}
    os.makedirs(out_dir, exist_ok=True)
    for name in benchmarks:
        expected = ("math500", "math500_sub100") if name == "math500" else (name,)
        if not overwrite and all(os.path.isfile(os.path.join(out_dir, item + ".jsonl")) for item in expected):
            log.info("skip %s (exists)", name)
            continue
        revision = dataset_revisions.get(name)
        if name == "math500":
            _math500(out_dir, revision=revision)
            continue
        from datasets import load_dataset

        if name == "amc23":
            rows = [
                {"id": f"amc23_{r['id']}", "problem": r["question"], "answer": str(r["answer"])}
                for r in load_dataset("math-ai/amc23", split="test", revision=revision)
            ]
        elif name == "minerva":
            rows = [
                {"id": f"minerva_{i}", "problem": r["question"], "answer": str(r["answer"])}
                for i, r in enumerate(load_dataset("math-ai/minervamath", split="test", revision=revision))
            ]
        elif name == "olympiad":
            rows = []
            for r in load_dataset("math-ai/olympiadbench", split="test", revision=revision):
                fa = r["final_answer"]
                fa = (
                    fa
                    if isinstance(fa, list)
                    else json.loads(fa.replace("'", '"'))
                    if fa.startswith("[")
                    else [fa]
                )
                rows.append(
                    {
                        "id": f"olympiad_{r['id']}",
                        "problem": r["question"],
                        "answer": ", ".join(str(x) for x in fa),
                    }
                )
        elif name == "gsm8k":
            rows = [
                {
                    "id": f"gsm8k_test_{i}",
                    "problem": r["question"],
                    "answer": r["answer"].split("####")[-1].strip().replace(",", ""),
                }
                for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test", revision=revision))
            ]
        else:
            raise ValueError("unknown benchmark " + name)
        _dump(out_dir, name, rows)
    if practice_dir:
        practice_files = ("practice_collect", "practice_screen", "practice_rs", "practice_screen512")
        if not overwrite and all(
            os.path.isfile(os.path.join(out_dir, item + ".jsonl")) for item in practice_files
        ):
            log.info("skip practice sets (exist)")
        else:
            if not os.path.isdir(practice_dir):
                log.info("fetching practice revision %s into %s", practice_revision or "HEAD", practice_dir)
                ensure_parent(practice_dir)
                parent = os.path.dirname(os.path.abspath(practice_dir))
                with tempfile.TemporaryDirectory(prefix=".treeising-practice-", dir=parent) as temporary:
                    repo = os.path.join(temporary, "upstream")
                    subprocess.run(["git", "init", repo], check=True)
                    subprocess.run(["git", "-C", repo, "remote", "add", "origin", PRACTICE_REPO], check=True)
                    subprocess.run(["git", "-C", repo, "config", "core.sparseCheckout", "true"], check=True)
                    subprocess.run(
                        ["git", "-C", repo, "sparse-checkout", "set", "exp8_math/practice"], check=True
                    )
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            repo,
                            "fetch",
                            "--depth",
                            "1",
                            "--filter=blob:none",
                            "origin",
                            practice_revision or "HEAD",
                        ],
                        check=True,
                    )
                    subprocess.run(["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"], check=True)
                    downloaded = os.path.join(repo, "exp8_math", "practice")
                    if not all(
                        os.path.isfile(os.path.join(downloaded, name))
                        for name in ("math_train.jsonl", "gsm8k_train.jsonl")
                    ):
                        raise FileNotFoundError(
                            "The selected practice revision is missing math_train.jsonl or gsm8k_train.jsonl"
                        )
                    os.replace(downloaded, practice_dir)
            _practice(out_dir, practice_dir)


def verify_md5(md5sums: str = "data/MD5SUMS"):
    """Check files against an md5sum list (paths relative to the current directory); raise on any mismatch."""
    bad = []
    for line in open(md5sums):
        if line.strip():
            h, path = line.split(None, 1)
            path = path.strip()
            if not os.path.exists(path) or hashlib.md5(open(path, "rb").read()).hexdigest() != h:
                bad.append(path)
    if bad:
        raise RuntimeError("md5 mismatch: " + ", ".join(bad))
    log.info("md5 ok: every file in %s", md5sums)


# -------------------------------------------------------------------------------------- repair / held-out splits
def _write(p, rows):
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if os.path.exists(p):
        if open(p).read() != text:
            raise FileExistsError("%s exists with different content" % p)
        log.info("%s exists and matches", p)
        return
    with atomic_output(p) as f:
        f.write(text)


def make_splits(
    out_dir: str = "data/heldout",
    data_dir: str = "data",
    split: list = ("math500", "gsm8k", "olympiad"),
    per_benchmark: int = 250,
    pool_name: str = "repair_pool_rl2",
):
    os.makedirs(out_dir, exist_ok=True)
    pool = []
    for name in split:
        rows = [json.loads(l) for l in open(os.path.join(data_dir, name + ".jsonl"))]
        rep = [r for r in rows if int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 2 == 0]
        held = [r for r in rows if int(hashlib.md5(r["id"].encode()).hexdigest(), 16) % 2 == 1]
        for tag, part in (("repair", rep), ("held", held)):
            _write(os.path.join(out_dir, "%s_%s.jsonl" % (name, tag)), part)
        sub = list(rep)
        random.Random(1).shuffle(sub)
        pool += sub[:per_benchmark]
        log.info(
            "%s: repair %d, held %d, in pool %d", name, len(rep), len(held), min(per_benchmark, len(rep))
        )
    random.Random(0).shuffle(pool)
    p = os.path.join(out_dir, pool_name + ".jsonl")
    _write(p, pool)
    log.info("pool %d -> %s", len(pool), p)


# ------------------------------------------------------------------------- repair-loop layers -> model directory
def build_model(layers_pt: str, out_path: str, base_model: str = "models/Qwen3.5-0.8B"):
    if model_complete(out_path):
        log.info("skip %s (complete)", out_path)
        return
    if os.path.exists(out_path) and os.listdir(out_path):
        raise FileExistsError(
            "Incomplete checkpoint at %s. Move it aside or choose a fresh output path." % out_path
        )
    W = {int(k): v for k, v in torch.load(layers_pt).items()}
    sf, shards, sd = load_shards(base_model)
    new = dict(sd)
    for L, w in W.items():
        key = down_proj_key(sd, L)
        assert sd[key].shape == w.shape and w.dtype == torch.bfloat16
        log.info("layer %d: %d weights changed vs base", L, int((sd[key] != w).sum()))
        new[key] = w.contiguous()
    save_like(base_model, out_path, sf, shards, new)
    _, _, chk = load_shards(out_path)
    for L, w in W.items():
        assert torch.equal(chk[down_proj_key(chk, L)], w)
    log.info("saved %s", out_path)


# ------------------------------------------------------------------------------------------------ loop detection
def dup_frac(t):
    if len(t) < 16:
        return 0.0
    g = [tuple(t[i : i + 8]) for i in range(len(t) - 7)]
    c = Counter(g)
    return sum(1 for x in g if c[x] > 1) / len(g)
