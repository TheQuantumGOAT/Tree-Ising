"""Regression checks for configuration validation without the optional GPU stack."""

import subprocess
import sys
from pathlib import Path

import pytest

from TreeIsing.pipeline import _bind, read_config, resolve

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config_path", sorted((ROOT / "configs").glob("*.yaml")))
def test_all_bundled_experiments_validate(config_path):
    config = read_config(config_path)
    for name in config["experiments"]:
        _bind(config, name, load=False)


def test_listing_does_not_import_gpu_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from TreeIsing.pipeline import run_treeising; "
            "run_treeising('configs/qwen35_0.8b.yaml', only='treeising_*', list_only=True); "
            "import sys; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "treeising_sa" in result.stdout
    assert "treeising_sb" in result.stdout


def test_inheritance_cycle_reports_experiments():
    with pytest.raises(ValueError, match='circular "from"'):
        resolve({"experiments": {"a": {"from": "b"}, "b": {"from": "a"}}}, "a")


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"step": "download", "repo": ["example/model"]}, "unknown repo; missing repos"),
        ({"step": "download"}, "missing repos"),
    ],
)
def test_listing_reports_bad_step_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        _bind({"experiments": {"broken": arguments}}, "broken", load=False)


def test_completed_repair_recovers_final_snapshot_without_loading_vllm(tmp_path):
    import json

    import torch

    from TreeIsing.pipeline import repair_loop

    report = {"total_s": 1.0, "rounds": [{"round": 1}]}
    (tmp_path / "rounds.json").write_text(json.dumps(report))
    weights = {15: torch.ones(2, 64, dtype=torch.bfloat16)}
    torch.save(weights, tmp_path / "current_round1.pt")

    assert repair_loop("unused", str(tmp_path), [], "unused", "unused") == report
    torch.testing.assert_close(torch.load(tmp_path / "final.pt")[15], weights[15])
