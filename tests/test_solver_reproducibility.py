"""CPU regressions for solver precision and calibration failure modes."""

import json

import pytest
import torch

from TreeIsing import solver
from TreeIsing.calibration import loop_direction, require_token_counts


def test_annealing_keeps_float64_coefficients_and_energy():
    linear = torch.tensor([[-1.0, -1.0]], dtype=torch.float64)
    coupling = torch.tensor(
        [[[0.0, 0.123456789123], [0.123456789123, 0.0]]],
        dtype=torch.float64,
    )
    torch.manual_seed(0)
    bits, energy = solver.sa(linear, coupling, R=2, sweeps=2, greedy=True, dev="cpu")

    torch.testing.assert_close(bits, torch.ones_like(linear), rtol=0, atol=0)
    expected = (linear * bits).sum(-1) + torch.einsum("bi,bij,bj->b", bits, coupling, bits)
    torch.testing.assert_close(energy, expected, rtol=0, atol=1e-12)


def test_annealing_rejects_one_sweep():
    with pytest.raises(ValueError, match="at least two sweeps"):
        solver.sa(torch.zeros(1, 2), torch.zeros(1, 2, 2), sweeps=1, dev="cpu")


def test_loop_direction_without_concluding_statistics_when_beta_zero(tmp_path):
    stats_path = tmp_path / "onset.pt"
    torch.save(
        {15: {"resR": torch.tensor([2.0, 0.0]), "resP": torch.tensor([0.0, 0.0]), "nR": 2, "nP": 2}},
        stats_path,
    )

    direction = loop_direction(stats_path, None, [15], beta=0)

    torch.testing.assert_close(direction[15], torch.tensor([1.0, 0.0]))


def test_empty_calibration_fails_with_actionable_message():
    with pytest.raises(ValueError, match="positive token counts for R"):
        require_token_counts({"nR": 0, "nP": 12}, "RP", "Layer 15")


def test_continuous_edit_without_ending_statistics_when_beta_zero(tmp_path, monkeypatch):
    key = "model.language_model.layers.15.mlp.down_proj.weight"
    weights = torch.ones(2, 64, dtype=torch.bfloat16)
    stats_path = tmp_path / "onset.pt"
    reference_path = tmp_path / "reference.json"
    torch.save(
        {
            15: {
                "resR": torch.tensor([2.0, 0.0]),
                "resP": torch.zeros(2),
                "aR": torch.full((64,), 2.0),
                "CP": 2 * torch.eye(64),
                "nR": 2,
                "nP": 2,
            }
        },
        stats_path,
    )
    reference_path.write_text(json.dumps({"layers": {"15": {"ours": {"dose": -1.0}}}}))
    monkeypatch.setattr(solver, "load_shards", lambda path: ([], {}, {key: weights}))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    report = solver.continuous_edit(
        "base-model",
        str(stats_path),
        None,
        str(reference_path),
        "cont_abl",
        str(tmp_path / "edit"),
        layers=[15],
        beta=0,
        save_model=False,
    )

    assert report["layers"]["15"]["dose"] <= -1.0
    assert torch.isfinite(torch.tensor(report["layers"]["15"]["coll"]))


@pytest.mark.parametrize("option", ["solver", "moves", "objective", "u_source", "dose_rule"])
def test_unknown_edit_options_fail_before_loading_a_checkpoint(option):
    with pytest.raises(ValueError, match=option):
        solver.tree_ising_edit("missing-model", "missing-stats", "output", **{option: "typo"})


def test_incomplete_checkpoint_is_not_treated_as_a_finished_edit(tmp_path, monkeypatch):
    out_dir = tmp_path / "partial-model"
    out_dir.mkdir()
    (out_dir / "tree_ising_report.json").write_text(json.dumps({"old": "report"}))
    monkeypatch.setattr(solver, "model_complete", lambda path: False)

    def load_checkpoint(path):
        raise RuntimeError("checkpoint loading attempted")

    monkeypatch.setattr(solver, "load_shards", load_checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint loading attempted"):
        solver.tree_ising_edit(
            "base-model", "onset.pt", str(out_dir), reference_model="agent-model", end_stats_path="end.pt"
        )
