"""CPU regressions for interrupted outputs and invalid evaluation inputs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from TreeIsing import evaluation, utils


def write_checkpoint(directory, value=1):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    save_file({"weight": torch.tensor([value], dtype=torch.bfloat16)}, directory / "model.safetensors")


def test_atomic_output_preserves_previous_file_on_failure(tmp_path):
    path = tmp_path / "result.json"
    path.write_text("previous", encoding="utf-8")
    with pytest.raises(RuntimeError):
        with utils.atomic_output(path) as handle:
            handle.write("partial")
            raise RuntimeError("interrupted")
    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.iterdir()) == [path]


def test_model_complete_rejects_missing_or_invalid_shards(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert not utils.model_complete(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"partial")
    assert not utils.model_complete(tmp_path)
    write_checkpoint(tmp_path)
    assert utils.model_complete(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}}), encoding="utf-8"
    )
    assert not utils.model_complete(tmp_path)


def test_checkpoint_rebuild_replaces_complete_output(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "edited"
    write_checkpoint(source, value=1)
    write_checkpoint(destination, value=2)
    names, shards, weights = utils.load_shards(source)
    weights["weight"] = torch.tensor([3], dtype=torch.bfloat16)
    utils.save_like(source, destination, names, shards, weights)
    assert utils.model_complete(destination)
    assert load_file(destination / "model.safetensors")["weight"].item() == 3
    assert not list(tmp_path.glob(".edited-*"))


def test_failed_checkpoint_write_preserves_complete_output(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "edited"
    write_checkpoint(source, value=1)
    write_checkpoint(destination, value=2)
    names, shards, weights = utils.load_shards(source)
    with patch("safetensors.torch.save_file", side_effect=RuntimeError("interrupted")):
        with pytest.raises(RuntimeError, match="interrupted"):
            utils.save_like(source, destination, names, shards, weights)
    assert utils.model_complete(destination)
    assert load_file(destination / "model.safetensors")["weight"].item() == 2
    assert not list(tmp_path.glob(".edited-*"))


def test_failed_checkpoint_publish_restores_previous_output(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "edited"
    write_checkpoint(source, value=1)
    write_checkpoint(destination, value=2)
    names, shards, weights = utils.load_shards(source)
    replace = utils.os.replace

    def fail_publication(src, dst):
        if (
            Path(dst) == destination
            and Path(src).name.startswith(".edited-")
            and "-backup-" not in Path(src).name
        ):
            raise OSError("publication failed")
        return replace(src, dst)

    with patch.object(utils.os, "replace", side_effect=fail_publication):
        with pytest.raises(OSError, match="publication failed"):
            utils.save_like(source, destination, names, shards, weights)
    assert utils.model_complete(destination)
    assert load_file(destination / "model.safetensors")["weight"].item() == 2
    assert not list(tmp_path.glob(".edited-*"))


def test_download_resumes_config_only_checkpoint(tmp_path):
    destination = tmp_path / "example"
    destination.mkdir()
    (destination / "config.json").write_text("{}", encoding="utf-8")
    download = Mock(side_effect=lambda **kwargs: write_checkpoint(Path(kwargs["local_dir"])))
    with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
        utils.download_models(["owner/example"], str(tmp_path))
    download.assert_called_once()
    assert utils.model_complete(destination)


def test_pinned_download_verifies_revision_even_with_complete_local_model(tmp_path):
    destination = tmp_path / "example"
    write_checkpoint(destination)
    download = Mock()
    with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
        utils.download_models(
            ["owner/example"], str(tmp_path), revisions={"owner/example": "verified-commit"}
        )
    download.assert_called_once_with(
        repo_id="owner/example", local_dir=str(destination), revision="verified-commit"
    )


@pytest.mark.parametrize("indexed", [False, True])
def test_checkpoint_shards_exclude_files_from_previous_revisions(tmp_path, indexed):
    write_checkpoint(tmp_path)
    save_file({"stale": torch.tensor([9])}, tmp_path / "previous.safetensors")
    expected = "model.safetensors"
    if indexed:
        expected = "model-00001-of-00001.safetensors"
        save_file({"weight": torch.tensor([2])}, tmp_path / expected)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": expected}}), encoding="utf-8"
        )
    names, _, weights = utils.load_shards(tmp_path)
    assert names == [expected]
    assert list(weights) == ["weight"]
    assert evaluation._shards(tmp_path) == [str(tmp_path / expected)]


def test_data_rebuilds_missing_math_subset(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [{"unique_id": str(i), "problem": str(i), "answer": str(i)} for i in range(100)]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "math500.jsonl").write_text("{}\n", encoding="utf-8")
    download = Mock(return_value=str(raw))
    with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(hf_hub_download=download)}):
        utils.prepare_data(str(tmp_path), benchmarks=["math500"])
    assert len((tmp_path / "math500_sub100.jsonl").read_text(encoding="utf-8").splitlines()) == 100
    download.assert_called_once()


def test_data_revisions_are_forwarded_to_every_upstream_download(tmp_path):
    raw = tmp_path / "raw.jsonl"
    rows = [{"unique_id": str(i), "problem": str(i), "answer": str(i)} for i in range(100)]
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    download = Mock(return_value=str(raw))
    load = Mock(return_value=[])
    revisions = {name: name + "-commit" for name in ("math500", "gsm8k", "amc23", "minerva", "olympiad")}
    with patch.dict(
        sys.modules,
        {
            "huggingface_hub": SimpleNamespace(hf_hub_download=download),
            "datasets": SimpleNamespace(load_dataset=load),
        },
    ):
        utils.prepare_data(str(tmp_path / "data"), dataset_revisions=revisions)
    assert download.call_args.kwargs["revision"] == "math500-commit"
    for call in load.call_args_list:
        source = call.args[0]
        name = {
            "openai/gsm8k": "gsm8k",
            "math-ai/amc23": "amc23",
            "math-ai/minervamath": "minerva",
            "math-ai/olympiadbench": "olympiad",
        }[source]
        assert call.kwargs["revision"] == revisions[name]
    assert load.call_count == 4


def test_data_rebuilds_incomplete_practice_group(tmp_path):
    practice = tmp_path / "practice"
    practice.mkdir()
    (tmp_path / "practice_collect.jsonl").write_text("{}\n", encoding="utf-8")
    with patch.object(utils, "_practice") as rebuild:
        utils.prepare_data(str(tmp_path), benchmarks=[], practice_dir=str(practice))
    rebuild.assert_called_once_with(str(tmp_path), str(practice))


def test_missing_practice_folder_fetches_pinned_revision_without_resetting_existing_repo(tmp_path):
    repo = tmp_path / "upstream"
    (repo / ".git").mkdir(parents=True)
    sentinel = repo / "keep.txt"
    sentinel.write_text("user checkout", encoding="utf-8")
    practice = repo / "exp8_math" / "practice"

    def fake_git(command, **kwargs):
        if "checkout" in command:
            downloaded = Path(command[2]) / "exp8_math" / "practice"
            downloaded.mkdir(parents=True)
            for name in ("math_train.jsonl", "gsm8k_train.jsonl"):
                (downloaded / name).write_text("{}\n", encoding="utf-8")

    with patch.object(utils, "_practice"), patch.object(utils.subprocess, "run", side_effect=fake_git) as run:
        utils.prepare_data(
            str(tmp_path / "data"),
            benchmarks=[],
            practice_dir=str(practice),
            practice_revision="verified-commit",
        )
    fetch = [call.args[0] for call in run.call_args_list if "fetch" in call.args[0]]
    assert len(fetch) == 1
    assert fetch[0][-1] == "verified-commit"
    assert fetch[0][2] != str(repo)
    assert (practice / "math_train.jsonl").is_file()
    assert sentinel.read_text(encoding="utf-8") == "user checkout"
    assert not list(practice.parent.glob(".treeising-practice-*"))


def test_existing_practice_folder_is_read_without_git_changes(tmp_path):
    practice = tmp_path / "practice"
    practice.mkdir()
    sentinel = practice / "math_train.jsonl"
    sentinel.write_text("user data", encoding="utf-8")
    with patch.object(utils, "_practice"), patch.object(utils.subprocess, "run") as run:
        utils.prepare_data(
            str(tmp_path / "data"), benchmarks=[], practice_dir=str(practice), practice_revision="pinned"
        )
    run.assert_not_called()
    assert sentinel.read_text(encoding="utf-8") == "user data"


def test_generate_rejects_empty_selection_before_loading_vllm(tmp_path):
    data = tmp_path / "empty.jsonl"
    data.write_text("", encoding="utf-8")
    output = tmp_path / "output.jsonl"
    Path(str(output) + ".summary.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="No problems selected"):
        evaluation.generate("unused-model", str(data), str(output))


def test_paired_comparison_rejects_disjoint_ids(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"id": "a", "correct": True}) + "\n", encoding="utf-8")
    b.write_text(json.dumps({"id": "b", "correct": True}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No shared problem IDs"):
        evaluation.paired_test(str(a), str(b))
    with pytest.raises(ValueError, match="No shared problem IDs"):
        evaluation._paired({"a": 1}, {"b": 1})
