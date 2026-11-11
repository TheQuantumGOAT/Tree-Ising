# Tree-Ising

Code for **Tree-Ising: Repairing Language Models with Bit-Level Ising Edits**, by Hadi Salloum and Ammar Ali.

Tree-Ising repairs reasoning loops by choosing sparse sign flips and power-of-two changes to BF16 model weights. Forward-pass statistics define a QUBO for each output row. The implementation includes simulated annealing (SA), simulated bifurcation (SB), continuous baselines, evaluation, and the experimental repair loop.

## Start here

Clone the repository and create a Python 3.11 environment. Run the commands from the repository root.

```bash
git clone https://github.com/TheQuantumGOAT/Tree-Ising.git
cd Tree-Ising
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
run_treeising --config configs/qwen35_0.8b.yaml --list
```

The base installation needs only PyYAML. Listing experiments validates their names, inheritance, and function arguments and works on a laptop. The same CLI is available as `python -m TreeIsing`.

For model repair and generation, use **Linux with an NVIDIA CUDA GPU**. The supplied experiments report an A100 with 40 GB of GPU memory. Install the reproduction dependencies in the environment above.

```bash
python -m pip install -e ".[reproduce]" -c requirements-reproduce.txt
python -c 'import torch, vllm; print("PyTorch", torch.__version__, "vLLM", vllm.__version__); assert torch.cuda.is_available(), "A CUDA GPU is required"'
```

`requirements-reproduce.txt` fixes the resolved supporting dependencies for Linux x86_64 and Python 3.11. It retains the supplied Torch 2.13.0, vLLM 0.32.0, Transformers 5.19.0, Safetensors 0.8.0, Math-Verify 0.9.0, NumPy 2.3.5, and PyYAML 6.0.3 versions. This dependency set resolves CUDA 13 packages. The source archive reports CUDA 12.9 for its original experiments, so select an NVIDIA driver compatible with the installed runtime when reproducing them.

Allow disk space for downloaded checkpoints, edited checkpoints, and generated answers. Each edited model is saved as a complete Hugging Face checkpoint.

## Run the main repair

First download the two input checkpoints and prepare the benchmark and practice files. The data step fetches the upstream practice problems through Git. The checksum step checks every generated input against the supplied manifest.

```bash
run_treeising --config configs/qwen35_0.8b.yaml --only models,data,splits,check_data
```

Then collect base-model answers and statistics and create the SA repair.

```bash
run_treeising --config configs/qwen35_0.8b.yaml --only practice_answers,statistics,layer_scan,treeising_sa
```

The repaired checkpoint is written to `models/treeising_sa`. The edit report is `results/qwen35_0.8b/edits/treeising_sa.json`, and calibration statistics are saved in `results/qwen35_0.8b/`.

To create the SB repair from the same statistics, run the next command.

```bash
run_treeising --config configs/qwen35_0.8b.yaml --only treeising_sb
```

The SB checkpoint is written to `models/treeising_sb`. Both output directories work as ordinary model paths for Hugging Face Transformers or vLLM.

To compare the agent checkpoint and both repairs on the configured benchmarks, run the following command after both edits finish.

```bash
run_treeising --config configs/qwen35_0.8b.yaml --only benchmarks,heldout400
```

Evaluation writes answer JSONL files and summaries under `results/qwen35_0.8b/gen/`, plus paired comparisons in `results/qwen35_0.8b/benchmarks.json` and `heldout400.json`. These workloads use long generation limits and multiple completions per problem.

## Reproduce the experiment suites

| Configuration | Contents | Prerequisites |
| --- | --- | --- |
| `configs/qwen35_0.8b.yaml` | Input preparation, statistics, layer scan, SA/SB, ablations, continuous edits, benchmark comparisons, footprint | Reproduction environment |
| `configs/qwen35_2b.yaml` | Qwen3.5-2B transfer, dose sweep, three-layer variants, greedy GSM8K comparisons | Data prepared by the 0.8B configuration |
| `configs/repair_loop.yaml` | Experimental iterative repair, model-slot comparisons, held-out reports | 0.8B checkpoints, statistics, and benchmark outputs |

Each complete configuration runs in file order. Running the first configuration without `--only` includes every ablation and evaluation.

```bash
run_treeising --config configs/qwen35_0.8b.yaml
run_treeising --config configs/qwen35_2b.yaml
run_treeising --config configs/repair_loop.yaml
```

`--only` selects experiment names or comma-separated wildcard patterns. Selected experiments keep their order in the YAML file. Run their input and calibration steps before selecting a later experiment.

```bash
run_treeising --config configs/qwen35_0.8b.yaml --only 'treeising_*' --list
run_treeising --config configs/qwen35_2b.yaml --list
```

The repair loop also writes `final.pt`, which lets the build step use the final accepted weights after an early stop. Completed outputs are reused on reruns. Statistics and reports use atomic writes, generated answers are published with their summaries, and edited checkpoints are staged before publication. An incomplete checkpoint from an older run produces an error with recovery instructions. Move that incomplete output aside and rerun the affected step. To change experiment settings, choose new output paths so the run produces separate artifacts.

## Inputs and reproducibility

The configurations pin model revisions, benchmark dataset revisions, and the practice-source Git commit. `data/MD5SUMS` records the supplied benchmark and practice-file checksums. The data preparation instructions and file counts are in [data/README.md](data/README.md).

| Model | Revision |
| --- | --- |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` |
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| `RikiyaT/Qwen3.5-0.8B-weight-space-programming` | `d1671a178705bcab7644806d23c535517f35723b` |

Use the same configuration, input hashes, seed, layer order, hardware, and runtime when comparing runs. Generation can vary with GPU kernels and runtime versions. Exact benchmark reproduction requires executing the complete CUDA workloads and checking their generated reports. The CPU regression tests cover configuration validation, QUBO helpers, data preparation, and interruption recovery. Release validation regenerated all 17 input files and matched every supplied checksum.

The code release includes the classical SA/SB repair pipelines. The quantum proof-of-concept experiment is described in the paper and has a separate hardware workflow.

## Reference results supplied with the code

These are the reference values included in the source archive. The corresponding commands above generate new measurements in your environment.

| Qwen3.5-0.8B | MATH-500@4 | GSM8K@1 | AMC23@16 | Minerva@4 | OlympiadBench@1 | Changed weights | State-tree bit flips |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Agent edit | 51.7 | 65.6 | 22.0 | 24.2 | 21.4 | 130,188 | 720,586 |
| Tree-Ising SA | 54.6 | 70.5 | 22.7 | 26.1 | 24.3 | 112,630 | 181,062 |
| Tree-Ising SB | 55.1 | 70.8 | 25.0 | 25.6 | 24.0 | 108,327 | 174,245 |

`@k` means `k` completions per problem. Accuracy averages correctness across those completions.

## Development checks

Install the development tools in the reproduction environment. The tests run on CPU.

```bash
python -m pip install -e ".[dev]"
python -m pytest -p no:capture -q
ruff check --select E9,F,I .
ruff format --check .
```

The `-p no:capture` option also avoids a native readline issue observed in the local Anaconda test environment. Configuration listing can be checked separately with the base installation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `TreeIsing/pipeline.py` | YAML runner and iterative repair loop |
| `TreeIsing/calibration.py` | Forward statistics, loop directions, and layer scan |
| `TreeIsing/solver.py` | QUBO construction, SA/SB solvers, calibration, continuous baselines |
| `TreeIsing/evaluation.py` | Generation, grading, paired comparisons, footprints, model-slot engine |
| `TreeIsing/utils.py` | Data preparation, pinned checkpoint downloads, state-tree fields, atomic I/O |
| `configs/` | Experiment definitions and supplied reference values |
| `data/MD5SUMS` | Input checksum manifest |
| `requirements-reproduce.txt` | Resolved Linux/Python 3.11 dependency constraints |
| `tests/` | Focused CPU regression tests |

## Citation and upstream work

```bibtex
@software{salloum2026treeising,
  title = {Tree-Ising: Repairing Language Models with Bit-Level Ising Edits},
  author = {Salloum, Hadi and Ali, Ammar},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/TheQuantumGOAT/Tree-Ising}
}
```

The agent checkpoint, practice problems, and evaluation protocol come from Takehi et al., *Sparks of Weight-Space Programming*. [Project](https://rikiyat.github.io/blog/weight-space-programming/) · [Released logs](https://github.com/RikiyaT/lm_understand-logs).
