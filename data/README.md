# Problem files

The repository includes preparation code and the authors' checksum manifest. Generate the JSONL problem files before running
calibration or evaluation. Run the commands below from the repository root.

Each line is one JSON object with `id`, `problem`, and `answer` fields. IDs identify problems across generation files and paired
comparisons. Answers are graded with Math-Verify against the last `\boxed{}` after `</think>` in thinking mode, or against the
full completion when `nothink` is enabled (`TreeIsing/utils.py`, `grade`).

`MD5SUMS` contains the supplied checksums of the paper's problem files. The supplied configuration pins all five benchmark
snapshots through `dataset_revisions`, as well as the practice source commit. The `check_data` step compares regenerated files
with the manifest and reports missing files or mismatches. Investigate a checksum failure before treating a rebuilt dataset
as the paper's input.

| File | Problems | Source | Used for |
|---|---|---|---|
| `math500.jsonl` | 500 | HuggingFaceH4/MATH-500 (`unique_id` as id) | MATH-500@1, MATH-500@4 |
| `math500_sub100.jsonl` | 100 | random subset of MATH-500, seed 0 | development problems; excluded in the held-out-400 check |
| `gsm8k.jsonl` | 1,319 | openai/gsm8k, main, test | GSM8K@1; Qwen3.5-2B transfer |
| `amc23.jsonl` | 40 | math-ai/amc23 | AMC23@16 |
| `minerva.jsonl` | 272 | math-ai/minervamath | Minerva@4 |
| `olympiad.jsonl` | 674 | math-ai/olympiadbench (English, text-only, open-ended) | OlympiadBench@1 |
| `practice_collect.jsonl` | 384 | 192 MATH-train + 192 GSM8K-train, seed 1 | the statistics of every edit |
| `practice_screen.jsonl` | 128 | 96 + 32, disjoint from practice_collect, seed 7 | practice screen (Table 5, Fig. 4) |
| `practice_rs.jsonl` | 2,000 | 1,000 + 1,000, disjoint from both, seed 11 | held-out practice comparison of the repair loop |
| `practice_screen512.jsonl` | 512 | next 384 + 128 of the same shuffle | part of the rl1 loop pool |
| `heldout/*_repair.jsonl`, `heldout/*_held.jsonl` | 246 / 254, 627 / 692, 344 / 330 | MATH-500, GSM8K, OlympiadBench split by the parity of md5(id) | rl2 repair and held-out halves |
| `heldout/repair_pool_rl2.jsonl` | 746 | at most 250 repair problems per benchmark | rl2 loop pool |

The benchmark mapping follows the agent's harness (MIT exp8, `code/evaluate.py prepare`). The practice problems are the MATH and
GSM8K training problems released with the agent's logs, [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)
(`exp8_math/practice`); none of them is in any evaluation set.

Prepare the data with the reproduction dependencies installed and Git available on `PATH`.

```bash
python -m pip install -e ".[reproduce]"
run_treeising --config configs/qwen35_0.8b.yaml --only data,splits,check_data
```

The configured practice folder is `external/lm_understand-logs/exp8_math/practice`. It must contain `math_train.jsonl` and
`gsm8k_train.jsonl`. The supplied configuration pins `practice_revision` to
`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. Rebuilding all four practice files from this commit reproduces their supplied
checksums. When the practice folder is missing, the data step fetches this commit in a temporary sparse checkout and publishes
the practice folder after the checkout succeeds. To use a particular existing checkout, set `practice_dir` to its
`exp8_math/practice` folder. Existing practice folders are read directly and checked through the generated-file manifest.

Complete benchmark files and complete practice-file groups are skipped on reruns. If a MATH-500 subset or one of the four
practice files is missing, that group is rebuilt. Writes are staged in temporary files, so interrupted writes can be retried.
The split step accepts an existing file only when its contents match the newly generated split.

You can also check the manifest directly from the repository root.

```bash
md5sum -c data/MD5SUMS
```

On macOS, use the portable Python `check_data` step above if `md5sum` is unavailable.
