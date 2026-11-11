"""Tree-Ising: repairing language models with bit-level Ising edits on the BF16 state tree.

utils        general helpers (configs, prompts and grading, checkpoints, state-tree fields, problem files)
calibration  activation statistics of the base model on its own practice answers, layer scan, loop direction
solver       the Tree-Ising QUBO edit (annealing, bifurcation, dose calibration) and its continuous baselines
evaluation   generation and grading, benchmark suites, paired tests, footprints, multi-slot engine
pipeline     run_treeising (joint YAML runner) and the evolving repair loop"""

__version__ = "0.2.0"
