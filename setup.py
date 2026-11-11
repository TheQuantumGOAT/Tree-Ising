from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent

# Runtime versions supplied with the experiments (A100-40GB, CUDA 12.9, Python 3.11).
reproduction_dependencies = [
    "torch==2.13.0",
    "vllm==0.32.0",
    "transformers==5.19.0",
    "safetensors==0.8.0",
    "math-verify==0.9.0",
    "numpy==2.3.5",
    "huggingface_hub",
    "colorama",
    "datasets",
]

setup(
    name="TreeIsing",
    version="0.2.0",
    packages=["TreeIsing"],
    install_requires=["PyYAML==6.0.3"],
    extras_require={
        "reproduce": reproduction_dependencies,
        "data": ["datasets"],
        "dev": ["ruff>=0.14,<1", "pytest>=8,<10"],
    },
    entry_points={
        "console_scripts": [
            "run_treeising=TreeIsing.pipeline:run_from_config",
        ],
    },
    author="Hadi Salloum and Ammar Ali (QDeep; LEAD, ITMO University)",
    description="Tree-Ising: repairing language models with bit-level Ising edits on the BF16 state tree.",
    url="https://github.com/TheQuantumGOAT/Tree-Ising",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.11",
)
