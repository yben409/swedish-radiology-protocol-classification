# Swedish Radiology Protocol Classification with Fine-tuned LLMs

Automatic assignment of CT examination protocols from Swedish free-text radiology referrals, comparing four fine-tuned open-weight LLMs (7–8B parameters).

## Problem
Each radiology referral must be matched to an examination protocol (e.g. `A1`, `B3`, `N14`) before the scan is performed, based on the clinical history and question. [insert: who does this today and why automating it matters for the client]. This project evaluates whether a fine-tuned LLM can predict the protocol directly from the Swedish referral text, and which model family and fine-tuning approach works best.

## Data
- **Source:** historical referrals provided by a Swedish healthcare client under contract. The data is the client's property and is **not included** in this repository; see [`data/README.md`](data/README.md).
- **Content:** 1,353 free-text referrals in Swedish (clinical history *Anamnes* + clinical question *Frågeställning*), each labelled with the protocol that was actually used.
- **Labels:** 33 distinct protocols in the raw data. Only protocols with ≥ 50 examples were kept → **14 classes, 1,127–1,135 referrals** depending on the dataset export.
- **Class imbalance:** the most frequent protocol (`B3`) accounts for 279 referrals; the smallest kept classes have 54–56.

## Method
Four models fine-tuned end-to-end (full fine-tuning, bf16) on a single NVIDIA A100 80 GB, using two approaches:

**1. Classification head** — `AutoModelForSequenceClassification`, one logit per protocol, cross-entropy loss.

| Model | Input | Max length | Epochs | LR | Effective batch | Split |
|---|---|---|---|---|---|---|
| Meta-Llama-3-8B-Instruct | Anamnes + Frågeställning | 1024 | 3 | 2e-5 | 32 | 80 / 10 / 10 |
| DeepSeek-R1-Distill-Llama-8B | Anamnes + Frågeställning | 1024 | 3 | 2e-5 | 32 | 80 / 10 / 10 |
| Salamandra-7B-Instruct | Summary wrapped in a Swedish instruction prompt | 768 | 3 | 1e-5 | 16 | 70 / 15 / 15 |

**2. Generative fine-tuning** — GPT-SW3-6.7B-v2-Instruct (Swedish-native model) trained as a causal LM to answer the protocol name after a Swedish instruction prompt listing the allowed labels. Loss computed on answer tokens only (prompt tokens masked with `-100`). At inference: greedy decoding (8 new tokens), then mapping of the generated text to the closest valid label. Duplicate referrals removed before splitting.

**Common setup**
- Stratified train / validation / test splits, fixed seed (42).
- Memory: gradient checkpointing, bf16, TF32, 8-bit AdamW (bitsandbytes) where available.
- Automatic out-of-memory fallback: training retries with shorter max length and smaller batch / larger gradient accumulation.
- Evaluation: top-1 accuracy, macro and weighted F1, top-3 accuracy, per-class report, confusion matrix.

## Results
Test-set results, top-1 unless stated:

| Model | Approach | Test n | Accuracy | F1 macro | F1 weighted | Top-3 accuracy |
|---|---|---|---|---|---|---|
| **Llama-3-8B-Instruct** | Classification head | 113 | **0.920** | **0.923** | **0.919** | — |
| DeepSeek-R1-Distill-Llama-8B | Classification head | 113 | 0.876 | 0.870 | 0.881 | 0.974 |
| Salamandra-7B-Instruct | Classification head | 171 | 0.866 | 0.850 | 0.864 | 0.965 |
| GPT-SW3-6.7B-v2-Instruct | Generative | 171 | 0.842 | 0.826 | 0.843 | — |

- Llama-3-8B with a classification head gave the best test results in these runs.
- On the 171-example split, the classification-head model (Salamandra) scored above the generative model (GPT-SW3).
- Most frequent confusions in the Llama-3 and DeepSeek confusion matrices: `A3` ↔ `A4`.
- "—": not reported (not computed for Llama-3; excluded for GPT-SW3 because its beam-search top-3 evaluation was unreliable).

## Limitations
- **Not a strictly controlled comparison:** Llama-3 and DeepSeek were evaluated on a 113-example test set, Salamandra and GPT-SW3 on a different 171-example split built from a different export of the dataset (different text field). Differences of a few points should not be over-interpreted.
- **Small test sets:** 5–13 examples per class for most protocols; per-class scores are noisy.
- **Rare protocols excluded:** the 19 protocols with < 50 examples were dropped, so the models do not cover the full protocol catalogue.
- **Single run per model:** one seed, no variance estimate, no hyperparameter search.
- **Not reproducible without the data:** the dataset is proprietary and cannot be shared.

## Tech stack
- Python, PyTorch
- Hugging Face Transformers, Datasets, Accelerate, Hub
- bitsandbytes (8-bit AdamW)
- scikit-learn (splits, metrics), pandas, openpyxl, matplotlib
- Hardware: NVIDIA A100 80 GB (Google Colab)

## Repository structure
```
swedish-radiology-protocol-classification/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   └── README.md                              # expected data format (data not included)
├── radiologue_llama_git_lfs-2.ipynb           # Llama-3-8B-Instruct, classification head
├── radiologue_deepseek_r1_git_lfs-2.ipynb     # DeepSeek-R1-Distill-Llama-8B, classification head
├── salamandra_radiologue-3.ipynb              # Salamandra-7B-Instruct, classification head
└── radiologue_gpt_sw3_(1)_2.ipynb             # GPT-SW3-6.7B-v2-Instruct, generative
```

## How to run
Requires a GPU with ~80 GB VRAM for full fine-tuning of the 7–8B models (the OOM fallback lowers requirements, at the cost of shorter inputs).

```bash
pip install -r requirements.txt
cp .env.example .env        # then set HF_TOKEN=<your Hugging Face token>
```

1. Request access to the gated [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model on Hugging Face (needed for the Llama-3 notebook).
2. Place an Excel file in `data/` following the format described in `data/README.md`.
3. Run any notebook top to bottom. Checkpoints and evaluation outputs (confusion matrix, predictions) are written to the output directory set at the top of each notebook.

## Git LFS
Fine-tuned checkpoints are **not** stored in this repository: an 8B model in bf16 is ~16 GB of `.safetensors` shards, far beyond GitHub's 100 MB file limit. The Llama-3 and DeepSeek notebooks check that Git LFS is installed, which is required to push or pull checkpoints to/from a Hugging Face Hub model repository.

Install and enable Git LFS:
```bash
# macOS
brew install git-lfs
# Ubuntu / Debian
sudo apt-get install git-lfs
# Windows (Git for Windows ships with Git LFS)
winget install Git.Git

git lfs install
```

Version a checkpoint in a Hugging Face model repository:
```bash
git clone https://huggingface.co/<user>/<model-repo>
cd <model-repo>
git lfs track "*.safetensors" "*.bin" "tokenizer.json"
cp -r <output_dir>/* .
git add .gitattributes .
git commit -m "Add fine-tuned checkpoint"
git push
```
