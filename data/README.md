# Data

The dataset is the property of a Swedish healthcare client and is **not included** in this repository. This folder documents the expected format so the notebooks can be run on equivalent data.

## Expected input
A single Excel file (`.xlsx`), one row per radiology referral, free text in Swedish.

Two exports of the same 1,353 referrals were used, with different column names:

| Notebooks | Text column(s) | Label column |
|---|---|---|
| Llama-3, DeepSeek-R1-Distill | `Anamnes` (clinical history), optionally `Frågeställning` (clinical question), concatenated | `Protocol` |
| Salamandra, GPT-SW3 | `Summa` (referral summary) | `End Result` |

- **Label:** the CT protocol code actually used for the examination (e.g. `A1`, `B3`, `N14`, `A4 med sen fas`).
- **Filtering (done in the notebooks):** empty rows dropped; only protocols with ≥ 50 examples kept (14 of 33).

## File location
Place the file in this folder and point the path variable at the top of each notebook to it:

| Notebook | Variable |
|---|---|
| Llama-3, DeepSeek-R1-Distill | `EXCEL_PATHS` |
| GPT-SW3 | `CFG["excel_path"]` |
| Salamandra | `DATA_XLSX` |

Everything in `data/` except this README is ignored by git.
