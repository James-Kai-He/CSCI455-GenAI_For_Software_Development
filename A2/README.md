## Data Source & Pre-processing

### Source

The corpus was constructed by mining **public GitHub repositories**, following the methodology of the CodeXGLUE Java code-to-text dataset (Lu et al., 2021).

**Repository selection criteria:**
- Language: Java
- Stars: > 1,000
- Forks excluded
- 1,650 repositories fetched across three star-range bands (`>10,000`, `3,000–10,000`, `1,000–3,000`) to work around the GitHub Search API's hard cap of 1,000 results per query. This was the minimum required to hit the required amount of data points

Each repository was shallow-cloned (`git clone --depth 1`) to retrieve only the latest snapshot

### Pair Extraction

Each Java file was parsed with `javalang` to locate `MethodDeclaration` nodes. A method was included as a training sample only if it had a **Javadoc comment** ending within three lines above the method signature. The summary was extracted from the Javadoc block by:

1. Stripping the `/**` / `*/` delimiters and per-line `*` prefixes
2. Truncating at the first `@tag` line (e.g. `@param`, `@return`)
3. Removing HTML markup
4. Collapsing whitespace and lowercasing


Up to 40 Java files were randomly sampled per repository, excluding directories named `test`, `tests`, `example`, `examples`, `sample`, `samples`, `demo`, `generated`, or `gen`.

### Filtering

The following filters were applied to each extracted (code, summary) pair:

| Filter | Threshold | Rationale |
|---|---|---|
| Non-ASCII characters in code | Remove | Embedding model expects ASCII-safe input |
| Java token count | ≥ 5 tokens | Removes trivially short methods |
| Summary word count | 3 – 60 words | Removes uninformative labels and paragraph-length comments |
| TODO / FIXME / HACK / XXX in code | Remove | Incomplete implementations |
| Empty method body | Remove | No logic to summarise |
| Getter/setter heuristic | Remove | Single-statement accessors produce low-diversity summaries |

### Normalisation

- **Code**: each Java method was flattened to a **single whitespace-normalised line** (all newlines and tabs replaced by a single space)
- **Summary**: lowercased; stray whitespace collapsed

### Deduplication

Exact duplicates were removed by hashing the normalised code string.

### Dataset Splits

| Split | Size | Usage |
|---|---|---|
| Training | 50,000 pairs | Model training |
| Validation | 1,000 pairs | Early stopping (BLEU-1) |
| Test | 1,000 pairs | Final evaluation only — provided|

### Output Files

All files are written to `./datasets/lstm_dataset/`:

```
train_code.txt       — 50,000 normalised Java methods (one per line)
train_summary.txt    — 50,000 corresponding lowercase summaries
val_code.txt         — 1,000 validation code samples
val_summary.txt      — 1,000 validation summaries
metadata.json        — full filter statistics and per-repo mining log
```

## Dependencies and Installation
Download the google drive folder here:
https://drive.google.com/drive/u/2/folders/1TXo24H-gxxF9ZPnXIog5yYQHh1Hlv1YI

Download assignment-2-LSTM.ipynb and place it in the folder

Download the models folder from here and place it in the folder: https://drive.google.com/drive/folders/150xbvYtyuUNsd8hefjiZXqa_eFY8f67K

The folder structure should look like this:

Project/ \
├── assignment-2-LSTM.ipynd \
├── dataset \
│   └── test_dataset_tokenized.csv \
└── Models Folder

Create a venv with python 3.10 and then activate it.

**IMPORTANT**
Put your github access token in

```
GITHUB_TOKEN   = ""
```

Then you should be able to run the entire notebook.

## Outputs

### Dataset (`./datasets/lstm_dataset/`)

| File|
|---|
| `train_code.txt`|
| `train_summary.txt`|
| `val_code.txt`|
| `val_summary.txt`|
| `train_code.pt` |
| `train_summary.pt`|
| `val_code.pt`|
| `val_summary.pt` |
| `metadata.json`|

### Model Checkpoints (`./checkpoints/`)

| File|
|---|
| `lstm_codet5p_summarization.pt`|

### Predictions & Evaluation (`./predictions/`)

| File|
|---|
| `lstm_codet5p_summarization_test_results.json`|
| `training_curves.png`|
