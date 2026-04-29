# Assignment 3 — Pre-training, Fine-tuning, and Prompting Strategies for Bug Fixing

**Course:** CSCI 455 - Generative AI for Software Development  
**Author:** James He

## Overview

This project trains a SentencePiece tokenizer on Java code and compares three bug-fixing paradigms:

| Pipeline | Description |
|----------|-------------|
| A | Pre-train T5-small (span corruption) → fine-tune on bug fixing |
| B | Fine-tune T5-small from scratch (no pre-training) |
| C | RAG 3-shot with CodeBERT retriever + Qwen2.5-Coder-1.5B |
| D | Zero-shot prompting with Qwen2.5-Coder-1.5B (baseline) |

## Repository Structure

```
Assignment_3/
├── assignment3.ipynb          # Main notebook: tokenizer, pre-training, fine-tuning, evaluation
├── rag_code_generation.ipynb  # RAG pipeline: CodeBERT retriever + Qwen inference
├── Pretraining_CodeT5.ipynb   # Reference/exploratory pre-training notebook
├── report.tex                 # LaTeX source for the written report
├── report.pdf                 # Compiled report
├── outputs/
│   ├── tokenizer/
│   │   ├── sp_tokenizer.model # Trained SentencePiece model
│   │   └── sp_tokenizer.vocab # Vocabulary file
│   ├── pretrain_corpus.txt    # Pre-training corpus (~48K Java methods)
│   ├── pretrain_methods.pkl   # Filtered methods used for pre-training
│   ├── pretrained/            # Final pre-trained T5 checkpoint
│   ├── finetuned_a/           # Pipeline A fine-tuned checkpoint + training history
│   ├── finetuned_b/           # Pipeline B fine-tuned checkpoint + training history
│   ├── rag/                   # FAISS index, RAG predictions, zero-shot predictions
│   └── results/
│       ├── final_metrics.json # CodeBLEU and exact match for all 4 pipelines
│       ├── loss_curve.png
│       ├── finetuning_loss_curves.png
│       └── pipeline_comparison.png
```

## Setup

Python 3.13+ required

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install transformers accelerate datasets sentencepiece
pip install faiss-cpu codebleu tqdm numpy
```

## Reproducing Results

Run the notebooks in order. Each notebook saves its outputs to `outputs/` so later notebooks can load them.



## Outputs

| File | Description |
|------|-------------|
| `outputs/tokenizer/sp_tokenizer.model` | SentencePiece tokenizer used by all pipelines |
| `outputs/pretrained/` | Pre-trained T5 checkpoint (Pipeline A source) |
| `outputs/finetuned_a/` | Fine-tuned checkpoint for Pipeline A |
| `outputs/finetuned_b/` | Fine-tuned checkpoint for Pipeline B |
| `outputs/rag/` | FAISS index, per-example predictions for Pipelines C and D |
| `outputs/results/final_metrics.json` | Final CodeBLEU and exact match for all 4 pipelines |
| `outputs/results/*.png` | Loss curves and pipeline comparison bar charts |

## Results Summary

| Pipeline | Exact Match | CodeBLEU |
|----------|-------------|----------|
| A: Pre-train → Fine-tune (T5-small) | 0.0000 | 0.0970 |
| B: Scratch → Fine-tune (T5-small)   | 0.0000 | 0.0447 |
| C: RAG 3-shot (Qwen2.5-Coder 1.5B) | 0.0342 | 0.8606 |
| D: Zero-shot (Qwen2.5-Coder 1.5B)  | 0.0000 | 0.4152 |
