# Hierarchical Semantic Retrieval for EEG-to-Text

This repository contains the clean code path for an EEG-to-text generation pipeline:

```text
EEG
  -> frozen EEGPT backbone + trainable adapter
  -> low / mid / high semantic heads
  -> structured semantic retrieval
  -> evidence-based semantic decision
  -> LLM prompt composition
  -> one-sentence caption
```

The current main method is retrieval-based. It does not use the older MoE soft-prompt route.

## What Is Included

```text
eeg_text_codex/
  config.py      # paths, model names, dimensions
  data.py        # EEG dataset wrappers and collators
  modules.py     # EEGPT wrapper, semantic heads, optional reranker modules
  rerank.py      # candidate feature construction for reranker experiments
  utils.py       # checkpoint, model loading, and utility functions

scripts/
  stage1_train_eegpt_adapter_classifier.py   # EEGPT adapter classifier stage
  generate_qwen_structured_captions.py       # image -> structured caption by Qwen-VL
  normalize_qwen_structured_captions.py      # clean and normalize generated captions
  build_structured_semantic_database.py      # build low/mid/high semantic target DB
  stage2_train_structured_semantics.py       # train multi-head semantic alignment
  stage4_retrieval_infer.py                  # retrieve semantic anchors and generate captions
  evaluate_caption_metrics.py                # CLIP text sim, BLEU, ROUGE, METEOR-like, CIDEr-like
  stage5_train_candidate_reranker.py         # optional candidate reranker ablation

docs/
  METHOD_OVERVIEW.md
  RUN_COMMANDS.md

examples/metrics_summary/
  Small JSON summaries from existing runs.
```

Large EEG data, images, LLM weights, EEGPT checkpoints, and generated CSV files are intentionally excluded.

## Main Stages

### Stage 1: EEGPT Adapter Classification

Load a pretrained EEGPT backbone, freeze the backbone, and train only the adapter/classifier side for EEG object-label prediction. This gives the later retrieval stage a coarse object prior.

### Stage 2: Structured Semantic Alignment

For every image, a structured caption is converted into three text-semantic targets:

- low: visual attributes such as color, texture, shape, and size
- mid: object, layout, and action/state
- high: scene, context, and global meaning

The EEG model predicts three embeddings and aligns them with these targets using symmetric contrastive loss. A small classification loss is also kept as an auxiliary object constraint.

### Stage 4: Retrieval, Evidence, and LLM Generation

The trained EEG semantic heads retrieve top-k anchors from a semantic database. The evidence module combines low/mid/high retrieval signals with classifier candidates, then builds a structured prompt for the LLM. The LLM is asked to produce one conservative caption.

## Example Results

On the current full test run with structured prompt:

```text
Token F1:              0.4669
CLIP text similarity:  0.6608
BLEU-4 corpus:         0.2348
ROUGE-L:               0.4485
METEOR-like:           0.4440
CIDEr-like:            0.8766
```

The exact summary is stored in:

```text
examples/metrics_summary/stage4_structured_retrieval_full_evidence_structured_prompt_llm_metrics_summary.json
```

## Run

See [docs/RUN_COMMANDS.md](docs/RUN_COMMANDS.md).

Before running, update local paths in `eeg_text_codex/config.py`, especially EEG data, image data, CLIP, Qwen-VL/LLM, and EEGPT checkpoint paths.
