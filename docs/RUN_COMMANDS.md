# Run Commands

Set project root:

```bash
cd /path/to/eeg-text-codex
```

Use your own Python environment path if different.

## Stage 1: EEGPT Adapter Classifier

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage1_train_eegpt_adapter_classifier.py \
  --eeg_encoder_type eegpt \
  --num_epochs 50 \
  --batch_size 32 \
  --output_dir outputs/stage1_eegpt_adapter
```

This stage loads pretrained EEGPT, freezes the backbone, and trains adapter/classifier parameters.

## Generate Structured Captions With Qwen-VL

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/generate_qwen_structured_captions.py \
  --splits all \
  --output_json outputs/qwen_structured_captions_v3.json \
  --output_csv outputs/qwen_structured_captions_v3.csv
```

Normalize the captions:

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/normalize_qwen_structured_captions.py \
  --input_json outputs/qwen_structured_captions_v3.json \
  --input_csv outputs/qwen_structured_captions_v3.csv \
  --output_json outputs/qwen_structured_captions_v3.json \
  --output_csv outputs/qwen_structured_captions_v3.csv
```

## Build Structured Semantic Database

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/build_structured_semantic_database.py \
  --caption_map_path outputs/qwen_structured_captions_v3.json \
  --output_path outputs/structured_semantic_targets_all.pt \
  --summary_path outputs/structured_semantic_targets_all_smoke.json
```

## Stage 2: Multi-Head Structured Semantic Alignment

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage2_train_structured_semantics.py \
  --target_db_path outputs/structured_semantic_targets_all.pt \
  --eeg_encoder_path outputs/stage1_eegpt_adapter/best \
  --output_dir outputs/stage2_eegpt_structured \
  --num_epochs 50 \
  --batch_size 32
```

## Stage 4: Retrieval Only

Use this to inspect retrieval/evidence without LLM generation:

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage4_retrieval_infer.py \
  --checkpoint_dir outputs/stage2_eegpt_structured/best \
  --semantic_db_path outputs/structured_semantic_targets_all.pt \
  --output_csv outputs/stage4_structured_retrieval_full_evidence_skip.csv \
  --anchor_mode evidence \
  --max_samples -1 \
  --skip_llm
```

## Stage 4: Structured Prompt LLM Generation

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage4_retrieval_infer.py \
  --checkpoint_dir outputs/stage2_eegpt_structured/best \
  --semantic_db_path outputs/structured_semantic_targets_all.pt \
  --output_csv outputs/stage4_structured_retrieval_full_evidence_structured_prompt_llm.csv \
  --anchor_mode evidence \
  --generation_prompt_style structured \
  --max_samples -1
```

## Caption Metrics

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/evaluate_caption_metrics.py \
  --input_csv outputs/stage4_structured_retrieval_full_evidence_structured_prompt_llm.csv \
  --output_csv outputs/stage4_structured_retrieval_full_evidence_structured_prompt_llm_metrics.csv \
  --summary_json outputs/stage4_structured_retrieval_full_evidence_structured_prompt_llm_metrics_summary.json
```

## Ablations

Remove low semantic branch:

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage4_retrieval_infer.py \
  --checkpoint_dir outputs/stage2_eegpt_structured/best \
  --semantic_db_path outputs/structured_semantic_targets_all.pt \
  --output_csv outputs/ablation_no_low_evidence_llm.csv \
  --anchor_mode evidence \
  --semantic_levels mid high \
  --generation_prompt_style structured \
  --max_samples -1
```

Remove mid semantic branch:

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage4_retrieval_infer.py \
  --checkpoint_dir outputs/stage2_eegpt_structured/best \
  --semantic_db_path outputs/structured_semantic_targets_all.pt \
  --output_csv outputs/ablation_no_mid_evidence_llm.csv \
  --anchor_mode evidence \
  --semantic_levels low high \
  --generation_prompt_style structured \
  --max_samples -1
```

Remove high semantic branch:

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage4_retrieval_infer.py \
  --checkpoint_dir outputs/stage2_eegpt_structured/best \
  --semantic_db_path outputs/structured_semantic_targets_all.pt \
  --output_csv outputs/ablation_no_high_evidence_llm.csv \
  --anchor_mode evidence \
  --semantic_levels low mid \
  --generation_prompt_style structured \
  --max_samples -1
```

Remove evidence decision:

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage4_retrieval_infer.py \
  --checkpoint_dir outputs/stage2_eegpt_structured/best \
  --semantic_db_path outputs/structured_semantic_targets_all.pt \
  --output_csv outputs/ablation_no_evidence_decision_llm.csv \
  --anchor_mode topk_per_level \
  --generation_prompt_style structured \
  --max_samples -1
```

Run metrics on each ablation by replacing the input/output names:

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/evaluate_caption_metrics.py \
  --input_csv outputs/ablation_no_mid_evidence_llm.csv \
  --output_csv outputs/ablation_no_mid_evidence_llm_metrics.csv \
  --summary_json outputs/ablation_no_mid_evidence_llm_metrics_summary.json
```

## Optional Stage 5: Candidate Reranker

```bash
/home/dell/anaconda3/envs/thought2text-env/bin/python scripts/stage5_train_candidate_reranker.py \
  --checkpoint_dir outputs/stage2_eegpt_structured/best \
  --semantic_db_path outputs/structured_semantic_targets_all.pt \
  --output_dir outputs/stage5_candidate_reranker \
  --num_epochs 20
```

This stage is optional. In the current experiments, evidence decision performed better than the learned reranker.
