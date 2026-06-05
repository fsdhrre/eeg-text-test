# Paper Results Package

This folder contains the current paper-facing experimental artifacts.

## Folder Structure

```text
figures/
  Paper-ready PNG/PDF figures.

metrics/
  Main metric tables and JSON summaries.

examples/
  Semantic target examples, caption output examples, and diagnostic summaries.

docs/
  Human-readable descriptions of figures, metrics, semantic levels, and output columns.

qualitative_images/
  Eight image files used by the qualitative case table, with a CSV/README mapping image IDs to captions and scores.
```

## Current Main Experimental Setting

The current paper-oriented inference setting is:

```text
EEG
  -> EEGPT encoder + adapter/classifier
  -> classifier chooses the main object category
  -> low semantic retrieval provides reliable visual attributes
  -> high semantic retrieval provides scene/context evidence
  -> structured LLM prompt generates one caption
```

The latest 50-sample classifier-guided run is stored in:

```text
examples/stage4_classifier_main_label_low_high_llm_50.csv
```

The full-test old High + Reliable Low setting with low-threshold 0.35 is stored in:

```text
full_outputs/stage4_structured_old_high_reliable_low035_llm_full.csv
metrics/stage4_structured_old_high_reliable_low035_llm_full_semantic_metrics.csv
metrics/stage4_structured_old_high_reliable_low035_llm_full_semantic_metrics_summary.json
```

This is the main full-test output for the earlier High + Reliable Low retrieval setting.

Its metric summaries are:

```text
examples/stage4_classifier_main_label_low_high_llm_50_semantic_metrics_summary.json
examples/stage4_classifier_main_label_low_high_llm_50_templatefree_metrics_summary.json
```

## Important Metric Distinction

Two accuracy values should not be mixed:

- `EEG classifier accuracy`: computed before caption generation, measuring whether the EEG classifier predicts the correct object class.
- `caption object accuracy`: computed after LLM generation, measuring whether the generated caption contains the correct object category.

The t-SNE/classification figures report EEG classifier behavior. The caption result tables report final generated text behavior.
