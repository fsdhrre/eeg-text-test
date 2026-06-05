# Figure Descriptions

## `01_method_pipeline`

Shows the full method pipeline:

```text
Raw EEG -> EEGPT Encoder -> Adapter / Semantic Heads
        -> Semantic Retrieval -> Evidence Decision
        -> Structured Prompt -> LLM Caption
```

Use this as the main method figure.

## `02_main_result_table`

Image version of the main result table. It summarizes caption generation metrics for the current comparison settings.

## `main_metrics_grouped_bar`

Grouped bar chart comparing key metrics across methods, such as object accuracy, token F1, BLEU, ROUGE-L, METEOR, CIDEr, and evidence faithfulness.

## `main_ablation_metric_heatmap`

Ablation heatmap showing how metric values change when different semantic evidence settings are used.

## `relative_improvement_over_high_only`

Relative improvement plot using the high-only setting as the baseline.

## `tsne_test_by_true_label`

t-SNE visualization of EEG classifier features on the full test split. Points are colored by true object label. This figure supports the claim that the EEG representation contains class-discriminative structure.

## `tsne_test_correct_vs_wrong`

t-SNE visualization of the same EEG classifier features, colored by whether the classifier prediction is correct.

## `07_per_class_classification_accuracy`

Per-class EEG classifier accuracy on the full test split. This shows which object categories are easier or harder to decode from EEG.

## `08_qualitative_improved_cases`

Qualitative examples where reliable low-level evidence improves the generated caption.

## `09_semantic_retrieval_diagnostics`

Semantic retrieval diagnostics, including semantic hit rate and head label agreement. This figure is useful for discussing whether semantic heads are specialized or still correlated.

