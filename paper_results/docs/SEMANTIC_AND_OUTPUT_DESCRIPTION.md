# Semantic Levels and Output Files

## Semantic Level Definitions

Each image is converted into structured semantic text at multiple levels.

### Low Semantic

Low semantic evidence describes local visual attributes:

```text
color, texture, shape, size, material, local appearance
```

Example:

```text
Visual attributes of the camera: black, wooden.
```

### Mid Semantic

Mid semantic evidence describes object and action/state:

```text
main object, object state, action, pose, spatial layout
```

Example:

```text
Main object and action: canoe or boat; navigates.
```

### High Semantic

High semantic evidence describes scene and global context:

```text
environment, scene, event, global situation, contextual meaning
```

Example:

```text
Scene and context for the piano: a man in a red shirt sits at a piano in front of a Christmas tree.
```

## Current Inference Design

The current classifier-guided generation setting uses:

```text
classifier prediction -> main object category
low retrieval          -> reliable attributes
high retrieval         -> scene/context
LLM                    -> final one-sentence caption
```

This prevents low/high retrieval evidence from replacing the main object category. Instead, retrieved semantics enrich the description around the classifier-selected object.

## Important Output Columns

In `stage4_classifier_main_label_low_high_llm_50.csv`:

- `image_name`: test image identifier.
- `reference_caption`: reference structured caption.
- `true_object`: ground-truth object category.
- `classifier_pred`: object predicted by the EEG classifier.
- `classifier_correct`: whether classifier prediction is correct.
- `top_low_label`: label of the top low-level retrieved anchor.
- `top_low_score`: similarity score of the top low-level anchor.
- `top_high_label`: label of the top high-level retrieved anchor.
- `top_high_score`: similarity score of the top high-level anchor.
- `main_label`: object category used by the final prompt.
- `decision_reason`: reason for selecting the main label.
- `generated_caption`: final LLM-generated caption.
- `prompt`: full structured prompt sent to the LLM.

