# Metric Descriptions

## Object Accuracy

Measures whether the predicted/generated object category matches the ground-truth object label.

For classifier analysis:

```text
classifier_pred == true_object
```

For caption analysis:

```text
object extracted from generated caption == true_object
```

## Image-Text CLIPScore

Computes CLIP similarity between the original image and the generated caption. Higher values indicate better image-text semantic consistency.

## CLIP Text Similarity

Computes CLIP text embedding similarity between the reference caption and the generated caption.

## Token F1

Word-level overlap F1 between reference and generated caption.

## BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4

n-gram precision metrics. BLEU-1 is more forgiving; BLEU-4 is stricter and often lower for short captions.

## ROUGE-L

Longest common subsequence based overlap between reference and generated caption.

## METEOR-like

A lightweight METEOR-style score emphasizing unigram alignment with a balance between precision and recall.

## CIDEr-like

A captioning metric based on weighted n-gram similarity. It rewards captions that share important content words with references.

## Cue Coverage

Measures whether retrieved semantic cues appear in the generated caption.

## Evidence Faithfulness

Measures whether the generated caption stays faithful to reliable retrieved evidence rather than adding unsupported details.

