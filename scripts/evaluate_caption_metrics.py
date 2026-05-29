"""用词面指标和语义指标评估生成 caption。

输入是阶段四生成的 CSV，每个 EEG 样本包含 reference caption 和 generated caption。
脚本会给每一行添加指标，并额外写出一个紧凑的 JSON summary，方便放进论文表格。

实现的指标：
    - token F1，如果输入 CSV 中已经包含该列
    - reference 和 generated caption 之间的 CLIP text similarity
    - 可选的 Sentence-BERT / transformer 文本相似度，本地有模型时才计算
    - BLEU-1/2/3/4 以及 corpus BLEU-4
    - ROUGE-L
    - 类 METEOR 的 unigram 对齐分数
    - 类 CIDEr 的 TF-IDF n-gram 余弦分数
"""

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPTextModelWithProjection

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from eeg_text_codex.config import PathConfig, TrainConfig
from eeg_text_codex.utils import get_device


def parse_args():
    """定义输入输出文件，以及可选的本地文本编码器路径。"""

    parser = argparse.ArgumentParser(description="Evaluate generated captions with text similarity metrics.")
    parser.add_argument("--input_csv", default=os.path.join(PathConfig.staged_output_dir, "stage4_structured_retrieval_full_evidence_llm.csv"))
    parser.add_argument("--output_csv", default=os.path.join(PathConfig.staged_output_dir, "stage4_structured_retrieval_full_evidence_llm_metrics.csv"))
    parser.add_argument("--summary_json", default=os.path.join(PathConfig.staged_output_dir, "stage4_structured_retrieval_full_evidence_llm_metrics_summary.json"))
    parser.add_argument("--reference_col", default="reference_caption")
    parser.add_argument("--generated_col", default="generated_caption")
    parser.add_argument("--clip_path", default=PathConfig.clip_path)
    parser.add_argument("--sentence_model_path", default="", help="Optional local Sentence-BERT/transformer model path.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=TrainConfig.device)
    return parser.parse_args()


def tokenize(text):
    """词面指标共用的简单 tokenizer：转小写并保留字母数字。"""

    return re.findall(r"[a-z0-9]+", str(text).lower())


def ngrams(tokens, n):
    """把 token 序列转成连续 n-gram tuple。"""

    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def sentence_bleu(reference, hypothesis, max_n=4, smooth=1.0):
    """句子级 BLEU，使用简单加一平滑。"""

    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        ref_counts = Counter(ngrams(ref, n))
        hyp_counts = Counter(ngrams(hyp, n))
        overlap = sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
        total = sum(hyp_counts.values())
        precisions.append((overlap + smooth) / (total + smooth))
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1.0 - len(ref) / max(1, len(hyp)))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def corpus_bleu(references, hypotheses, max_n=4, smooth=1.0):
    """基于全局 n-gram 计数计算 corpus BLEU。"""

    matches = [0.0] * max_n
    totals = [0.0] * max_n
    ref_len = 0
    hyp_len = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref = tokenize(reference)
        hyp = tokenize(hypothesis)
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            ref_counts = Counter(ngrams(ref, n))
            hyp_counts = Counter(ngrams(hyp, n))
            matches[n - 1] += sum(min(count, ref_counts[gram]) for gram, count in hyp_counts.items())
            totals[n - 1] += sum(hyp_counts.values())
    precisions = [(matches[i] + smooth) / (totals[i] + smooth) for i in range(max_n)]
    bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / max(1, hyp_len))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def lcs_len(a, b):
    """计算 ROUGE-L 所需的最长公共子序列长度。"""

    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(reference, hypothesis):
    """基于最长公共子序列的 ROUGE-L F-score。"""

    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    lcs = lcs_len(ref, hyp)
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    beta = 1.2
    return ((1 + beta * beta) * precision * recall) / (recall + beta * beta * precision)


def meteor_like(reference, hypothesis):
    """轻量版 METEOR 风格分数。

    这不是官方 METEOR 实现，只保留核心思想：unigram precision / recall，
    再加一个片段化惩罚。
    """

    ref = tokenize(reference)
    hyp = tokenize(hypothesis)
    if not ref or not hyp:
        return 0.0
    ref_positions = defaultdict(list)
    for i, token in enumerate(ref):
        ref_positions[token].append(i)
    used = set()
    matches = []
    for token in hyp:
        positions = ref_positions.get(token, [])
        chosen = None
        for pos in positions:
            if pos not in used:
                chosen = pos
                break
        if chosen is not None:
            used.add(chosen)
            matches.append(chosen)
    m = len(matches)
    if m == 0:
        return 0.0
    precision = m / len(hyp)
    recall = m / len(ref)
    fmean = (10 * precision * recall) / (recall + 9 * precision) if precision + recall > 0 else 0.0
    chunks = 1
    for i in range(1, len(matches)):
        if matches[i] != matches[i - 1] + 1:
            chunks += 1
    penalty = 0.5 * (chunks / m) ** 3
    return fmean * (1 - penalty)


def cider_scores(references, hypotheses, max_n=4):
    """轻量版 CIDEr 风格分数：使用 TF-IDF n-gram 余弦相似度。"""

    tokenized_refs = [tokenize(text) for text in references]
    tokenized_hyps = [tokenize(text) for text in hypotheses]
    n_docs = len(tokenized_refs)
    dfs = [Counter() for _ in range(max_n)]
    for ref in tokenized_refs:
        for n in range(1, max_n + 1):
            dfs[n - 1].update(set(ngrams(ref, n)))

    def tfidf(tokens, n):
        counts = Counter(ngrams(tokens, n))
        total = sum(counts.values())
        if total == 0:
            return {}
        vec = {}
        for gram, count in counts.items():
            idf = math.log((n_docs + 1.0) / (dfs[n - 1].get(gram, 0) + 1.0))
            vec[gram] = (count / total) * idf
        return vec

    def cosine(vec_a, vec_b):
        if not vec_a or not vec_b:
            return 0.0
        dot = sum(value * vec_b.get(key, 0.0) for key, value in vec_a.items())
        norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
        norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    scores = []
    for ref, hyp in zip(tokenized_refs, tokenized_hyps):
        sims = []
        for n in range(1, max_n + 1):
            sims.append(cosine(tfidf(ref, n), tfidf(hyp, n)))
        scores.append(10.0 * sum(sims) / max_n)
    return scores


@torch.no_grad()
def encode_clip_texts(texts, tokenizer, model, device, batch_size):
    """使用 CLIP text encoder 编码文本，并对 embedding 做 L2 归一化。"""

    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP text encode", leave=False):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        chunks.append(F.normalize(model(**inputs).text_embeds.float(), dim=-1).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_sentence_texts(texts, tokenizer, model, device, batch_size):
    """使用通用 transformer 编码文本，对 token hidden states 做 mean pooling。"""

    chunks = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Sentence text encode", leave=False):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state.float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        chunks.append(F.normalize(pooled, dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def mean_or_none(values):
    """忽略 NaN 后返回 Python float 均值。"""

    values = [v for v in values if not pd.isna(v)]
    return float(sum(values) / len(values)) if values else None


def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    references = df[args.reference_col].fillna("").astype(str).tolist()
    hypotheses = df[args.generated_col].fillna("").astype(str).tolist()

    # 先计算词面指标，这些指标不需要加载神经网络模型。
    df["bleu1"] = [sentence_bleu(r, h, max_n=1) for r, h in zip(references, hypotheses)]
    df["bleu2"] = [sentence_bleu(r, h, max_n=2) for r, h in zip(references, hypotheses)]
    df["bleu3"] = [sentence_bleu(r, h, max_n=3) for r, h in zip(references, hypotheses)]
    df["bleu4"] = [sentence_bleu(r, h, max_n=4) for r, h in zip(references, hypotheses)]
    df["rouge_l"] = [rouge_l(r, h) for r, h in zip(references, hypotheses)]
    df["meteor"] = [meteor_like(r, h) for r, h in zip(references, hypotheses)]
    df["cider"] = cider_scores(references, hypotheses)

    # 语义文本相似度由冻结文本编码器计算。CLIP 必算，因为它和训练/检索时的语义空间一致。
    device = get_device(args.device)
    clip_tokenizer = AutoTokenizer.from_pretrained(args.clip_path, local_files_only=True)
    clip_model = CLIPTextModelWithProjection.from_pretrained(args.clip_path, local_files_only=True).to(device).eval()
    ref_clip = encode_clip_texts(references, clip_tokenizer, clip_model, device, args.batch_size)
    hyp_clip = encode_clip_texts(hypotheses, clip_tokenizer, clip_model, device, args.batch_size)
    df["clip_text_similarity"] = (ref_clip * hyp_clip).sum(dim=-1).tolist()

    sentence_available = False
    if args.sentence_model_path:
        # Sentence-BERT 是可选项，因为项目默认只使用本地模型文件，很多机器未必有该模型。
        sentence_tokenizer = AutoTokenizer.from_pretrained(args.sentence_model_path, local_files_only=True)
        sentence_model = AutoModel.from_pretrained(args.sentence_model_path, local_files_only=True).to(device).eval()
        ref_sent = encode_sentence_texts(references, sentence_tokenizer, sentence_model, device, args.batch_size)
        hyp_sent = encode_sentence_texts(hypotheses, sentence_tokenizer, sentence_model, device, args.batch_size)
        df["sentence_bert_similarity"] = (ref_sent * hyp_sent).sum(dim=-1).tolist()
        sentence_available = True
    else:
        df["sentence_bert_similarity"] = float("nan")

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    # JSON summary 是论文表格和消融实验中最常用的紧凑结果文件。
    summary = {
        "input_csv": args.input_csv,
        "rows": int(len(df)),
        "sentence_bert_available": sentence_available,
        "metrics": {
            "token_f1": mean_or_none(df["token_f1"].tolist()) if "token_f1" in df else None,
            "clip_text_similarity": float(df["clip_text_similarity"].mean()),
            "sentence_bert_similarity": mean_or_none(df["sentence_bert_similarity"].tolist()),
            "bleu1_sentence_avg": float(df["bleu1"].mean()),
            "bleu2_sentence_avg": float(df["bleu2"].mean()),
            "bleu3_sentence_avg": float(df["bleu3"].mean()),
            "bleu4_sentence_avg": float(df["bleu4"].mean()),
            "bleu4_corpus": corpus_bleu(references, hypotheses, max_n=4),
            "rouge_l": float(df["rouge_l"].mean()),
            "meteor": float(df["meteor"].mean()),
            "cider": float(df["cider"].mean()),
        },
    }
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved per-sample metrics to {args.output_csv}")
    print(f"Saved summary to {args.summary_json}")


if __name__ == "__main__":
    main()
