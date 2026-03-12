import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def load_chunks_map(chunks_jsonl: Path):
    mp = {}
    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mp[r["chunk_id"]] = {
                "text": r["text"],
                "source": r["source"],
                "start": r["start"],
                "end": r["end"],
            }
    return mp


def retrieve(query, topk, vectorizer, X):
    q = vectorizer.transform([query])
    scores = (X @ q.T).toarray().ravel()
    top_idx = np.argsort(-scores)[:topk]
    top_scores = scores[top_idx]
    return top_idx.tolist(), top_scores.tolist(), float(scores.max())


def build_rag_prompt(query, retrieved, max_chars=800):
    blocks = []
    for r in retrieved:
        txt = r["text"].strip().replace("\n", " ")
        if len(txt) > max_chars:
            txt = txt[:max_chars] + "…"
        blocks.append(
            f"[chunk_id={r['chunk_id']} | score={r['score']:.4f} | source={r['source']} | span={r['start']}:{r['end']}]\n{txt}\n"
        )
    ctx = "\n".join(blocks)

    return f"""你是一个严谨的技术助理。请仅基于【给定证据】回答【用户问题】，不要编造。
如果证据不足，请明确说“证据不足”，并指出缺少什么信息。

引用规则（非常重要）：
- 你给出的每个关键结论后面都必须标注引用，格式为 [chunk_id]，可多个，如 [12][15]。
- 引用必须来自【给定证据】中的 chunk_id。

输出要求：
1) 先用 3–6 条要点回答（每条都带引用）
2) 再给一段简短总结（也带引用）
3) 如果有不确定/缺口，单独列出“信息缺口”

【给定证据】
{ctx}

【用户问题】
{query}

请用中文回答。
"""


def build_insufficient_answer(query, top1_score, threshold, top1_chunk):
    # 注意：这里不是让模型回答，而是直接给用户一个“拒答模板”
    preview = top1_chunk["text"].replace("\n", " ")[:200]
    return f"""证据不足：当前检索到的最高相关性分数为 {top1_score:.4f}，低于阈值 {threshold:.4f}，无法可靠回答该问题。

用户问题：{query}

最相关证据（供核查）：
- chunk_id={top1_chunk['chunk_id']} score={top1_chunk['score']:.4f} source={top1_chunk['source']} span={top1_chunk['start']}:{top1_chunk['end']}
- preview: {preview}...

建议补充信息：
- 请提供包含该问题答案的文档片段/参数来源，或扩大知识库覆盖范围（加入包含该参数的文档）。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--rule", type=str, default="gate_only", choices=["gate_only"])
    ap.add_argument("--threshold_file", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\threshold.json")
    ap.add_argument("--vec", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\tfidf_vectorizer.joblib")
    ap.add_argument("--mat", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\tfidf_matrix.joblib")
    ap.add_argument("--chunks", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\chunks.jsonl")
    ap.add_argument("--out", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\rag_prompt_or_refusal.txt")
    args = ap.parse_args()

    vectorizer = joblib.load(args.vec)
    X = joblib.load(args.mat)
    chunk_map = load_chunks_map(Path(args.chunks))

    threshold = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))["top1_score_threshold"]

    top_idx, top_scores, top1_score = retrieve(args.query, args.topk, vectorizer, X)

    retrieved = []
    for i, s in zip(top_idx, top_scores):
        c = chunk_map.get(i, {"text": "", "source": "", "start": 0, "end": 0})
        retrieved.append({
            "chunk_id": i,
            "score": float(s),
            "source": c["source"],
            "start": c["start"],
            "end": c["end"],
            "text": c["text"],
        })

    if retrieved:
        top1_chunk = retrieved[0]
    else:
        top1_chunk = {"chunk_id": -1, "score": 0.0, "source": "", "start": 0, "end": 0, "text": ""}

    if top1_score < threshold:
        out_text = build_insufficient_answer(args.query, top1_score, threshold, top1_chunk)
        mode = "REFUSAL"
    else:
        out_text = build_rag_prompt(args.query, retrieved)
        mode = "RAG_PROMPT"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(out_text, encoding="utf-8")
    print(f"[OK] mode={mode} top1_score={top1_score:.4f} threshold={threshold:.4f} saved={args.out}")


if __name__ == "__main__":
    main()