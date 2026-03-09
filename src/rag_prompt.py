import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def load_meta(meta_path: Path):
    metas = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def load_text_by_chunk(chunks_jsonl: Path):
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


def retrieve_tfidf(query: str, topk: int,
                   vec_path: Path, mat_path: Path,
                   meta_path: Path, chunks_path: Path):
    vectorizer = joblib.load(vec_path)
    X = joblib.load(mat_path)  # sparse matrix (n_chunks, vocab)
    metas = load_meta(meta_path)
    chunk_map = load_text_by_chunk(chunks_path)

    q = vectorizer.transform([query])
    scores = (X @ q.T).toarray().ravel()
    top_idx = np.argsort(-scores)[:topk]

    results = []
    for i in top_idx:
        m = metas[i]
        cid = m["chunk_id"]
        cinfo = chunk_map.get(cid, {})
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[i]),
            "chunk_id": cid,
            "source": m["source"],
            "span": [m["start"], m["end"]],
            "text": cinfo.get("text", "")
        })
    return results


def build_prompt(query: str, retrieved, max_chars_per_chunk: int = 800) -> str:
    # 证据区：每条都带 chunk_id，要求回答引用 [chunk_id]
    ctx_blocks = []
    for r in retrieved:
        txt = (r["text"] or "").strip().replace("\n", " ")
        if len(txt) > max_chars_per_chunk:
            txt = txt[:max_chars_per_chunk] + "…"
        ctx_blocks.append(
            f"[chunk_id={r['chunk_id']} | score={r['score']:.4f} | source={r['source']} | span={r['span'][0]}:{r['span'][1]}]\n"
            f"{txt}\n"
        )
    ctx = "\n".join(ctx_blocks)

    prompt = f"""你是一个严谨的技术助理。请仅基于【给定证据】回答【用户问题】，不要编造。
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
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_chars", type=int, default=800)
    ap.add_argument("--vec", type=str, default="../src/outputs/tfidf_vectorizer.joblib")
    ap.add_argument("--mat", type=str, default="../src/outputs/tfidf_matrix.joblib")
    ap.add_argument("--meta", type=str, default="../src/outputs/meta.jsonl")
    ap.add_argument("--chunks", type=str, default="../src/outputs/chunks.jsonl")
    ap.add_argument("--out_prompt", type=str, default="../src/outputs/rag_prompt.txt")
    ap.add_argument("--out_retrieved", type=str, default="../src/outputs/retrieved.json")
    args = ap.parse_args()

    retrieved = retrieve_tfidf(
        query=args.query,
        topk=args.topk,
        vec_path=Path(args.vec),
        mat_path=Path(args.mat),
        meta_path=Path(args.meta),
        chunks_path=Path(args.chunks),
    )

    prompt = build_prompt(args.query, retrieved, max_chars_per_chunk=args.max_chars)

    Path(args.out_prompt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_retrieved).parent.mkdir(parents=True, exist_ok=True)

    Path(args.out_prompt).write_text(prompt, encoding="utf-8")
    Path(args.out_retrieved).write_text(json.dumps(retrieved, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] topk={len(retrieved)} prompt_saved={args.out_prompt} retrieved_saved={args.out_retrieved}")
    for r in retrieved[:3]:
        preview = (r["text"] or "").replace("\n", " ")[:80]
        print(f"  rank#{r['rank']} score={r['score']:.4f} chunk_id={r['chunk_id']} preview={preview}...")


if __name__ == "__main__":
    main()