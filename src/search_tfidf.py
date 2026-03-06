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
            mp[r["chunk_id"]] = r["text"]
    return mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--vec", type=str, default="../src/outputs/tfidf_vectorizer.joblib")
    ap.add_argument("--mat", type=str, default="../src/outputs/tfidf_matrix.joblib")
    ap.add_argument("--meta", type=str, default="../src/outputs/meta.jsonl")
    ap.add_argument("--chunks", type=str, default="../src/outputs/chunks.jsonl")
    args = ap.parse_args()

    vectorizer = joblib.load(args.vec)
    X = joblib.load(args.mat)  # (n_chunks, vocab) sparse
    metas = load_meta(Path(args.meta))
    chunk_text = load_text_by_chunk(Path(args.chunks))

    q = vectorizer.transform([args.query])  # sparse
    # TF-IDF 默认 L2 归一化，余弦相似度 = 点积
    scores = (X @ q.T).toarray().ravel()

    top_idx = np.argsort(-scores)[:args.topk]

    print(f"[QUERY] {args.query}")
    for rank, i in enumerate(top_idx, start=1):
        m = metas[i]
        s = float(scores[i])
        text = chunk_text.get(m["chunk_id"], "")
        preview = text[:140].replace("\n", " ")
        print(f"\n#{rank} score={s:.4f} chunk_id={m['chunk_id']} range=[{m['start']}:{m['end']}]")
        print(f"source={m['source']}")
        print(f"preview={preview}...")


if __name__ == "__main__":
    main()