import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_chunks(jsonl_path: Path):
    metas, texts = [], []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            metas.append({
                "source": r["source"],
                "chunk_id": r["chunk_id"],
                "start": r["start"],
                "end": r["end"],
            })
            texts.append(r["text"])
    return metas, texts


def save_meta(metas, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=str, default="../src/outputs/chunks.jsonl")
    ap.add_argument("--out_vec", type=str, default="../src/outputs/tfidf_vectorizer.joblib")
    ap.add_argument("--out_mat", type=str, default="../src/outputs/tfidf_matrix.joblib")
    ap.add_argument("--out_meta", type=str, default="../src/outputs/meta.jsonl")
    args = ap.parse_args()

    metas, texts = load_chunks(Path(args.chunks))
    print(f"[LOAD] chunks={len(texts)}")

    # 中文/英文混合也能用：先用字符 ngram（对中文更友好）
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_features=200000
    )
    X = vectorizer.fit_transform(texts)  # 稀疏矩阵

    Path(args.out_vec).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, args.out_vec)
    joblib.dump(X, args.out_mat)
    save_meta(metas, Path(args.out_meta))

    print(f"[OK] vocab={len(vectorizer.vocabulary_)} matrix_shape={X.shape}")
    print(f"[SAVE] {args.out_vec}")
    print(f"[SAVE] {args.out_mat}")
    print(f"[SAVE] {args.out_meta}")


if __name__ == "__main__":
    main()