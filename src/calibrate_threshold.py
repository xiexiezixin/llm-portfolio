import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def top1_score(query, vectorizer, X):
    q = vectorizer.transform([query])
    scores = (X @ q.T).toarray().ravel()
    return float(scores.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=str, default="data/eval_questions.jsonl")
    ap.add_argument("--vec", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\tfidf_vectorizer.joblib")
    ap.add_argument("--mat", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\tfidf_matrix.joblib")
    ap.add_argument("--margin", type=float, default=0.001)
    ap.add_argument("--out", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\threshold.json")
    args = ap.parse_args()

    vectorizer = joblib.load(args.vec)
    X = joblib.load(args.mat)

    pos, neg = [], []
    with Path(args.eval).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            s = top1_score(item["question"], vectorizer, X)
            if bool(item.get("answerable", True)):
                pos.append(s)
            else:
                neg.append(s)

    pos = np.array(pos) if pos else np.array([0.0])
    neg = np.array(neg) if neg else np.array([0.0])

    thr = float(neg.max() + args.margin)

    report = {
        "top1_score_threshold": thr,
        "margin": args.margin,
        "pos": {
            "n": int(len(pos)),
            "min": float(pos.min()),
            "median": float(np.median(pos)),
            "p75": float(np.percentile(pos, 75)),
            "max": float(pos.max()),
        },
        "neg": {
            "n": int(len(neg)),
            "min": float(neg.min()),
            "median": float(np.median(neg)),
            "max": float(neg.max()),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[OK] saved:", args.out)
    print("recommended threshold =", thr)
    print("pos median =", report["pos"]["median"], "neg max =", report["neg"]["max"])


if __name__ == "__main__":
    main()