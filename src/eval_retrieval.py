import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def load_chunks_text(chunks_jsonl: Path):
    # chunk_id -> text
    mp = {}
    with chunks_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mp[r["chunk_id"]] = r["text"]
    return mp


def retrieve_tfidf(query: str, topk: int, vectorizer, X):
    q = vectorizer.transform([query])
    scores = (X @ q.T).toarray().ravel()
    top_idx = np.argsort(-scores)[:topk]
    return top_idx.tolist(), scores[top_idx].tolist()


def hit_by_keywords(retrieved_texts, must_include):
    """
    粗评测：只要 topK 任何一个 chunk 文本包含 must_include 里的任意关键词（或多个），就算 hit
    更严格可以改成：至少命中 2 个关键词 / 必须全部命中
    """
    if not must_include:
        return False, []
    hits = []
    for kw in must_include:
        for t in retrieved_texts:
            if kw in t:
                hits.append(kw)
                break
    return (len(hits) > 0), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=str, default="../src/data/eval_questions.jsonl")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--vec", type=str, default="../src/outputs/tfidf_vectorizer.joblib")
    ap.add_argument("--mat", type=str, default="../src/outputs/tfidf_matrix.joblib")
    ap.add_argument("--chunks", type=str, default="../src/outputs/chunks.jsonl")
    ap.add_argument("--out", type=str, default="../src/outputs/retrieval_report.json")
    args = ap.parse_args()

    vectorizer = joblib.load(args.vec)
    X = joblib.load(args.mat)
    chunk_text = load_chunks_text(Path(args.chunks))

    results = []
    total = 0
    hit_cnt = 0

    with Path(args.eval).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["qid"]
            question = item["question"]
            must_include = item.get("must_include", [])

            top_idx, top_scores = retrieve_tfidf(question, args.topk, vectorizer, X)

            retrieved = []
            for rank, (i, s) in enumerate(zip(top_idx, top_scores), start=1):
                # 注意：这里 i 是“行号”，对应 chunk_id（因为你 meta 顺序就是 chunk_id 顺序）
                # 如果你后面用 meta 映射，就用 meta[i]["chunk_id"]
                txt = chunk_text.get(i, "")
                retrieved.append({
                    "rank": rank,
                    "row_id": i,
                    "score": float(s),
                    "preview": txt[:160].replace("\n", " ")
                })

            retrieved_texts = [chunk_text.get(i, "") for i in top_idx]
            hit, hit_keywords = hit_by_keywords(retrieved_texts, must_include)

            total += 1
            hit_cnt += int(hit)

            results.append({
                "qid": qid,
                "question": question,
                "must_include": must_include,
                "hit": hit,
                "hit_keywords": hit_keywords,
                "topk": retrieved
            })

    report = {
        "topk": args.topk,
        "total": total,
        "hit_count": hit_cnt,
        "hit_rate": hit_cnt / max(total, 1),
        "details": results
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] total={total} hit={hit_cnt} hit_rate={hit_cnt/max(total,1):.3f}")
    # 打印前5个失败样例
    fails = [r for r in results if not r["hit"]]
    print(f"[FAIL] {len(fails)}")
    for r in fails[:5]:
        print(f"- qid={r['qid']} q={r['question']} must={r['must_include']}")

if __name__ == "__main__":
    main()