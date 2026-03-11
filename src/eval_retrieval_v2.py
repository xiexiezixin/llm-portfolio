import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def load_chunks_text(chunks_jsonl: Path):
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


def keyword_hit(retrieved_texts, must_include, rule: str):
    if not must_include:
        return False, []
    hit_kws = []
    for kw in must_include:
        found = any(kw in t for t in retrieved_texts)
        if found:
            hit_kws.append(kw)

    if rule == "all":
        hit = (len(hit_kws) == len(must_include))
    elif rule == "at_least_2":
        hit = (len(hit_kws) >= 2)
    else:  # any
        hit = (len(hit_kws) >= 1)
    return hit, hit_kws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\data\\eval_questions.jsonl")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--rule", type=str, default="all", choices=["any", "at_least_2", "all"])
    ap.add_argument("--vec", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\tfidf_vectorizer.joblib")
    ap.add_argument("--mat", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\tfidf_matrix.joblib")
    ap.add_argument("--chunks", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\chunks.jsonl")
    ap.add_argument("--out", type=str, default="E:\\xzx\\code\\SAR_python\\LLM\\llm-portfolio\\src\\outputs\\retrieval_report_v2.json")
    args = ap.parse_args()

    vectorizer = joblib.load(args.vec)
    X = joblib.load(args.mat)
    chunk_text = load_chunks_text(Path(args.chunks))

    details = []
    total_ans = total_unans = 0
    hit_ans = falsehit_unans = 0

    with Path(args.eval).open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            qid = item["qid"]
            question = item["question"]
            must_include = item.get("must_include", [])
            answerable = bool(item.get("answerable", True))

            top_idx, top_scores = retrieve_tfidf(question, args.topk, vectorizer, X)
            retrieved_texts = [chunk_text.get(i, "") for i in top_idx]
            hit, hit_kws = keyword_hit(retrieved_texts, must_include, args.rule)

            if answerable:
                total_ans += 1
                hit_ans += int(hit)
            else:
                total_unans += 1
                falsehit_unans += int(hit)

            details.append({
                "qid": qid,
                "answerable": answerable,
                "question": question,
                "must_include": must_include,
                "hit": hit,
                "hit_keywords": hit_kws,
                "top_idx": top_idx,
                "top_scores": [float(s) for s in top_scores],
                "top_preview": [chunk_text.get(i, "")[:120].replace("\n", " ") for i in top_idx],
            })

    report = {
        "topk": args.topk,
        "rule": args.rule,
        "answerable_total": total_ans,
        "answerable_hit": hit_ans,
        "hit_rate_answerable": hit_ans / max(total_ans, 1),
        "unanswerable_total": total_unans,
        "unanswerable_falsehit": falsehit_unans,
        "false_hit_rate_unanswerable": falsehit_unans / max(total_unans, 1),
        "details": details
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] rule={args.rule} topk={args.topk}")
    print(f"  Answerable: {hit_ans}/{total_ans} hit_rate={report['hit_rate_answerable']:.3f}")
    print(f"  Unanswerable: falsehit {falsehit_unans}/{total_unans} false_hit_rate={report['false_hit_rate_unanswerable']:.3f}")

    # 打印前3个“不可回答但命中”的问题（最危险）
    risky = [d for d in details if (not d["answerable"]) and d["hit"]]
    if risky:
        print("[RISKY] Unanswerable but hit:")
        for d in risky[:3]:
            print(f"- qid={d['qid']} q={d['question']} hit_kws={d['hit_keywords']}")

if __name__ == "__main__":
    main()