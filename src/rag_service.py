import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


@dataclass
class RAGConfig:
    topk: int = 5
    max_chars: int = 800


class RAGService:
    def __init__(self, vec_path: Path, mat_path: Path, chunks_path: Path, threshold_path: Path):
        self.vectorizer = joblib.load(vec_path)
        self.X = joblib.load(mat_path)  # sparse matrix
        self.chunk_map = self._load_chunks_map(chunks_path)
        self.threshold = json.loads(threshold_path.read_text(encoding="utf-8"))["top1_score_threshold"]

    @staticmethod
    def _load_chunks_map(chunks_jsonl: Path):
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

    def _retrieve(self, query: str, topk: int):
        q = self.vectorizer.transform([query])
        scores = (self.X @ q.T).toarray().ravel()
        top_idx = np.argsort(-scores)[:topk]
        top_scores = scores[top_idx]
        top1_score = float(scores.max()) if scores.size > 0 else 0.0
        return top_idx.tolist(), [float(s) for s in top_scores.tolist()], top1_score

    def _build_prompt(self, query: str, retrieved, max_chars: int):
        blocks = []
        for r in retrieved:
            txt = (r["text"] or "").strip().replace("\n", " ")
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

    def answer(self, query: str, cfg: RAGConfig):
        top_idx, top_scores, top1_score = self._retrieve(query, cfg.topk)

        retrieved = []
        for i, s in zip(top_idx, top_scores):
            c = self.chunk_map.get(i, {"text": "", "source": "", "start": 0, "end": 0})
            retrieved.append({
                "chunk_id": i,
                "score": float(s),
                "source": c["source"],
                "start": c["start"],
                "end": c["end"],
                "text": c["text"],
            })

        mode = "RAG_PROMPT" if top1_score >= self.threshold else "REFUSAL"

        resp = {
            "query": query,
            "mode": mode,
            "top1_score": top1_score,
            "threshold": self.threshold,
            "topk": cfg.topk,
            "retrieved": [
                {
                    "rank": r_i + 1,
                    "chunk_id": r["chunk_id"],
                    "score": r["score"],
                    "source": r["source"],
                    "span": [r["start"], r["end"]],
                    "preview": r["text"][:160].replace("\n", " ")
                }
                for r_i, r in enumerate(retrieved)
            ],
            "next_action": None,
            "prompt": None
        }

        if mode == "REFUSAL":
            resp["next_action"] = "NEED_MORE_EVIDENCE"
            resp["prompt"] = (
                f"证据不足：top1_score={top1_score:.4f} < threshold={self.threshold:.4f}。\n"
                f"建议：补充包含该参数/答案的文档片段，或扩充知识库。"
            )
        else:
            resp["next_action"] = "CALL_LLM_WITH_PROMPT"
            resp["prompt"] = self._build_prompt(query, retrieved, cfg.max_chars)

        return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", type=str, default="src/outputs/demo_answer.json")
    args = ap.parse_args()

    service = RAGService(
        vec_path=Path("src/outputs/tfidf_vectorizer.joblib"),
        mat_path=Path("src/outputs/tfidf_matrix.joblib"),
        chunks_path=Path("src/outputs/chunks.jsonl"),
        threshold_path=Path("src/outputs/threshold.json"),
    )

    resp = service.answer(args.query, RAGConfig(topk=args.topk))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] mode={resp['mode']} top1_score={resp['top1_score']:.4f} thr={resp['threshold']:.4f} saved={args.out}")


if __name__ == "__main__":
    main()