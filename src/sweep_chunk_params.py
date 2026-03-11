import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]   # .../llm-portfolio
PY = sys.executable                          # 当前正在运行的 python（venv）


def run(cmd):
    r = subprocess.run(
        cmd,
        cwd=str(ROOT),               # 关键：所有子进程都在 repo 根目录执行
        capture_output=True,
        text=True,
        shell=False
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"CMD failed: {' '.join(map(str, cmd))}\n"
            f"cwd: {ROOT}\n"
            f"STDOUT:\n{r.stdout}\n"
            f"STDERR:\n{r.stderr}"
        )
    return r.stdout + "\n" + r.stderr


def as_root_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (ROOT / pp).resolve()


def read_report(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="../llm-portfolio/src/data/sample.txt")
    ap.add_argument("--eval", type=str, default="../llm-portfolio/src/data/eval_questions.jsonl")
    ap.add_argument("--rule", type=str, default="all", choices=["any", "at_least_2", "all"])
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out_csv", type=str, default="../llm-portfolio/src/outputs/sweep_table.csv")
    args = ap.parse_args()

    input_path = as_root_path(args.input)
    eval_path = as_root_path(args.eval)
    out_csv = as_root_path(args.out_csv)

    # 1小时内跑完：先扫小一点
    chunk_sizes = [200, 400, 800]
    overlaps = [0, 50, 100]

    rows = []
    for cs in chunk_sizes:
        for ov in overlaps:
            if ov >= cs:
                continue

            # 1) chunk
            run([
                PY, str(ROOT / "src" / "chunker.py"),
                "--input", str(input_path),
                "--chunk_size", str(cs),
                "--overlap", str(ov),
                "--out", "outputs/chunks.jsonl"
            ])

            # 2) build tfidf
            run([
                PY, str(ROOT / "src" / "build_tfidf.py"),
                "--chunks", "outputs/chunks.jsonl",
                "--out_vec", "outputs/tfidf_vectorizer.joblib",
                "--out_mat", "outputs/tfidf_matrix.joblib",
                "--out_meta", "outputs/meta.jsonl"
            ])

            # 3) eval v2
            run([
                PY, str(ROOT / "src" / "eval_retrieval_v2.py"),
                "--eval", str(eval_path),
                "--topk", str(args.topk),
                "--rule", args.rule,
                "--out", "outputs/retrieval_report_v2.json"
            ])

            rep = read_report(ROOT / "outputs" / "retrieval_report_v2.json")
            rows.append({
                "chunk_size": cs,
                "overlap": ov,
                "topk": args.topk,
                "rule": args.rule,
                "hit_rate_answerable": rep["hit_rate_answerable"],
                "false_hit_rate_unanswerable": rep["false_hit_rate_unanswerable"],
                "answerable_total": rep["answerable_total"],
                "unanswerable_total": rep["unanswerable_total"]
            })

            print(f"[SWEEP] cs={cs} ov={ov} hit={rep['hit_rate_answerable']:.3f} falsehit={rep['false_hit_rate_unanswerable']:.3f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] saved {out_csv}")


if __name__ == "__main__":
    main()