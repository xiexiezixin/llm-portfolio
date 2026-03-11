import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

#功能：读 txt/md → 按字符长度切分 → overlap 重叠 → 输出 JSONL（每行一个 chunk，含 id、来源、起止位置、文本）

def read_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    # 统一换行/空白（最小清洗）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict]:
    """
    简单字符级切分：chunk_size 每块长度；相邻块重叠 overlap 字符。
    """
    assert chunk_size > 0
    assert 0 <= overlap < chunk_size

    chunks = []
    start = 0
    n = len(text)
    idx = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_id": idx,
                "start": start,
                "end": end,
                "text": chunk
            })
            idx += 1

        if end >= n:
            break
        start = end - overlap  # 重叠回退

    return chunks


def save_jsonl(chunks: List[Dict], out_path: Path, source: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "source": source,
                **c
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Simple text chunker for txt/md files.")
    parser.add_argument("--input", type=str, default="../src/data/sample.txt", help="Path to input .txt/.md")
    parser.add_argument("--chunk_size", type=int, default=200, help="Chars per chunk")
    parser.add_argument("--overlap", type=int, default=100, help="Overlap chars between chunks")
    parser.add_argument("--out", type=str, default="../src/outputs/chunks.jsonl", help="Output jsonl path")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    text = read_text(in_path)
    chunks = chunk_text(text, args.chunk_size, args.overlap)

    save_jsonl(chunks, Path(args.out), source=str(in_path))
    print(f"[OK] input={in_path} chars={len(text)} chunks={len(chunks)} out={args.out}")

    # 打印前2块预览
    for c in chunks[:2]:
        preview = c["text"][:120].replace("\n", " ")
        print(f"chunk#{c['chunk_id']} [{c['start']}:{c['end']}] {preview}...")


if __name__ == "__main__":
    main()