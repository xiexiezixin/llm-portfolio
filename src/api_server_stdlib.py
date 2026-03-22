import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rag_service import RAGService, RAGConfig

HOST = "127.0.0.1"
PORT = 8000

# 启动时加载一次（很关键：避免每次请求重复加载索引）
SERVICE = RAGService(
    vec_path=Path("src/outputs/tfidf_vectorizer.joblib"),
    mat_path=Path("src/outputs/tfidf_matrix.joblib"),
    chunks_path=Path("src/outputs/chunks.jsonl"),
    threshold_path=Path("src/outputs/threshold.json"),
)


def _send_json(handler: BaseHTTPRequestHandler, status_code: int, obj):
    data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/health":
            _send_json(self, 200, {"status": "ok"})
        else:
            _send_json(self, 404, {"error": "not found", "path": path})

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/query":
            _send_json(self, 404, {"error": "not found", "path": path})
            return

        # 读取请求体
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            _send_json(self, 400, {"error": "invalid json"})
            return

        query = payload.get("query", "")
        topk = payload.get("topk", 5)

        if not isinstance(query, str) or not query.strip():
            _send_json(self, 400, {"error": "query must be non-empty string"})
            return
        try:
            topk = int(topk)
            if topk < 1 or topk > 20:
                raise ValueError
        except Exception:
            _send_json(self, 400, {"error": "topk must be int in [1, 20]"})
            return

        resp = SERVICE.answer(query.strip(), RAGConfig(topk=topk))
        _send_json(self, 200, resp)

    # 让日志更干净（可选）
    def log_message(self, format, *args):
        return


def main():
    httpd = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[OK] stdlib server running: http://{HOST}:{PORT}")
    print("  GET  /health")
    print("  POST /query  body: {\"query\":\"...\",\"topk\":5}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()