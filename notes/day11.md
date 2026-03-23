# Day11: Web demo (no deps)

## Run
- Backend: `python src/api_server_stdlib.py`
- Frontend: `python -m http.server 9000`
- Open: http://127.0.0.1:9000/web/demo.html

## CORS fix
- Added `Access-Control-Allow-*` headers + `OPTIONS` handler in stdlib server.

## Tests
- ATI query -> mode=RAG_PROMPT
- PRF query -> mode=REFUSAL

## Screenshots
- web/screenshots/rag_prompt.png
- web/screenshots/refusal.png