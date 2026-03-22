# Day10: stdlib API server (no extra deps)

## Start server
python src/api_server_stdlib.py

## Test
- GET http://127.0.0.1:8000/health -> {"status":"ok"}
- POST /query (ATI) saved: src/outputs/api_stdlib_test_ati.json  (mode=RAG_PROMPT)
- POST /query (PRF) saved: src/outputs/api_stdlib_test_prf.json  (mode=REFUSAL)