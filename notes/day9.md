# Day9: Structured RAG response (JSON) + CLI demo

## Output schema
- mode: RAG_PROMPT | REFUSAL
- top1_score, threshold: evidence gate
- retrieved: topK chunks (chunk_id, score, source, preview)
- next_action: CALL_LLM_WITH_PROMPT | NEED_MORE_EVIDENCE
- prompt: either RAG prompt or refusal message

## Demo runs
- ATI: mode=RAG_PROMPT, top1_score=0.2046, thr=0.0572 -> src/outputs/demo_answer_ati.json
- Height: mode=REFUSAL, top1_score=0.0562, thr=0.0572 -> src/outputs/demo_answer_height.json
- PRF: mode=REFUSAL, top1_score=0.0397, thr=0.0572 -> src/outputs/demo_answer_prf.json