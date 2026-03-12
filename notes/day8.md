# Day8: Evidence Gate (Threshold) for RAG

## Calibration
- Method: threshold = max(top1_score on unanswerable) + margin
- recommended threshold = 0.0572
- pos median = 0.1788, neg max = 0.0562

## Cases
1) Answerable query
- Q: 顺轨干涉ATI可反演什么物理量？
- mode=RAG_PROMPT, top1_score=0.2046, threshold=0.0572

2) Unanswerable query
- Q: 平台飞行高度是多少米？
- mode=REFUSAL, top1_score=0.0562, threshold=0.0572

3) Unanswerable query
- Q: 本文PRF是多少Hz？
- mode=REFUSAL, top1_score=0.0397, threshold=0.0572

## Takeaway
- A simple top1-score gate can effectively reduce hallucination risk on out-of-scope numeric queries while preserving answerable retrieval.