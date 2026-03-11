# Day7: Stricter Retrieval Eval + Negative Set + Sweep

## Setup
- topk = 5
- eval set: 20 answerable + 5 unanswerable
- retrieval: TF-IDF (char 2-4gram)

## Rule comparison
- rule=any:
  - Answerable Hit@5 = 1.000 (20/20)
  - Unanswerable FalseHit@5 = 0.200 (1/5)
  - Risky case: qid=103 “平台飞行高度是多少米？” triggered by generic keyword “高度”
- rule=all:
  - Answerable Hit@5 = 1.000 (20/20)
  - Unanswerable FalseHit@5 = 0.000 (0/5)

## Sweep results (rule=all)
- chunk_size in {200, 400, 800}, overlap in {0, 50, 100}
- All settings: Hit@5=1.000, FalseHit@5=0.000
- Observation: current doc/questions are small & wording-aligned, so chunk params are not sensitive.

## Conclusion
- For “answerability / evidence-sufficiency” judgement, rule=all is much safer than rule=any on negative queries.
- Next improvement (still within retrieval): refine negative keywords to avoid generic tokens; optionally add score threshold gating.