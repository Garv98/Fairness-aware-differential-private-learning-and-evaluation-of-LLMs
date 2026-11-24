# FairDP dashboard (Next.js)

Front end of the FairDP demo. Talks to the FastAPI backend in `../server` (default `http://127.0.0.1:8000`, override with `NEXT_PUBLIC_API`).

| Tab | What it shows |
|---|---|
| Decision demo | Score a real applicant with all 5 models at ε = 1/3/8, counterfactual sex × race grid, per-field explanation, group recall |
| Overview | KPIs, pipeline, model comparison, worst-group recall vs ε, full metrics table |
| Fairness | Recall / FPR / selection rate by sex, race, age and intersection; cost of privacy per group; heatmap; gap summary |
| Privacy | Live Opacus accountant, multi-seed privacy–utility trade-off, membership-inference attack (overall + per group), noise plans |
| Train live | Starts a real training job on the backend; per-epoch curves and ε spent |
| LLM audit | Prompt from a real record; counterfactual identity audit against Groq / Gemini / OpenRouter (needs an API key) |
| Credit study | First workstream: DP-SGD on the credit-risk data |

```bash
npm install
npm run dev        # development, http://localhost:3000
npm run build && npm run start   # presentation
```
