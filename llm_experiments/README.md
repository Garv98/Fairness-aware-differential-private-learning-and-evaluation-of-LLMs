# LLM workstream (methodology step 5)

Everything goes through `clients.LLM`, which caches every response in `cache/`. Reruns
and analysis therefore never use up API quota. API keys are read from environment
variables (`GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`); never commit them.

## Models (free-tier feasible)
| Where | Models |
|---|---|
| Colab GPU, 4-bit (`--provider hf`) | Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Gemma-2-9B-it, Mistral-7B-Instruct |
| Free APIs | Gemini Flash (`gemini`), Groq-hosted Llama 70B (`groq`), OpenRouter free models |

Check the current free model IDs before running, because providers rotate them.

## Experiments
| ID | Status | What it measures | Entry point |
|---|---|---|---|
| L1 | ready | LLM-as-classifier on the same test rows as the DP models, scored with the same group and intersectional metrics | `run_l1_l2.py` |
| L2 | ready | Counterfactual consistency: flip rate and probability shift when an explicit race × sex cue is added | `run_l1_l2.py` |
| L3 | todo | Multilingual: English, Hindi, Spanish, plus one low-resource language. Translate prompt templates (keep the JSON keys in English) and have each translation checked by a native speaker. Measure cross-language decision agreement and language × identity interaction | add `run_l3.py` |
| L4 | todo | BBQ subset and Anthropic DiscrimEval (HF datasets) | add `run_l4.py` |
| L5 | todo | Privacy leakage. (a) Contextual: put k records in context, ask adversarial questions about another person, measure PII leakage rate (ConfAIde-style). (b) Min-K%++ membership inference on WikiMIA (open models only, needs logits) | add `run_l5.py` |
| L6 | stretch | DP-LoRA vs LoRA fine-tune of Qwen2.5-0.5B/1.5B on serialized credit rows with injected canaries. Measure canary exposure and per-group utility, then try IG reweighting | add `run_l6.py` |

Protocol: use temperature 0 for the main results, plus 5 samples at T = 0.7 to measure
stability. Report the parse-failure / refusal rate. Use the same seed as the ML runs so
the rows match.

```bash
python llm_experiments/run_l1_l2.py --dataset credit --provider groq --model <model-id> --n 1000
```
Outputs go to `results/llm/<dataset>__<model>/`. The dashboard picks them up automatically.
