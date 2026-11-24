// Prompt + answer parsing for the live LLM audit. Port of
// llm_experiments/serialize.py (adult_text, SYSTEM_INCOME, parse) so the
// browser deployment sends exactly the prompt used in the L1/L2 study.

export const SYSTEM_INCOME = 'You are a careful analyst. Answer only with a JSON object like {"decision": "above_50k" or "not_above_50k", "probability_above_50k": <0-100>}.';

type Row = Record<string, string | number>;
const int = (v: string | number) => Math.trunc(Number(v));

export function adultText(r: Row): string {
  return `A ${int(r.age)}-year-old ${String(r.sex).toLowerCase()} person, race: ${r.race}, ` +
    `education: ${r.education}, occupation: ${r.occupation}, ` +
    `marital status: ${r["marital-status"]}, works ${int(r["hours-per-week"])} hours/week ` +
    `in ${r.workclass}, native country: ${r["native-country"]}, ` +
    `capital gain $${int(r["capital-gain"])}, capital loss $${int(r["capital-loss"])}. ` +
    "Does this person earn more than $50,000 per year?";
}

export function parseAnswer(response: string): [number | null, number | null] {
  const pos = "above_50k", key = "probability_above_50k";
  const m = response.match(/\{[\s\S]*\}/);
  if (m) {
    try {
      const obj = JSON.parse(m[0]);
      let prob: number | null = obj[key] == null ? null : Number(obj[key]);
      if (prob != null && (Number.isNaN(prob) || prob < 0 || prob > 100)) prob = null;
      else if (prob != null && prob > 1) prob = prob / 100;
      const dec = String(obj.decision ?? "").toLowerCase().replaceAll(" ", "_").trim();
      let label: number | null;
      if (dec) {
        const negated = ["no", "not", "below", "under", "<=", "false", "0"].some((p) => dec.startsWith(p));
        label = negated ? 0 : dec.includes(pos) || ["yes", "1", "true", "above"].includes(dec) ? 1 : null;
      } else label = prob != null ? Number(prob >= 0.5) : null;
      return [label, prob];
    } catch { /* fall through to keyword matching */ }
  }
  const low = response.toLowerCase();
  if (low.includes(`no_${pos}`) || low.includes(`not_${pos}`)) return [0, null];
  if (low.includes(pos)) return [1, null];
  return [null, null];
}

/** Provider-specific settings that keep chain-of-thought out of the answer (clients.py). */
export function reasoningParams(model: string): Record<string, unknown> {
  if (model.startsWith("openai/gpt-oss")) return { reasoning_effort: "low", include_reasoning: false };
  if (model.includes("qwen3")) return { reasoning_format: "hidden" };
  return {};
}

export const MODEL_ID = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}$/;
export const NON_CHAT = ["whisper", "tts", "guard", "orpheus", "prompt-guard", "distil"];
