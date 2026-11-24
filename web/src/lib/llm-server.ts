// Server-only helpers for the LLM audit route handlers (Vercel functions).
// The provider key is read from the environment and never sent to the browser.
import bundle from "../../public/static/bundle.json";
import { MODEL_ID, NON_CHAT } from "./llm";

export const PROVIDERS: Record<string, { key: string; base: string }> = {
  groq: { key: "GROQ_API_KEY", base: "https://api.groq.com/openai/v1" },
};
export const UA = "fairdp-research/1.0";   // Cloudflare-fronted providers reject default agents

type Schema = { fields: string[]; numeric: Record<string, { min: number; max: number }>; categorical: Record<string, string[]> };
export const schema = (bundle as unknown as { meta: { schema: Schema } }).meta.schema;

export function validate(a: unknown): string[] {
  if (!a || typeof a !== "object") return ["applicant must be an object"];
  const r = a as Record<string, unknown>, errs: string[] = [];
  for (const f of schema.fields) {
    if (!(f in r)) { errs.push(`missing field '${f}'`); continue; }
    if (f in schema.numeric) {
      const x = Number(r[f]);
      const { min, max } = schema.numeric[f];
      if (r[f] === "" || r[f] == null || !Number.isFinite(x) || x < min || x > max) errs.push(`'${f}' must be between ${min} and ${max}`);
    } else if (!schema.categorical[f].includes(String(r[f]))) errs.push(`'${f}' has unknown value '${r[f]}'`);
  }
  return errs;
}

export const json = (data: unknown, status = 200) => Response.json(data, { status });
export const fail = (status: number, detail: string) => json({ detail }, status);
export const validModel = (m: unknown): m is string => typeof m === "string" && MODEL_ID.test(m) && !m.includes("..");

export async function listModels(provider: string): Promise<string[]> {
  const p = PROVIDERS[provider], key = p && process.env[p.key];
  if (!key) return [];
  try {
    const r = await fetch(`${p.base}/models`, { headers: { Authorization: `Bearer ${key}`, "User-Agent": UA }, next: { revalidate: 3600 } });
    if (!r.ok) return [];
    const ids = ((await r.json()).data as { id: string }[]).map((m) => m.id);
    return ids.filter((i) => !NON_CHAT.some((x) => i.includes(x))).sort();
  } catch { return []; }
}
