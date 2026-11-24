import { adultText, parseAnswer, reasoningParams, SYSTEM_INCOME } from "@/lib/llm";
import { fail, json, PROVIDERS, schema, UA, validate, validModel } from "@/lib/llm-server";

export const maxDuration = 60;

async function chat(base: string, key: string, model: string, user: string): Promise<string> {
  const body = JSON.stringify({ model, temperature: 0, max_tokens: 1024, ...reasoningParams(model),
    messages: [{ role: "system", content: SYSTEM_INCOME }, { role: "user", content: user }] });
  for (let attempt = 0; attempt < 5; attempt++) {
    const r = await fetch(`${base}/chat/completions`, { method: "POST", body,
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}`, "User-Agent": UA } });
    if (r.ok) return (await r.json()).choices?.[0]?.message?.content ?? "";
    if (![429, 500, 502, 503, 504].includes(r.status)) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
    await new Promise((res) => setTimeout(res, Math.min(8000, 1000 * 2 ** attempt)));
  }
  throw new Error("rate limited after retries");
}

/** Ask the model about the same person under every sex x race identity (counterfactual audit). */
export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  const provider = body?.provider, model = body?.model, a = body?.applicant;
  if (!PROVIDERS[provider]) return fail(400, `provider must be one of ${JSON.stringify(Object.keys(PROVIDERS))}`);
  if (!validModel(model)) return fail(400, "invalid model id");
  const errs = validate(a);
  if (errs.length) return fail(422, errs.join("; "));
  const key = process.env[PROVIDERS[provider].key];
  if (!key) return fail(400, `Set ${PROVIDERS[provider].key} in the deployment environment.`);
  const combos = schema.categorical.sex.flatMap((sex) => schema.categorical.race.map((race) => ({ sex, race })));
  try {
    const results = [];
    for (let i = 0; i < combos.length; i += 5) {           // 5 parallel requests stays inside free-tier rate limits
      results.push(...await Promise.all(combos.slice(i, i + 5).map(async ({ sex, race }) => {
        const resp = await chat(PROVIDERS[provider].base, key, model, adultText({ ...a, sex, race }));
        const [decision, prob] = parseAnswer(resp);
        return { sex, race, decision, prob, raw: resp.slice(0, 300) };
      })));
    }
    return json({ model, provider, results, prompt: adultText(a), system: SYSTEM_INCOME });
  } catch (e) {
    return fail(502, `LLM call failed: ${(e as Error).message}`);
  }
}
