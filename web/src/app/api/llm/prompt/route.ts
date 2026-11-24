import { adultText, SYSTEM_INCOME } from "@/lib/llm";
import { fail, json, validate } from "@/lib/llm-server";

/** The exact prompt the audit sends (llm_experiments/serialize.py). */
export async function POST(req: Request) {
  const body = await req.json().catch(() => null);
  const errs = validate(body?.applicant);
  if (errs.length) return fail(422, errs.join("; "));
  return json({ system: SYSTEM_INCOME, user: adultText(body.applicant) });
}
