import { json, listModels, PROVIDERS } from "@/lib/llm-server";

export const dynamic = "force-dynamic";

/** Which providers have a key configured, and the chat models that key can use (live). */
export async function GET() {
  const names = Object.keys(PROVIDERS);
  const models = Object.fromEntries(await Promise.all(names.map(async (p) => [p, await listModels(p)] as const)));
  return json({ providers: Object.fromEntries(names.map((p) => [p, !!process.env[PROVIDERS[p].key]])), models });
}
