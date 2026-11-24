// Presentation-only config: colour per model id (colour follows the entity)
// and human labels for form fields. All data values come from the API.
const PALETTE = ["#2a78d6", "#e87ba4", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#008300"];
const FIXED: Record<string, string> = { erm: "#2a78d6", groupdro: "#e87ba4", dpsgd: "#eb6834", dpsgd_balanced: "#1baf7a", ig_dpsgd: "#6250d6" };
const assigned = new Map<string, string>(Object.entries(FIXED));
export function color(id: string): string {
  if (!assigned.has(id)) assigned.set(id, PALETTE[assigned.size % PALETTE.length]);
  return assigned.get(id)!;
}

export const FIELD_LABEL: Record<string, string> = {
  age: "Age", sex: "Sex", race: "Race", workclass: "Work class", education: "Education", "marital-status": "Marital status",
  occupation: "Occupation", relationship: "Relationship", "native-country": "Country", "hours-per-week": "Hours / week",
  "capital-gain": "Capital gain ($)", "capital-loss": "Capital loss ($)",
};
export const label = (f: string) => FIELD_LABEL[f] ?? f;
export const pct = (v: number | null | undefined, d = 1) => (v == null || !Number.isFinite(v) ? "–" : `${(v * 100).toFixed(d)}%`);
export const f3 = (v: number | null | undefined, d = 3) => (v == null || !Number.isFinite(v) ? "–" : v.toFixed(d));
export const cellLabel = (c: string) => c.replaceAll("|", " · ").replace("<=", "≤");
export const epsLabel = (e: number | null) => (e == null ? "∞" : String(e));
