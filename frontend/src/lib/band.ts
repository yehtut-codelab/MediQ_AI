export type Band = "green" | "amber" | "red";

// Matches backend/app/api/routes.py SGT_BANDS thresholds exactly.
export function bandFor(waitMin: number): Band {
  if (waitMin < 30) return "green";
  if (waitMin < 60) return "amber";
  return "red";
}

// Text label paired with every band color so severity never depends on hue
// alone (colorblind-safe: WCAG 1.4.1 "use of color").
export const BAND_LABEL: Record<Band, string> = {
  green: "On track",
  amber: "Busy",
  red: "Delayed",
};
