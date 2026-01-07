// central theme reference (keep it simple and use everywhere).
// goal: one place for brand colors + gradients so the UI stays consistent.

export const theme = {
  name: "sentiment",

  colors: {
    // brand
    primary: "#7c3aed", // purple
    secondary: "#f97316", // orange

    // neutrals
    ink: "#0f172a",
    slate: "#334155",
    muted: "#475569",
    border: "rgba(15, 23, 42, 0.10)",
    glass: "rgba(255, 255, 255, 0.8)",
    white: "#ffffff",
  },

  gradients: {
    // main background + highlights
    page: "linear-gradient(135deg, #f9fafb 0%, #eff6ff 45%, #faf5ff 100%)",
    brand: "linear-gradient(135deg, #7c3aed 0%, #f97316 100%)", // purple → orange
    cta: "linear-gradient(135deg, #7c3aed 0%, #2563eb 45%, #f97316 100%)",
  },

  radius: {
    sm: 10,
    md: 14,
    lg: 18,
    xl: 22,
    pill: 999,
  },

  shadow: {
    soft: "0 12px 24px rgba(15, 23, 42, 0.06)",
    primary: "0 10px 30px rgba(124, 58, 237, 0.22)",
  },
};


