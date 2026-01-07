import { Link } from "react-router-dom";

export function LoginPage() {
  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h1 style={styles.h1}>Log in</h1>
        <p style={styles.p}>Placeholder page. We’ll wire auth later.</p>
        <Link to="/" style={styles.link}>
          ← Back to Home
        </Link>
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: "100vh",
    display: "grid",
    placeItems: "center",
    padding: 24,
    background: "linear-gradient(135deg, #f9fafb 0%, #eff6ff 45%, #faf5ff 100%)",
    fontFamily:
      'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji"',
  },
  card: {
    width: "min(520px, 100%)",
    background: "#ffffff",
    border: "1px solid rgba(15, 23, 42, 0.10)",
    borderRadius: 18,
    padding: 20,
  },
  h1: { margin: 0, fontSize: 22 },
  p: { margin: "10px 0 0", color: "#475569" },
  link: { display: "inline-block", marginTop: 14, color: "#2563eb", fontWeight: 700, textDecoration: "none" },
};


