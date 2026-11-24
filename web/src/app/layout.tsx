import type { Metadata } from "next";
// Self-hosted fonts (bundled at build time; no request to Google Fonts)
import "@fontsource-variable/bricolage-grotesque";
import "@fontsource/ibm-plex-sans/latin-400.css";
import "@fontsource/ibm-plex-sans/latin-500.css";
import "@fontsource/ibm-plex-sans/latin-600.css";
import "@fontsource/ibm-plex-mono/latin-400.css";
import "@fontsource/ibm-plex-mono/latin-500.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "FairDP Audit Console",
  description: "Fairness-aware differentially private learning: live decisions, fairness, privacy and responsible-AI audit.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
