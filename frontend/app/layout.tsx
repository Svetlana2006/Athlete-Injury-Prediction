import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "ATHLETEIQ – Precision Injury Risk Platform",
  description: "AI-powered athlete injury risk prediction using Transformer neural networks.",
};

// Minimal layout — the Stitch HTML pages are fully self-contained
// (they include their own sidebar, header, fonts, Tailwind CDN)
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
