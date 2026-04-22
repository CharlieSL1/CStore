import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CStore — Text-to-Csound Console",
  description:
    "Local console for the CStore V1.0.1 Csound generator. Generates interpretable, editable .csd source and renders audio via Csound.",
  icons: [{ rel: "icon", url: "/favicon.svg" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        {/* Typography: a serif display + neo-grotesque sans + technical mono.
            Loaded from Google Fonts with preconnect for a fast first paint. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin=""
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
