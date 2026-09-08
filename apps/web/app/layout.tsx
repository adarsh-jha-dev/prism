import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Prism — stack status",
  description: "Phase 0: proves the environment is wired up.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
