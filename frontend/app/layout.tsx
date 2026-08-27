import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/header";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "RetinaScreen AI — Explainable DR Screening",
  description:
    "An explainable, uncertainty-aware deep learning system for grading diabetic retinopathy severity from retinal fundus images.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-slate-950 text-slate-100 min-h-screen flex flex-col`}>
        <Header />
        <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <footer className="border-t border-slate-900/50 bg-slate-950 py-6 text-center text-xs text-slate-500/60">
          <p>
            © 2026 RetinaScreen AI · EfficientNet-B4 · Built for Clinical Screening Assistance
          </p>
        </footer>
      </body>
    </html>
  );
}
