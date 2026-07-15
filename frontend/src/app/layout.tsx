import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "MediQ AI — Queue & Wait Advisor",
  description: "RAG-based wait time estimates from historical visit events",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <nav className="mediq-nav">
          <div>
            <div className="logo-name">MediQ AI</div>
            <div className="logo-sub">AI-Driven Digital Twin for Eye Clinic</div>
          </div>
          <div className="nav-links">
            <a href="/">Arrival Simulator</a>
            <a href="/register">New Patient Entry</a>
            <a href="/stations">Station Map</a>
            <a href="/agent">AI Agent Console</a>
            <a href="/qa">SOP Q&amp;A</a>
            <a href="/documents">Knowledge Base</a>
          </div>
        </nav>
        <main className="page">{children}</main>
      </body>
    </html>
  );
}
