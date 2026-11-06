"use client";
import { Bell, ChevronDown } from "lucide-react";

export default function TopBar() {
  return (
    <header
      className="h-12 flex items-center justify-between px-4 border-b flex-shrink-0"
      style={{ background: "var(--bg-panel)", borderColor: "var(--border)" }}
    >
      {/* Left: squad selector */}
      <div className="flex items-center gap-3">
        <button
          className="flex items-center gap-2 text-sm px-3 py-1 rounded border"
          style={{ borderColor: "var(--border)", color: "var(--text-secondary)", background: "var(--bg-card)" }}
        >
          <span className="text-green-400">▲</span>
          Premier Squad · All Sports
          <ChevronDown size={12} />
        </button>

        <div
          className="demo-banner"
        >
          ● SIMULATED DATA DEMO · MOCK MODEL V2.4
        </div>
      </div>

      {/* Right: user */}
      <div className="flex items-center gap-4">
        <div className="relative">
          <Bell size={16} style={{ color: "var(--text-secondary)" }} />
          <span
            className="absolute -top-1 -right-1 w-3.5 h-3.5 rounded-full text-xs flex items-center justify-center font-bold"
            style={{ background: "var(--accent-red)", color: "#fff", fontSize: "9px" }}
          >
            3
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs"
            style={{ background: "linear-gradient(135deg,#3b82f6,#8b5cf6)", color: "#fff" }}
          >
            DV
          </div>
          <div className="text-xs">
            <div className="font-semibold" style={{ color: "var(--text-primary)" }}>Dr. Marcus Vance</div>
            <div style={{ color: "var(--text-muted)" }}>Head of Sports Science</div>
          </div>
        </div>
      </div>
    </header>
  );
}
