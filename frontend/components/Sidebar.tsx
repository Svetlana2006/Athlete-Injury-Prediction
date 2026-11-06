"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, Users, Activity, BarChart3, BrainCircuit
} from "lucide-react";

const NAV = [
  { href: "/",           label: "Overview",             icon: LayoutDashboard },
  { href: "/athletes",   label: "Athletes",             icon: Users           },
  { href: "/prediction", label: "Injury Prediction",    icon: Activity        },
  { href: "/analytics",  label: "Performance Analytics",icon: BarChart3       },
  { href: "/insights",   label: "Model Insights",       icon: BrainCircuit    },
];

export default function Sidebar() {
  const pathname = usePathname();
  return (
    <aside
      className="w-52 flex-shrink-0 flex flex-col border-r"
      style={{ background: "var(--bg-panel)", borderColor: "var(--border)" }}
    >
      {/* Logo */}
      <div className="px-4 py-4 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center font-black text-xs"
            style={{ background: "linear-gradient(135deg,#00e676,#3b82f6)", color: "#000" }}
          >
            AI
          </div>
          <div>
            <div className="font-black text-sm tracking-tight" style={{ color: "var(--text-primary)" }}>
              ATHLETEIQ
            </div>
            <div className="text-xs" style={{ color: "var(--text-muted)" }}>
              Precision Biometrics
            </div>
          </div>
        </div>
      </div>

      {/* System status */}
      <div className="px-4 py-3 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="stat-label mb-1">System Status</div>
        <div className="flex items-center gap-2 text-xs" style={{ color: "var(--accent-green)" }}>
          <div className="pulse-dot" /> Online
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 py-2">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-4 py-2.5 text-sm font-medium transition-all border-l-2 ${
                active
                  ? "nav-active"
                  : "border-l-transparent hover:bg-white/5"
              }`}
              style={{ color: active ? "var(--accent-green)" : "var(--text-secondary)" }}
            >
              <Icon size={15} />
              <span>{label}</span>
              {href === "/prediction" && (
                <span
                  className="ml-auto text-xs px-1.5 py-0.5 rounded font-bold"
                  style={{ background: "rgba(0,230,118,0.15)", color: "var(--accent-green)", fontSize: "9px" }}
                >
                  AI
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t" style={{ borderColor: "var(--border)" }}>
        <div className="stat-label mb-1">Inference Engine</div>
        <div className="text-xs font-mono" style={{ color: "var(--accent-cyan)" }}>Transformer-Focal-v1</div>
        <div className="mt-1 progress-bar">
          <div className="progress-fill" style={{ width: "92%", background: "var(--accent-green)" }} />
        </div>
        <div className="text-xs mt-1 flex justify-between" style={{ color: "var(--text-muted)" }}>
          <span>Telemetry Drift</span>
          <span className="font-mono">0.12%</span>
        </div>
      </div>
    </aside>
  );
}
