"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  variant?: "default" | "dark" | "light" | "accent" | "gradient";
  hover?: boolean;
  glow?: boolean;
  animated?: boolean;
}

export function GlassCard({
  children,
  className,
  variant = "default",
  hover = false,
  glow = false,
  animated = false,
}: GlassCardProps) {
  const variants = {
    default: "bg-white/[0.06] border-white/[0.1] shadow-[0_8px_32px_rgba(0,0,0,0.4)]",
    dark: "bg-black/40 border-white/[0.08] shadow-[0_8px_32px_rgba(0,0,0,0.5)]",
    light: "bg-white/15 border-white/25 shadow-[0_8px_32px_rgba(255,255,255,0.1)]",
    accent: "bg-purple-900/30 border-purple-500/20 shadow-[0_8px_32px_rgba(139,92,246,0.2)]",
    gradient: "bg-gradient-to-br from-white/[0.08] to-white/[0.02] border-white/[0.1] shadow-[0_8px_32px_rgba(0,0,0,0.4)]",
  };

  return (
    <div
      className={cn(
        "group/card relative backdrop-blur-xl border rounded-2xl p-6 transition-all duration-500 overflow-hidden",
        variants[variant],
        hover && "hover:bg-white/[0.1] hover:shadow-[0_20px_60px_rgba(0,0,0,0.5)] hover:scale-[1.02] hover:border-white/20 cursor-pointer",
        glow && "before:absolute before:inset-0 before:rounded-2xl before:bg-gradient-to-r before:from-purple-500/20 before:via-cyan-500/10 before:to-blue-500/20 before:blur-2xl before:-z-10 before:opacity-60",
        animated && "before:animate-pulse",
        className
      )}
    >
      {/* Animated gradient border */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover/card:opacity-100 transition-opacity duration-500 pointer-events-none">
        <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/20 via-cyan-500/20 to-purple-500/20 blur-sm" />
      </div>

      {/* Inner highlight */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/[0.1] via-transparent to-transparent pointer-events-none" />

      {/* Subtle noise texture */}
      <div className="absolute inset-0 rounded-2xl opacity-[0.02] pointer-events-none" style={{ backgroundImage: 'url("data:image/svg+xml,%3Csvg viewBox=\'0 0 256 256\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cfilter id=\'noise\'%3E%3CfeTurbulence type=\'fractalNoise\' baseFrequency=\'0.9\' numOctaves=\'4\' stitchTiles=\'stitch\'/%3E%3C/filter%3E%3Crect width=\'100%25\' height=\'100%25\' filter=\'url(%23noise)\'/%3E%3C/svg%3E")' }} />

      {/* Content */}
      <div className="relative z-10">{children}</div>
    </div>
  );
}
