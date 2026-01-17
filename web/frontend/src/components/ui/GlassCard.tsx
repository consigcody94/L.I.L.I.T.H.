"use client";

import { ReactNode } from "react";
import { cn } from "@/lib/utils";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  variant?: "default" | "dark" | "light";
  hover?: boolean;
}

export function GlassCard({
  children,
  className,
  variant = "default",
  hover = false,
}: GlassCardProps) {
  const variants = {
    default: "bg-white/10 border-white/20",
    dark: "bg-black/20 border-white/10",
    light: "bg-white/20 border-white/30",
  };

  return (
    <div
      className={cn(
        "backdrop-blur-xl border rounded-2xl p-6 shadow-xl transition-all duration-300",
        variants[variant],
        hover && "hover:bg-white/15 hover:shadow-2xl hover:scale-[1.02]",
        className
      )}
    >
      {children}
    </div>
  );
}
