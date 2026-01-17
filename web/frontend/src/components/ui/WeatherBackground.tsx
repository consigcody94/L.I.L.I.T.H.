"use client";

import { useEffect, useState, useMemo } from "react";
import { motion } from "framer-motion";

interface WeatherBackgroundProps {
  condition: "clear" | "cloudy" | "rain" | "snow" | "storm";
}

// Floating orbs for ambient effect
const FloatingOrb = ({ delay, duration, size, x, y, color }: {
  delay: number;
  duration: number;
  size: number;
  x: string;
  y: string;
  color: string;
}) => (
  <motion.div
    className="absolute rounded-full blur-3xl"
    style={{
      width: size,
      height: size,
      left: x,
      top: y,
      background: color,
    }}
    animate={{
      y: [0, -30, 0, 30, 0],
      x: [0, 20, 0, -20, 0],
      scale: [1, 1.1, 1, 0.9, 1],
      opacity: [0.3, 0.5, 0.3, 0.4, 0.3],
    }}
    transition={{
      duration,
      repeat: Infinity,
      delay,
      ease: "easeInOut",
    }}
  />
);

export function WeatherBackground({ condition }: WeatherBackgroundProps) {
  const [particles, setParticles] = useState<{ id: number; x: number; delay: number; size?: number }[]>([]);

  // Generate floating orbs data
  const orbs = useMemo(() => [
    { delay: 0, duration: 20, size: 400, x: "10%", y: "20%", color: "rgba(139, 92, 246, 0.15)" },
    { delay: 5, duration: 25, size: 350, x: "70%", y: "60%", color: "rgba(6, 182, 212, 0.12)" },
    { delay: 10, duration: 22, size: 300, x: "50%", y: "80%", color: "rgba(59, 130, 246, 0.1)" },
    { delay: 3, duration: 18, size: 250, x: "80%", y: "10%", color: "rgba(168, 85, 247, 0.12)" },
    { delay: 8, duration: 28, size: 200, x: "20%", y: "70%", color: "rgba(14, 165, 233, 0.1)" },
  ], []);

  useEffect(() => {
    if (condition === "rain" || condition === "snow") {
      const count = condition === "rain" ? 80 : 50;
      const newParticles = Array.from({ length: count }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        delay: Math.random() * 2,
        size: condition === "snow" ? Math.random() * 4 + 2 : undefined,
      }));
      setParticles(newParticles);
    } else {
      setParticles([]);
    }
  }, [condition]);

  return (
    <div className="fixed inset-0 overflow-hidden -z-10">
      {/* Neural network background image */}
      <div
        className="absolute inset-0 bg-cover bg-center bg-no-repeat scale-105"
        style={{
          backgroundImage: `url("/images/background.png")`,
        }}
      />

      {/* Dark overlay for better text readability - enhanced gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950/85 via-purple-950/50 to-slate-950/90" />

      {/* Mesh gradient overlay for depth */}
      <div
        className="absolute inset-0 opacity-70"
        style={{
          background: `
            radial-gradient(ellipse 80% 50% at 20% 40%, rgba(120, 0, 255, 0.12), transparent),
            radial-gradient(ellipse 60% 40% at 80% 60%, rgba(0, 100, 255, 0.1), transparent),
            radial-gradient(ellipse 50% 30% at 50% 20%, rgba(180, 0, 200, 0.08), transparent)
          `
        }}
      />

      {/* Floating orbs for depth and movement */}
      {orbs.map((orb, i) => (
        <FloatingOrb key={i} {...orb} />
      ))}

      {/* Animated glow effect that moves across the background */}
      <motion.div
        className="absolute inset-0 opacity-30"
        animate={{
          background: [
            "radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.25) 0%, transparent 50%)",
            "radial-gradient(circle at 80% 70%, rgba(59, 130, 246, 0.25) 0%, transparent 50%)",
            "radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.25) 0%, transparent 50%)",
            "radial-gradient(circle at 20% 30%, rgba(59, 130, 246, 0.25) 0%, transparent 50%)",
          ],
        }}
        transition={{
          duration: 20,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Secondary pulsing glow */}
      <motion.div
        className="absolute inset-0"
        animate={{
          opacity: [0.08, 0.15, 0.08],
        }}
        transition={{
          duration: 5,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{
          background: "radial-gradient(circle at 50% 50%, rgba(6, 182, 212, 0.2) 0%, transparent 70%)",
        }}
      />

      {/* Top ambient glow */}
      <motion.div
        className="absolute top-0 left-0 right-0 h-96"
        animate={{
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut",
        }}
        style={{
          background: "linear-gradient(to bottom, rgba(139, 92, 246, 0.1) 0%, transparent 100%)",
        }}
      />

      {/* Rain particles */}
      {condition === "rain" &&
        particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-[1px] h-8 bg-gradient-to-b from-transparent via-blue-400/40 to-blue-400/70"
            style={{ left: `${particle.x}%` }}
            initial={{ y: "-10%" }}
            animate={{ y: "110vh" }}
            transition={{
              duration: 0.8 + Math.random() * 0.4,
              repeat: Infinity,
              delay: particle.delay,
              ease: "linear",
            }}
          />
        ))}

      {/* Snow particles */}
      {condition === "snow" &&
        particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute rounded-full bg-white/80"
            style={{
              left: `${particle.x}%`,
              width: particle.size || 4,
              height: particle.size || 4,
            }}
            initial={{ y: "-10%", rotate: 0, x: 0 }}
            animate={{ y: "110vh", rotate: 360, x: [0, 20, -20, 10, 0] }}
            transition={{
              duration: 6 + Math.random() * 4,
              repeat: Infinity,
              delay: particle.delay,
              ease: "linear",
            }}
          />
        ))}

      {/* Noise texture overlay for subtle grain effect */}
      <div
        className="absolute inset-0 opacity-[0.015] mix-blend-overlay pointer-events-none"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />

      {/* Grid overlay for subtle tech effect */}
      <div
        className="absolute inset-0 opacity-[0.02] pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
        }}
      />

      {/* Vignette effect for depth */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: "radial-gradient(ellipse at center, transparent 0%, rgba(0,0,0,0.5) 100%)",
        }}
      />

      {/* Bottom fade for content separation */}
      <div
        className="absolute bottom-0 left-0 right-0 h-32 pointer-events-none"
        style={{
          background: "linear-gradient(to top, rgba(0,0,0,0.3) 0%, transparent 100%)",
        }}
      />
    </div>
  );
}
