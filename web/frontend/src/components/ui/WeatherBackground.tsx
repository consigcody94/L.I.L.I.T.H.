"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface WeatherBackgroundProps {
  condition: "clear" | "cloudy" | "rain" | "snow" | "storm";
}

export function WeatherBackground({ condition }: WeatherBackgroundProps) {
  const [particles, setParticles] = useState<{ id: number; x: number; delay: number }[]>([]);

  useEffect(() => {
    if (condition === "rain" || condition === "snow") {
      const newParticles = Array.from({ length: 50 }, (_, i) => ({
        id: i,
        x: Math.random() * 100,
        delay: Math.random() * 2,
      }));
      setParticles(newParticles);
    } else {
      setParticles([]);
    }
  }, [condition]);

  const gradients = {
    clear: "from-sky-900 via-blue-800 to-indigo-900",
    cloudy: "from-slate-700 via-slate-600 to-slate-800",
    rain: "from-slate-800 via-slate-700 to-blue-900",
    snow: "from-slate-600 via-blue-200 to-slate-500",
    storm: "from-slate-900 via-purple-900 to-slate-800",
  };

  return (
    <div className="fixed inset-0 overflow-hidden -z-10">
      {/* Gradient background */}
      <div
        className={`absolute inset-0 bg-gradient-to-b ${gradients[condition]} transition-all duration-1000`}
      />

      {/* Animated gradient overlay */}
      <motion.div
        className="absolute inset-0 opacity-30"
        animate={{
          background: [
            "radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%)",
            "radial-gradient(circle at 80% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%)",
            "radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%)",
          ],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />

      {/* Rain particles */}
      {condition === "rain" &&
        particles.map((particle) => (
          <motion.div
            key={particle.id}
            className="absolute w-0.5 h-6 bg-gradient-to-b from-transparent to-blue-400/60"
            style={{ left: `${particle.x}%` }}
            initial={{ y: "-10%" }}
            animate={{ y: "110vh" }}
            transition={{
              duration: 1,
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
            className="absolute w-2 h-2 rounded-full bg-white/80"
            style={{ left: `${particle.x}%` }}
            initial={{ y: "-10%", rotate: 0 }}
            animate={{ y: "110vh", rotate: 360 }}
            transition={{
              duration: 5 + Math.random() * 3,
              repeat: Infinity,
              delay: particle.delay,
              ease: "linear",
            }}
          />
        ))}

      {/* Noise texture overlay */}
      <div
        className="absolute inset-0 opacity-[0.03]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`,
        }}
      />
    </div>
  );
}
