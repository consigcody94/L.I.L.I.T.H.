"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";

interface TemperatureDisplayProps {
  high: number;
  low: number;
  precipitation: number;
  precipitationProbability: number;
  unit?: "C" | "F";
}

// Convert Celsius to Fahrenheit
function toFahrenheit(celsius: number): number {
  return celsius * 9 / 5 + 32;
}

// Convert temperature based on unit
function convertTemp(celsius: number, unit: "C" | "F"): number {
  return unit === "F" ? toFahrenheit(celsius) : celsius;
}

function getTempStyles(temp: number, unit: "C" | "F"): { text: string; glow: string; shadow: string } {
  // Normalize to Celsius for color calculation
  const tempC = unit === "F" ? (temp - 32) * 5 / 9 : temp;

  if (tempC < 0) return { text: "text-blue-400", glow: "from-blue-400", shadow: "shadow-blue-500/40" };
  if (tempC < 10) return { text: "text-cyan-400", glow: "from-cyan-400", shadow: "shadow-cyan-500/40" };
  if (tempC < 15) return { text: "text-teal-400", glow: "from-teal-400", shadow: "shadow-teal-500/40" };
  if (tempC < 20) return { text: "text-emerald-400", glow: "from-emerald-400", shadow: "shadow-emerald-500/40" };
  if (tempC < 25) return { text: "text-amber-400", glow: "from-amber-400", shadow: "shadow-amber-500/40" };
  if (tempC < 30) return { text: "text-orange-400", glow: "from-orange-400", shadow: "shadow-orange-500/40" };
  return { text: "text-red-400", glow: "from-red-400", shadow: "shadow-red-500/40" };
}

export function TemperatureDisplay({
  high,
  low,
  precipitation,
  precipitationProbability,
  unit = "C",
}: TemperatureDisplayProps) {
  const displayHigh = useMemo(() => convertTemp(high, unit), [high, unit]);
  const displayLow = useMemo(() => convertTemp(low, unit), [low, unit]);

  const highStyles = useMemo(() => getTempStyles(displayHigh, unit), [displayHigh, unit]);
  const lowStyles = useMemo(() => getTempStyles(displayLow, unit), [displayLow, unit]);

  // Calculate bar position based on original Celsius values
  // Scale: -30°C (0%) to 45°C (100%) for a 75° range
  const tempToPosition = (tempC: number) => {
    const minTemp = -30;
    const maxTemp = 45;
    return ((tempC - minTemp) / (maxTemp - minTemp)) * 100;
  };

  const lowBarPos = tempToPosition(low);
  const highBarPos = tempToPosition(high);

  return (
    <div className="space-y-6">
      {/* Main temperature display */}
      <div className="flex items-center justify-center gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="text-center relative"
        >
          <p className="text-xs font-medium text-white/40 uppercase tracking-wider mb-2">High</p>
          <div className="relative">
            <p className={`text-6xl font-black ${highStyles.text} drop-shadow-lg`}>
              {Math.round(displayHigh)}°
            </p>
            {/* Glow effect */}
            <div className={`absolute inset-0 blur-2xl opacity-30 bg-gradient-to-t ${highStyles.glow} to-transparent -z-10`} />
          </div>
        </motion.div>

        {/* Divider with gradient */}
        <div className="relative h-20 w-px">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent via-white/30 to-transparent" />
        </div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
          className="text-center relative"
        >
          <p className="text-xs font-medium text-white/40 uppercase tracking-wider mb-2">Low</p>
          <div className="relative">
            <p className={`text-6xl font-black ${lowStyles.text} drop-shadow-lg`}>
              {Math.round(displayLow)}°
            </p>
            {/* Glow effect */}
            <div className={`absolute inset-0 blur-2xl opacity-30 bg-gradient-to-t ${lowStyles.glow} to-transparent -z-10`} />
          </div>
        </motion.div>
      </div>

      {/* Temperature bar */}
      <div className="relative h-8 flex items-center">
        <div className="relative w-full h-3 bg-white/[0.08] rounded-full overflow-hidden backdrop-blur-sm border border-white/[0.05]">
          <motion.div
            className="absolute h-full bg-gradient-to-r from-blue-500 via-emerald-500 via-amber-500 to-red-500"
            initial={{ width: 0 }}
            animate={{ width: "100%" }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          />
          {/* Glass overlay */}
          <div className="absolute inset-0 bg-gradient-to-b from-white/20 to-transparent" />
        </div>

        {/* Low marker */}
        <motion.div
          className="absolute"
          style={{
            left: `${Math.max(4, Math.min(96, lowBarPos))}%`,
            top: '50%',
            transform: 'translate(-50%, -50%)'
          }}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.5, type: "spring" }}
        >
          <div className="w-7 h-7 rounded-full bg-blue-500 border-2 border-white shadow-lg shadow-blue-500/50 flex items-center justify-center">
            <span className="text-[10px] font-bold text-white">L</span>
          </div>
        </motion.div>

        {/* High marker */}
        <motion.div
          className="absolute"
          style={{
            left: `${Math.max(4, Math.min(96, highBarPos))}%`,
            top: '50%',
            transform: 'translate(-50%, -50%)'
          }}
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.6, type: "spring" }}
        >
          <div className="w-7 h-7 rounded-full bg-red-500 border-2 border-white shadow-lg shadow-red-500/50 flex items-center justify-center">
            <span className="text-[10px] font-bold text-white">H</span>
          </div>
        </motion.div>
      </div>

      {/* Precipitation info */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="flex items-center justify-between pt-4 border-t border-white/[0.08]"
      >
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-blue-500/20 flex items-center justify-center">
              <svg className="w-5 h-5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M5.5 17a4.5 4.5 0 01-1.44-8.765 4.5 4.5 0 018.302-3.046 3.5 3.5 0 014.504 4.272A4 4 0 0115 17H5.5z" clipRule="evenodd" />
              </svg>
            </div>
          </div>
          <div>
            <p className="text-sm font-semibold text-white">
              {Math.round(precipitationProbability * 100)}% chance
            </p>
            <p className="text-xs text-white/50">
              {precipitation > 0 ? `${precipitation.toFixed(1)} mm expected` : "No rain expected"}
            </p>
          </div>
        </div>

        {/* Rain intensity indicator */}
        {precipitationProbability > 0.1 && (
          <div className="flex gap-1">
            {[...Array(5)].map((_, i) => (
              <div
                key={i}
                className={`w-1.5 h-4 rounded-full transition-colors ${
                  i < Math.ceil(precipitationProbability * 5)
                    ? "bg-blue-400"
                    : "bg-white/10"
                }`}
              />
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );
}
