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

function getTempColor(temp: number): string {
  if (temp < 0) return "text-blue-400";
  if (temp < 10) return "text-cyan-400";
  if (temp < 15) return "text-teal-400";
  if (temp < 20) return "text-green-400";
  if (temp < 25) return "text-yellow-400";
  if (temp < 30) return "text-orange-400";
  return "text-red-400";
}

export function TemperatureDisplay({
  high,
  low,
  precipitation,
  precipitationProbability,
  unit = "C",
}: TemperatureDisplayProps) {
  const highColor = useMemo(() => getTempColor(high), [high]);
  const lowColor = useMemo(() => getTempColor(low), [low]);

  return (
    <div className="space-y-4">
      {/* Main temperature display */}
      <div className="flex items-end justify-center gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center"
        >
          <p className="text-sm text-white/60 mb-1">High</p>
          <p className={`text-5xl font-bold ${highColor}`}>
            {Math.round(high)}°
          </p>
        </motion.div>

        <div className="h-12 w-px bg-white/20" />

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-center"
        >
          <p className="text-sm text-white/60 mb-1">Low</p>
          <p className={`text-5xl font-bold ${lowColor}`}>
            {Math.round(low)}°
          </p>
        </motion.div>
      </div>

      {/* Temperature bar */}
      <div className="relative h-2 bg-white/10 rounded-full overflow-hidden">
        <motion.div
          className="absolute h-full bg-gradient-to-r from-blue-500 via-green-400 to-red-500"
          initial={{ width: 0 }}
          animate={{ width: "100%" }}
          transition={{ duration: 0.5 }}
        />
        {/* Low marker */}
        <motion.div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full border-2 border-blue-400"
          style={{ left: `${((low + 20) / 60) * 100}%` }}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.3 }}
        />
        {/* High marker */}
        <motion.div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full border-2 border-red-400"
          style={{ left: `${((high + 20) / 60) * 100}%` }}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.4 }}
        />
      </div>

      {/* Precipitation info */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="flex items-center justify-between pt-4 border-t border-white/10"
      >
        <div className="flex items-center gap-2">
          <svg
            className="w-5 h-5 text-blue-400"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M5.5 17a4.5 4.5 0 01-1.44-8.765 4.5 4.5 0 018.302-3.046 3.5 3.5 0 014.504 4.272A4 4 0 0115 17H5.5zm3.75-2.75a.75.75 0 001.5 0V9.66l1.95 2.1a.75.75 0 101.1-1.02l-3.25-3.5a.75.75 0 00-1.1 0l-3.25 3.5a.75.75 0 101.1 1.02l1.95-2.1v4.59z"
              clipRule="evenodd"
            />
          </svg>
          <span className="text-white/80">
            {Math.round(precipitationProbability * 100)}%
          </span>
        </div>
        <div className="text-white/60">
          {precipitation > 0 ? `${precipitation.toFixed(1)} mm` : "No rain"}
        </div>
      </motion.div>
    </div>
  );
}
