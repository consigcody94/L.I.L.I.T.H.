"use client";

import { motion } from "framer-motion";
import { format, parseISO } from "date-fns";

interface DailyForecast {
  date: string;
  temperature_max: number;
  temperature_min: number;
  precipitation: number;
  precipitation_probability: number;
}

interface DailyCardsProps {
  forecasts: DailyForecast[];
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

function getWeatherIcon(precip: number, precipProb: number): string {
  if (precipProb > 0.6 && precip > 5) return "🌧️";
  if (precipProb > 0.4) return "🌦️";
  if (precipProb > 0.2) return "⛅";
  return "☀️";
}

function getTempColor(temp: number): { text: string; bg: string; glow: string } {
  if (temp < 0) return { text: "text-blue-400", bg: "bg-blue-500", glow: "shadow-blue-500/50" };
  if (temp < 10) return { text: "text-cyan-400", bg: "bg-cyan-500", glow: "shadow-cyan-500/50" };
  if (temp < 20) return { text: "text-emerald-400", bg: "bg-emerald-500", glow: "shadow-emerald-500/50" };
  if (temp < 30) return { text: "text-amber-400", bg: "bg-amber-500", glow: "shadow-amber-500/50" };
  return { text: "text-red-400", bg: "bg-red-500", glow: "shadow-red-500/50" };
}

function getTempGradient(high: number, low: number): string {
  const getColor = (temp: number) => {
    if (temp < 0) return "rgb(59, 130, 246)"; // blue-500
    if (temp < 10) return "rgb(6, 182, 212)"; // cyan-500
    if (temp < 20) return "rgb(16, 185, 129)"; // emerald-500
    if (temp < 30) return "rgb(245, 158, 11)"; // amber-500
    return "rgb(239, 68, 68)"; // red-500
  };

  return `linear-gradient(to top, ${getColor(low)}, ${getColor(high)})`;
}

export function DailyCards({ forecasts, unit = "C" }: DailyCardsProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
      {forecasts.map((forecast, index) => {
        const date = parseISO(forecast.date);
        const icon = getWeatherIcon(
          forecast.precipitation,
          forecast.precipitation_probability
        );

        const displayHigh = convertTemp(forecast.temperature_max, unit);
        const displayLow = convertTemp(forecast.temperature_min, unit);
        const highColors = getTempColor(forecast.temperature_max);
        const isToday = index === 0;

        return (
          <motion.div
            key={forecast.date}
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: index * 0.04, type: "spring", stiffness: 100 }}
            whileHover={{ scale: 1.05, y: -4 }}
            className={`
              group relative overflow-hidden rounded-2xl p-4 text-center cursor-pointer
              transition-all duration-300
              ${isToday
                ? "bg-gradient-to-br from-purple-500/20 to-cyan-500/20 border-2 border-purple-500/30"
                : "bg-white/[0.04] border border-white/[0.08] hover:bg-white/[0.08] hover:border-white/[0.15]"
              }
            `}
          >
            {/* Hover glow effect */}
            <div className={`absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 ${highColors.bg}/10 blur-xl`} />

            {/* Today badge */}
            {isToday && (
              <div className="absolute -top-1 -right-1">
                <span className="inline-flex items-center px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-purple-500 text-white rounded-full shadow-lg shadow-purple-500/50">
                  Today
                </span>
              </div>
            )}

            {/* Day name */}
            <p className={`text-sm font-semibold ${isToday ? "text-white" : "text-white/70"}`}>
              {format(date, "EEE")}
            </p>
            <p className="text-xs text-white/40 mb-3">
              {format(date, "MMM d")}
            </p>

            {/* Weather icon with glow */}
            <div className="relative my-3">
              <span className="text-4xl relative z-10 drop-shadow-lg">{icon}</span>
              <div className="absolute inset-0 blur-xl opacity-50 text-4xl flex items-center justify-center">
                {icon}
              </div>
            </div>

            {/* Temperature display */}
            <div className="space-y-1">
              <div className="flex items-center justify-center gap-3">
                {/* Temperature bar */}
                <div
                  className="w-1.5 h-10 rounded-full shadow-lg"
                  style={{
                    background: getTempGradient(
                      forecast.temperature_max,
                      forecast.temperature_min
                    ),
                    boxShadow: `0 0 12px ${getTempGradient(forecast.temperature_max, forecast.temperature_min).includes("239, 68, 68") ? "rgba(239, 68, 68, 0.4)" : "rgba(6, 182, 212, 0.4)"}`
                  }}
                />

                <div className="text-left">
                  <p className={`text-lg font-bold ${highColors.text}`}>
                    {Math.round(displayHigh)}°
                  </p>
                  <p className="text-sm text-white/50 font-medium">
                    {Math.round(displayLow)}°
                  </p>
                </div>
              </div>
            </div>

            {/* Precipitation indicator */}
            {forecast.precipitation_probability > 0.1 && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: index * 0.04 + 0.2 }}
                className="mt-3 flex items-center justify-center gap-1"
              >
                <svg className="w-3.5 h-3.5 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M5.5 17a4.5 4.5 0 01-1.44-8.765 4.5 4.5 0 018.302-3.046 3.5 3.5 0 014.504 4.272A4 4 0 0115 17H5.5z" clipRule="evenodd" />
                </svg>
                <span className="text-xs font-medium text-blue-400">
                  {Math.round(forecast.precipitation_probability * 100)}%
                </span>
              </motion.div>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}
