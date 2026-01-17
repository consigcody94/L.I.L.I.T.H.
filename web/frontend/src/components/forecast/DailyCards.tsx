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
}

function getWeatherIcon(precip: number, precipProb: number): string {
  if (precipProb > 0.6 && precip > 5) return "🌧️";
  if (precipProb > 0.4) return "🌦️";
  if (precipProb > 0.2) return "⛅";
  return "☀️";
}

function getTempGradient(high: number, low: number): string {
  const getColor = (temp: number) => {
    if (temp < 0) return "rgb(96, 165, 250)"; // blue-400
    if (temp < 10) return "rgb(34, 211, 238)"; // cyan-400
    if (temp < 20) return "rgb(74, 222, 128)"; // green-400
    if (temp < 30) return "rgb(251, 191, 36)"; // amber-400
    return "rgb(248, 113, 113)"; // red-400
  };

  return `linear-gradient(to top, ${getColor(low)}, ${getColor(high)})`;
}

export function DailyCards({ forecasts }: DailyCardsProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
      {forecasts.map((forecast, index) => {
        const date = parseISO(forecast.date);
        const icon = getWeatherIcon(
          forecast.precipitation,
          forecast.precipitation_probability
        );

        return (
          <motion.div
            key={forecast.date}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            className="bg-white/5 backdrop-blur-sm border border-white/10 rounded-xl p-4 text-center hover:bg-white/10 transition-colors cursor-pointer"
          >
            {/* Day name */}
            <p className="text-sm font-medium text-white/80">
              {format(date, "EEE")}
            </p>
            <p className="text-xs text-white/50 mb-2">
              {format(date, "MMM d")}
            </p>

            {/* Weather icon */}
            <div className="text-3xl my-2">{icon}</div>

            {/* Temperature bar */}
            <div className="flex items-center justify-center gap-2 mb-2">
              <div
                className="w-1 h-8 rounded-full"
                style={{
                  background: getTempGradient(
                    forecast.temperature_max,
                    forecast.temperature_min
                  ),
                }}
              />
              <div className="text-right">
                <p className="text-sm font-semibold text-white">
                  {Math.round(forecast.temperature_max)}°
                </p>
                <p className="text-xs text-white/60">
                  {Math.round(forecast.temperature_min)}°
                </p>
              </div>
            </div>

            {/* Precipitation */}
            {forecast.precipitation_probability > 0.1 && (
              <p className="text-xs text-blue-400">
                💧 {Math.round(forecast.precipitation_probability * 100)}%
              </p>
            )}
          </motion.div>
        );
      })}
    </div>
  );
}
