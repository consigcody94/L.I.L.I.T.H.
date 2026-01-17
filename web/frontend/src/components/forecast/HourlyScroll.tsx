"use client";

import { useRef } from "react";
import { motion } from "framer-motion";
import { format, parseISO } from "date-fns";
import type { HourlyForecast } from "@/hooks/useForecast";

interface HourlyScrollProps {
  forecasts: HourlyForecast[];
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

// Get weather icon based on conditions
function getWeatherIcon(hour: number, cloudCover: number, precipProb: number, precip: number): string {
  const isNight = hour < 6 || hour >= 20;

  if (precipProb > 0.6 && precip > 0.5) {
    return isNight ? "🌧️" : "🌧️";
  }
  if (precipProb > 0.4) {
    return isNight ? "🌧️" : "🌦️";
  }
  if (cloudCover > 70) {
    return isNight ? "☁️" : "☁️";
  }
  if (cloudCover > 40) {
    return isNight ? "🌙" : "⛅";
  }
  return isNight ? "🌙" : "☀️";
}

// Get wind direction arrow
function getWindArrow(degrees: number): string {
  const arrows = ["↓", "↙", "←", "↖", "↑", "↗", "→", "↘"];
  const index = Math.round(degrees / 45) % 8;
  return arrows[index];
}

// Get temperature color
function getTempColor(tempC: number): string {
  if (tempC < 0) return "text-blue-400";
  if (tempC < 10) return "text-cyan-400";
  if (tempC < 15) return "text-teal-400";
  if (tempC < 20) return "text-green-400";
  if (tempC < 25) return "text-yellow-400";
  if (tempC < 30) return "text-orange-400";
  return "text-red-400";
}

export function HourlyScroll({ forecasts, unit = "C" }: HourlyScrollProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  const scroll = (direction: "left" | "right") => {
    if (scrollRef.current) {
      const scrollAmount = 300;
      scrollRef.current.scrollBy({
        left: direction === "left" ? -scrollAmount : scrollAmount,
        behavior: "smooth",
      });
    }
  };

  return (
    <div className="relative group/scroll">
      {/* Scroll buttons */}
      <button
        onClick={() => scroll("left")}
        className="absolute left-0 top-1/2 -translate-y-1/2 z-10 bg-white/[0.08] hover:bg-white/[0.15] backdrop-blur-md p-2.5 rounded-xl shadow-lg border border-white/[0.1] opacity-0 group-hover/scroll:opacity-100 transition-all duration-300 hover:scale-110"
      >
        <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
      </button>
      <button
        onClick={() => scroll("right")}
        className="absolute right-0 top-1/2 -translate-y-1/2 z-10 bg-white/[0.08] hover:bg-white/[0.15] backdrop-blur-md p-2.5 rounded-xl shadow-lg border border-white/[0.1] opacity-0 group-hover/scroll:opacity-100 transition-all duration-300 hover:scale-110"
      >
        <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
      </button>

      {/* Scrollable container */}
      <div
        ref={scrollRef}
        className="flex gap-4 overflow-x-auto scrollbar-hide px-10 py-2"
        style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
      >
        {forecasts.map((hour, index) => {
          const time = parseISO(hour.datetime);
          const icon = getWeatherIcon(hour.hour, hour.cloud_cover, hour.precipitation_probability, hour.precipitation);
          const displayTemp = convertTemp(hour.temperature, unit);
          const displayFeelsLike = convertTemp(hour.feels_like, unit);
          const tempColor = getTempColor(hour.temperature);
          const isNewDay = index === 0 || hour.hour === 0;
          const isNow = index === 0;

          return (
            <motion.div
              key={hour.datetime}
              initial={{ opacity: 0, y: 20, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ delay: index * 0.02, type: "spring", stiffness: 120 }}
              whileHover={{ scale: 1.05, y: -4 }}
              className={`
                group flex-shrink-0 w-[88px] rounded-2xl p-3 text-center cursor-pointer
                transition-all duration-300 relative overflow-hidden
                ${isNow
                  ? "bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border-2 border-cyan-500/30"
                  : isNewDay
                    ? "bg-white/[0.04] border border-purple-500/30 border-l-2 border-l-purple-500"
                    : "bg-white/[0.03] border border-white/[0.06] hover:bg-white/[0.08] hover:border-white/[0.12]"
                }
              `}
            >
              {/* Now badge */}
              {isNow && (
                <div className="absolute -top-0.5 -right-0.5">
                  <span className="inline-flex items-center px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-wider bg-cyan-500 text-white rounded-full shadow-lg shadow-cyan-500/50">
                    Now
                  </span>
                </div>
              )}

              {/* Time */}
              {isNewDay && (
                <p className="text-[10px] text-purple-400/80 font-medium mb-0.5">
                  {format(time, "MMM d")}
                </p>
              )}
              <p className={`text-sm font-semibold ${isNow ? "text-white" : "text-white/70"}`}>
                {format(time, "h a")}
              </p>

              {/* Weather icon with glow */}
              <div className="relative my-2">
                <span className="text-3xl relative z-10 drop-shadow-lg">{icon}</span>
                <div className="absolute inset-0 blur-lg opacity-40 text-3xl flex items-center justify-center">
                  {icon}
                </div>
              </div>

              {/* Temperature */}
              <p className={`text-xl font-bold ${tempColor}`}>
                {Math.round(displayTemp)}°
              </p>
              <p className="text-[10px] text-white/40 mt-0.5">
                Feels {Math.round(displayFeelsLike)}°
              </p>

              {/* Precipitation */}
              {hour.precipitation_probability > 0.1 && (
                <div className="mt-2 flex items-center justify-center gap-1">
                  <svg className="w-3 h-3 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M5.5 17a4.5 4.5 0 01-1.44-8.765 4.5 4.5 0 018.302-3.046 3.5 3.5 0 014.504 4.272A4 4 0 0115 17H5.5z" clipRule="evenodd" />
                  </svg>
                  <span className="text-[10px] font-medium text-blue-400">
                    {Math.round(hour.precipitation_probability * 100)}%
                  </span>
                </div>
              )}

              {/* Wind */}
              <div className="mt-1.5 pt-1.5 border-t border-white/[0.06]">
                <div className="text-[10px] text-white/40 flex items-center justify-center gap-1">
                  <span className="font-medium">{getWindArrow(hour.wind_direction)}</span>
                  <span>{Math.round(hour.wind_speed)} m/s</span>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Gradient overlays */}
      <div className="absolute left-10 top-0 bottom-0 w-12 bg-gradient-to-r from-black/40 to-transparent pointer-events-none" />
      <div className="absolute right-10 top-0 bottom-0 w-12 bg-gradient-to-l from-black/40 to-transparent pointer-events-none" />
    </div>
  );
}
