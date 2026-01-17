"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useWeatherStore } from "@/stores/weatherStore";

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

export function Settings({ isOpen, onClose }: SettingsProps) {
  const {
    temperatureUnit,
    setTemperatureUnit,
    showUncertainty,
    setShowUncertainty,
    theme,
    setTheme,
    savedLocations,
    removeSavedLocation,
  } = useWeatherStore();

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
          />

          {/* Panel */}
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-full max-w-md bg-slate-900/95 backdrop-blur-xl border-l border-white/10 z-50 overflow-y-auto"
          >
            {/* Header */}
            <div className="sticky top-0 bg-slate-900/80 backdrop-blur-sm border-b border-white/10 p-4 flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Settings</h2>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              >
                <svg
                  className="w-5 h-5 text-white/70"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            <div className="p-4 space-y-6">
              {/* Temperature Unit */}
              <div>
                <h3 className="text-sm font-medium text-white/80 mb-3">
                  Temperature Unit
                </h3>
                <div className="flex gap-2">
                  <button
                    onClick={() => setTemperatureUnit("C")}
                    className={`flex-1 py-2 px-4 rounded-lg transition-colors ${
                      temperatureUnit === "C"
                        ? "bg-sky-500 text-white"
                        : "bg-white/10 text-white/70 hover:bg-white/20"
                    }`}
                  >
                    Celsius (°C)
                  </button>
                  <button
                    onClick={() => setTemperatureUnit("F")}
                    className={`flex-1 py-2 px-4 rounded-lg transition-colors ${
                      temperatureUnit === "F"
                        ? "bg-sky-500 text-white"
                        : "bg-white/10 text-white/70 hover:bg-white/20"
                    }`}
                  >
                    Fahrenheit (°F)
                  </button>
                </div>
              </div>

              {/* Show Uncertainty */}
              <div>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-white/80">
                      Show Uncertainty
                    </h3>
                    <p className="text-xs text-white/50 mt-1">
                      Display confidence intervals in forecasts
                    </p>
                  </div>
                  <button
                    onClick={() => setShowUncertainty(!showUncertainty)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      showUncertainty ? "bg-sky-500" : "bg-white/20"
                    }`}
                  >
                    <motion.div
                      className="absolute top-1 w-4 h-4 bg-white rounded-full shadow"
                      animate={{ left: showUncertainty ? 28 : 4 }}
                      transition={{ type: "spring", stiffness: 500, damping: 30 }}
                    />
                  </button>
                </div>
              </div>

              {/* Theme */}
              <div>
                <h3 className="text-sm font-medium text-white/80 mb-3">Theme</h3>
                <div className="flex gap-2">
                  {(["dark", "light", "auto"] as const).map((t) => (
                    <button
                      key={t}
                      onClick={() => setTheme(t)}
                      className={`flex-1 py-2 px-4 rounded-lg capitalize transition-colors ${
                        theme === t
                          ? "bg-sky-500 text-white"
                          : "bg-white/10 text-white/70 hover:bg-white/20"
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              </div>

              {/* Saved Locations */}
              <div>
                <h3 className="text-sm font-medium text-white/80 mb-3">
                  Saved Locations ({savedLocations.length})
                </h3>
                {savedLocations.length === 0 ? (
                  <p className="text-sm text-white/50">No saved locations</p>
                ) : (
                  <div className="space-y-2">
                    {savedLocations.map((location) => (
                      <div
                        key={`${location.latitude}-${location.longitude}`}
                        className="flex items-center justify-between p-3 bg-white/5 rounded-lg"
                      >
                        <div>
                          <p className="text-white">{location.name}</p>
                          <p className="text-xs text-white/50">
                            {location.latitude.toFixed(2)}°,{" "}
                            {location.longitude.toFixed(2)}°
                          </p>
                        </div>
                        <button
                          onClick={() =>
                            removeSavedLocation(
                              location.latitude,
                              location.longitude
                            )
                          }
                          className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                        >
                          <svg
                            className="w-4 h-4 text-red-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                            />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* About */}
              <div className="pt-4 border-t border-white/10">
                <h3 className="text-sm font-medium text-white/80 mb-3">About</h3>
                <div className="space-y-2 text-sm text-white/60">
                  <p>LILITH v1.0.0</p>
                  <p>Long-range Intelligent Learning for Integrated Trend Hindcasting</p>
                  <p className="pt-2">
                    Open source weather forecasting powered by machine learning.
                  </p>
                  <a
                    href="https://github.com/lilith-project/lilith"
                    className="inline-block text-sky-400 hover:underline mt-2"
                  >
                    View on GitHub →
                  </a>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
