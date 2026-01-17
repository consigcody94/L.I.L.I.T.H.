"use client";

import { useState } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import { GlassCard } from "@/components/ui/GlassCard";
import { TemperatureDisplay } from "@/components/forecast/TemperatureDisplay";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { LocationSearch } from "@/components/LocationSearch";
import { DailyCards } from "@/components/forecast/DailyCards";
import { HourlyScroll } from "@/components/forecast/HourlyScroll";
import { WeatherBackground } from "@/components/ui/WeatherBackground";
import { Settings } from "@/components/Settings";
import { AccuracyTracker } from "@/components/accuracy/AccuracyTracker";
import { useForecast, useHourlyForecast, useAccuracyReport } from "@/hooks/useForecast";
import { useWeatherStore } from "@/stores/weatherStore";

export default function Home() {
  const [location, setLocation] = useState({
    latitude: 40.7128,
    longitude: -74.006,
    name: "New York, NY",
  });

  const [settingsOpen, setSettingsOpen] = useState(false);

  const { temperatureUnit, setTemperatureUnit } = useWeatherStore();
  const { data: forecast, isLoading, error } = useForecast(location);
  const { data: hourlyForecast, isLoading: hourlyLoading } = useHourlyForecast(location, 48);
  const { data: accuracyReport, isLoading: accuracyLoading } = useAccuracyReport(location, 30);

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Dynamic weather background */}
      <WeatherBackground condition={forecast?.forecasts[0] ? "clear" : "clear"} />

      {/* Settings Panel */}
      <Settings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {/* Content */}
      <div className="relative z-10 min-h-screen flex flex-col">
        {/* Header */}
        <header className="w-full py-4 px-4 border-b border-white/[0.06] backdrop-blur-md bg-black/20">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            {/* Logo and Title */}
            <Link href="/" className="flex items-center gap-4 group">
              <div className="relative">
                <div className="absolute inset-0 bg-purple-500/30 blur-2xl rounded-full scale-150 group-hover:bg-purple-400/40 transition-all duration-500" />
                <img
                  src="/images/logo.png"
                  alt="LILITH"
                  className="h-14 w-auto object-contain drop-shadow-[0_0_25px_rgba(139,92,246,0.5)] relative z-10 group-hover:scale-110 transition-transform duration-300"
                />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold bg-gradient-to-r from-white via-purple-200 to-purple-400 bg-clip-text text-transparent">
                  LILITH
                </h1>
                <p className="text-[10px] text-white/40 uppercase tracking-[0.15em] font-medium">
                  AI Weather Intelligence
                </p>
              </div>
            </Link>

            {/* Controls - Right side */}
            <div className="flex items-center gap-2 sm:gap-3">
              {/* Command Center Link */}
              <Link
                href="/stations"
                className="group flex items-center gap-2 px-3 sm:px-4 py-2 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 hover:from-purple-500/20 hover:to-cyan-500/20 border border-purple-500/20 hover:border-purple-400/40 rounded-xl text-purple-300 hover:text-purple-200 transition-all duration-300 text-sm font-medium"
              >
                <svg className="w-4 h-4 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                <span className="hidden sm:inline">Command Center</span>
              </Link>

              {/* METAR Monitor Link */}
              <Link
                href="/metar"
                className="group flex items-center gap-2 px-3 sm:px-4 py-2 bg-gradient-to-r from-cyan-500/10 to-emerald-500/10 hover:from-cyan-500/20 hover:to-emerald-500/20 border border-cyan-500/20 hover:border-cyan-400/40 rounded-xl text-cyan-300 hover:text-cyan-200 transition-all duration-300 text-sm font-medium"
              >
                <svg className="w-4 h-4 group-hover:scale-110 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" />
                </svg>
                <span className="hidden sm:inline">METAR</span>
              </Link>

              {/* Temperature Unit Toggle */}
              <div className="flex bg-white/[0.06] backdrop-blur-sm rounded-xl p-1 border border-white/[0.08]">
                <button
                  onClick={() => setTemperatureUnit("C")}
                  className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-all duration-300 ${
                    temperatureUnit === "C"
                      ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/30"
                      : "text-white/50 hover:text-white hover:bg-white/[0.05]"
                  }`}
                >
                  °C
                </button>
                <button
                  onClick={() => setTemperatureUnit("F")}
                  className={`px-3 py-1.5 rounded-lg text-sm font-semibold transition-all duration-300 ${
                    temperatureUnit === "F"
                      ? "bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/30"
                      : "text-white/50 hover:text-white hover:bg-white/[0.05]"
                  }`}
                >
                  °F
                </button>
              </div>

              {/* Settings Button */}
              <button
                onClick={() => setSettingsOpen(true)}
                className="p-2.5 hover:bg-white/[0.08] rounded-xl transition-all duration-300 border border-transparent hover:border-white/10"
                title="Settings"
              >
                <svg
                  className="w-5 h-5 text-white/50 hover:text-white transition-colors"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                  />
                </svg>
              </button>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <div className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Hero Section - Location & Search */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-8"
          >
            {/* Location Search */}
            <div className="max-w-xl mx-auto mb-6">
              <LocationSearch onLocationSelect={setLocation} />
            </div>

            {/* Current Location Display */}
            <div className="text-center">
              <motion.h2
                key={location.name}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-3xl sm:text-4xl font-bold bg-gradient-to-r from-white via-white to-white/70 bg-clip-text text-transparent mb-2"
              >
                {location.name}
              </motion.h2>
              <div className="flex items-center justify-center gap-4 text-sm text-white/40">
                <span className="flex items-center gap-1.5">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  {location.latitude.toFixed(2)}°N, {Math.abs(location.longitude).toFixed(2)}°W
                </span>
                <span className="w-1 h-1 rounded-full bg-white/20" />
                <span className="text-purple-400/70">90-Day Forecast</span>
              </div>
            </div>
          </motion.div>

          {/* Error Display */}
          {error && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="max-w-lg mx-auto mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-2xl text-red-300 text-center backdrop-blur-sm"
            >
              <svg className="w-6 h-6 mx-auto mb-2 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
              </svg>
              Failed to load forecast. Please try again.
            </motion.div>
          )}

          {/* Main Content Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
            {/* Current Conditions */}
            <GlassCard className="lg:col-span-1" variant="gradient" glow hover>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white/90 flex items-center gap-2">
                  <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
                  </span>
                  Tomorrow
                </h3>
                <span className="text-xs text-white/40 bg-white/[0.05] px-2 py-1 rounded-lg">
                  {new Date(Date.now() + 86400000).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                </span>
              </div>
              {isLoading ? (
                <div className="animate-pulse space-y-4">
                  <div className="h-24 bg-white/[0.05] rounded-xl"></div>
                  <div className="h-3 bg-white/[0.05] rounded-full"></div>
                  <div className="h-12 bg-white/[0.05] rounded-xl"></div>
                </div>
              ) : forecast?.forecasts[0] ? (
                <TemperatureDisplay
                  high={forecast.forecasts[0].temperature_max}
                  low={forecast.forecasts[0].temperature_min}
                  precipitation={forecast.forecasts[0].precipitation}
                  precipitationProbability={forecast.forecasts[0].precipitation_probability}
                  unit={temperatureUnit}
                />
              ) : (
                <p className="text-white/50">No forecast available</p>
              )}
            </GlassCard>

            {/* 90-Day Chart */}
            <GlassCard className="lg:col-span-2" glow>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white/90 flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
                    <svg className="w-4 h-4 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
                    </svg>
                  </div>
                  90-Day Temperature Forecast
                </h3>
                <span className="text-xs text-white/40 bg-white/[0.05] px-3 py-1.5 rounded-lg border border-white/[0.05]">
                  Powered by AI
                </span>
              </div>
              {isLoading ? (
                <div className="animate-pulse h-64 bg-white/[0.05] rounded-xl"></div>
              ) : forecast ? (
                <ForecastChart data={forecast.forecasts} unit={temperatureUnit} />
              ) : (
                <p className="text-white/50">No forecast available</p>
              )}
            </GlassCard>
          </div>

          {/* Hourly Forecast */}
          <GlassCard className="mb-8" hover>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white/90 flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-sky-500/20 flex items-center justify-center">
                  <svg className="w-4 h-4 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                48-Hour Forecast
              </h3>
              <span className="text-xs text-white/40">Scroll to see more</span>
            </div>
            {hourlyLoading ? (
              <div className="flex gap-4 overflow-hidden">
                {[...Array(12)].map((_, i) => (
                  <div key={i} className="flex-shrink-0 w-24 h-44 animate-pulse bg-white/[0.05] rounded-2xl" />
                ))}
              </div>
            ) : hourlyForecast ? (
              <HourlyScroll forecasts={hourlyForecast.forecasts} unit={temperatureUnit} />
            ) : (
              <p className="text-white/50">No hourly forecast available</p>
            )}
          </GlassCard>

          {/* Daily Forecast Cards */}
          <GlassCard className="mb-8" hover>
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-lg font-semibold text-white/90 flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-amber-500/20 flex items-center justify-center">
                  <svg className="w-4 h-4 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                </div>
                14-Day Extended Forecast
              </h3>
              <span className="text-xs text-white/40 bg-white/[0.05] px-3 py-1.5 rounded-lg border border-white/[0.05]">
                Updated hourly
              </span>
            </div>
            {isLoading ? (
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-4">
                {[...Array(7)].map((_, i) => (
                  <div key={i} className="animate-pulse h-36 bg-white/[0.05] rounded-2xl"></div>
                ))}
              </div>
            ) : forecast ? (
              <DailyCards forecasts={forecast.forecasts.slice(0, 14)} unit={temperatureUnit} />
            ) : (
              <p className="text-white/50">No forecast available</p>
            )}
          </GlassCard>

          {/* Prediction Accuracy Tracker */}
          <GlassCard className="mb-8" variant="accent" glow hover>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white/90 flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center">
                  <svg className="w-4 h-4 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                Prediction Accuracy
              </h3>
              <span className="text-xs text-purple-300/70 bg-purple-500/10 px-3 py-1.5 rounded-lg border border-purple-500/20">
                Predictions vs Actual Weather
              </span>
            </div>
            <AccuracyTracker
              report={accuracyReport}
              isLoading={accuracyLoading}
              unit={temperatureUnit}
            />
          </GlassCard>

          {/* Footer */}
          <footer className="mt-16 pt-8 border-t border-white/[0.06]">
            <div className="max-w-4xl mx-auto">
              <div className="flex flex-col md:flex-row items-center justify-between gap-6 mb-8">
                <div className="flex items-center gap-4">
                  <div className="relative">
                    <div className="absolute inset-0 bg-purple-500/20 blur-xl rounded-full" />
                    <img
                      src="/images/logo.png"
                      alt="L.I.L.I.T.H."
                      className="h-12 w-auto relative z-10 opacity-80"
                    />
                  </div>
                  <div className="text-left">
                    <p className="text-white/70 text-sm font-semibold">L.I.L.I.T.H.</p>
                    <p className="text-white/40 text-xs">Open Source Weather AI</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Link
                    href="/stations"
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm text-purple-300/80 hover:text-purple-300 hover:bg-purple-500/10 transition-all duration-300"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                    Command Center
                  </Link>
                  <Link
                    href="/historical"
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm text-white/50 hover:text-white hover:bg-white/[0.05] transition-all duration-300"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Historical
                  </Link>
                  <a
                    href="https://github.com/consigcody94/lilith"
                    className="flex items-center gap-2 px-4 py-2 rounded-xl text-sm text-white/50 hover:text-white hover:bg-white/[0.05] transition-all duration-300"
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                      <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
                    </svg>
                    GitHub
                  </a>
                </div>
              </div>

              {/* Model Info Bar */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-4 p-4 rounded-2xl bg-white/[0.03] border border-white/[0.05]">
                <div className="flex items-center gap-6 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
                    <span className="text-white/40">Model:</span>
                    <span className="text-purple-400 font-medium">{forecast?.model_version || "SimpleLILITH-v1"}</span>
                  </div>
                  <div className="hidden sm:flex items-center gap-2">
                    <span className="text-white/40">Generated:</span>
                    <span className="text-cyan-400/80">{forecast?.generated_at ? new Date(forecast.generated_at).toLocaleString() : "N/A"}</span>
                  </div>
                </div>
                <p className="text-xs text-white/30">
                  Built with GHCN public weather data • Open Source
                </p>
              </div>

              {/* Collaboration Credits */}
              <div className="mt-6 pt-6 border-t border-white/[0.05] text-center">
                <p className="text-xs text-white/40 mb-3">Made in Collaboration with</p>
                <div className="flex items-center justify-center gap-6">
                  <a
                    href="https://sentinelowl.org"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group flex items-center gap-2 px-4 py-2 rounded-xl bg-white/[0.03] border border-white/[0.08] hover:bg-white/[0.08] hover:border-purple-500/30 transition-all duration-300"
                  >
                    <span className="text-sm font-medium text-white/70 group-hover:text-purple-300 transition-colors">
                      SentinelOwl.org
                    </span>
                    <svg className="w-3.5 h-3.5 text-white/40 group-hover:text-purple-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </a>
                  <a
                    href="https://ikigai.one"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="group flex items-center gap-2 px-4 py-2 rounded-xl bg-white/[0.03] border border-white/[0.08] hover:bg-white/[0.08] hover:border-cyan-500/30 transition-all duration-300"
                  >
                    <span className="text-sm font-medium text-white/70 group-hover:text-cyan-300 transition-colors">
                      Ikigai.one
                    </span>
                    <svg className="w-3.5 h-3.5 text-white/40 group-hover:text-cyan-400 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                    </svg>
                  </a>
                </div>
              </div>
            </div>
          </footer>
        </div>
      </div>
    </main>
  );
}
