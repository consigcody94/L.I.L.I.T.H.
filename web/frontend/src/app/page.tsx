"use client";

import { useState } from "react";
import { GlassCard } from "@/components/ui/GlassCard";
import { TemperatureDisplay } from "@/components/forecast/TemperatureDisplay";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { LocationSearch } from "@/components/LocationSearch";
import { DailyCards } from "@/components/forecast/DailyCards";
import { WeatherBackground } from "@/components/ui/WeatherBackground";
import { Settings } from "@/components/Settings";
import { useForecast } from "@/hooks/useForecast";
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

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Dynamic weather background */}
      <WeatherBackground condition={forecast?.forecasts[0] ? "clear" : "clear"} />

      {/* Settings Panel */}
      <Settings isOpen={settingsOpen} onClose={() => setSettingsOpen(false)} />

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header with Settings */}
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div className="text-center flex-1">
              <h1 className="text-5xl font-bold text-gradient mb-2">LILITH</h1>
              <p className="text-white/70 text-lg">
                90-Day Weather Forecasting
              </p>
            </div>

            {/* Temperature Unit Toggle & Settings */}
            <div className="flex items-center gap-3">
              {/* Temperature Unit Toggle */}
              <div className="flex bg-white/10 backdrop-blur-sm rounded-lg p-1">
                <button
                  onClick={() => setTemperatureUnit("C")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    temperatureUnit === "C"
                      ? "bg-sky-500 text-white shadow-lg"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  °C
                </button>
                <button
                  onClick={() => setTemperatureUnit("F")}
                  className={`px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    temperatureUnit === "F"
                      ? "bg-sky-500 text-white shadow-lg"
                      : "text-white/70 hover:text-white"
                  }`}
                >
                  °F
                </button>
              </div>

              {/* Settings Button */}
              <button
                onClick={() => setSettingsOpen(true)}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                title="Settings"
              >
                <svg
                  className="w-6 h-6 text-white/70 hover:text-white"
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

        {/* Location Search */}
        <div className="max-w-xl mx-auto mb-8">
          <LocationSearch onLocationSelect={setLocation} />
        </div>

        {/* Current Location Display */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-semibold text-white">{location.name}</h2>
          <p className="text-white/50">
            {location.latitude.toFixed(4)}°, {location.longitude.toFixed(4)}°
          </p>
        </div>

        {/* Error Display */}
        {error && (
          <div className="max-w-xl mx-auto mb-8 p-4 bg-red-500/20 border border-red-500/50 rounded-xl text-red-200 text-center">
            Failed to load forecast. Please try again.
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Current Conditions */}
          <GlassCard className="lg:col-span-1">
            <h3 className="text-lg font-medium text-white/80 mb-4">Tomorrow</h3>
            {isLoading ? (
              <div className="animate-pulse">
                <div className="h-24 bg-white/10 rounded-lg mb-4"></div>
              </div>
            ) : forecast?.forecasts[0] ? (
              <TemperatureDisplay
                high={forecast.forecasts[0].temperature_max}
                low={forecast.forecasts[0].temperature_min}
                precipitation={forecast.forecasts[0].precipitation}
                precipitationProbability={
                  forecast.forecasts[0].precipitation_probability
                }
                unit={temperatureUnit}
              />
            ) : (
              <p className="text-white/50">No forecast available</p>
            )}
          </GlassCard>

          {/* 90-Day Chart */}
          <GlassCard className="lg:col-span-2">
            <h3 className="text-lg font-medium text-white/80 mb-4">
              90-Day Temperature Forecast
            </h3>
            {isLoading ? (
              <div className="animate-pulse h-64 bg-white/10 rounded-lg"></div>
            ) : forecast ? (
              <ForecastChart data={forecast.forecasts} unit={temperatureUnit} />
            ) : (
              <p className="text-white/50">No forecast available</p>
            )}
          </GlassCard>
        </div>

        {/* Daily Forecast Cards */}
        <GlassCard>
          <h3 className="text-lg font-medium text-white/80 mb-4">
            Extended Forecast
          </h3>
          {isLoading ? (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
              {[...Array(7)].map((_, i) => (
                <div
                  key={i}
                  className="animate-pulse h-32 bg-white/10 rounded-lg"
                ></div>
              ))}
            </div>
          ) : forecast ? (
            <DailyCards forecasts={forecast.forecasts.slice(0, 14)} unit={temperatureUnit} />
          ) : (
            <p className="text-white/50">No forecast available</p>
          )}
        </GlassCard>

        {/* Footer */}
        <footer className="mt-12 text-center text-white/50 text-sm">
          <p>
            LILITH - Open Source Weather Forecasting |{" "}
            <a
              href="https://github.com/consigcody94/lilith"
              className="text-sky-400 hover:underline"
            >
              GitHub
            </a>
          </p>
          <p className="mt-2">
            Model: {forecast?.model_version || "demo-v1"} | Generated:{" "}
            {forecast?.generated_at
              ? new Date(forecast.generated_at).toLocaleString()
              : "N/A"}
          </p>
        </footer>
      </div>
    </main>
  );
}
