"use client";

import { useState } from "react";
import { GlassCard } from "@/components/ui/GlassCard";
import { TemperatureDisplay } from "@/components/forecast/TemperatureDisplay";
import { ForecastChart } from "@/components/charts/ForecastChart";
import { LocationSearch } from "@/components/LocationSearch";
import { DailyCards } from "@/components/forecast/DailyCards";
import { WeatherBackground } from "@/components/ui/WeatherBackground";
import { useForecast } from "@/hooks/useForecast";

export default function Home() {
  const [location, setLocation] = useState({
    latitude: 40.7128,
    longitude: -74.006,
    name: "New York, NY",
  });

  const { data: forecast, isLoading, error } = useForecast(location);

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Dynamic weather background */}
      <WeatherBackground condition={forecast?.forecasts[0] ? "clear" : "clear"} />

      {/* Content */}
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gradient mb-2">LILITH</h1>
          <p className="text-white/70 text-lg">
            Long-range Intelligent Learning for Integrated Trend Hindcasting
          </p>
        </header>

        {/* Location Search */}
        <div className="max-w-xl mx-auto mb-8">
          <LocationSearch onLocationSelect={setLocation} />
        </div>

        {/* Current Location */}
        <div className="text-center mb-8">
          <h2 className="text-2xl font-semibold text-white">{location.name}</h2>
          <p className="text-white/50">
            {location.latitude.toFixed(2)}°, {location.longitude.toFixed(2)}°
          </p>
        </div>

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
              <ForecastChart data={forecast.forecasts} />
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
            <DailyCards forecasts={forecast.forecasts.slice(0, 14)} />
          ) : (
            <p className="text-white/50">No forecast available</p>
          )}
        </GlassCard>

        {/* Footer */}
        <footer className="mt-12 text-center text-white/50 text-sm">
          <p>
            LILITH - Open Source Weather Forecasting |{" "}
            <a
              href="https://github.com/lilith-project/lilith"
              className="text-sky-clear hover:underline"
            >
              GitHub
            </a>
          </p>
          <p className="mt-2">
            Model version: {forecast?.model_version || "demo-v1"} | Generated:{" "}
            {forecast?.generated_at
              ? new Date(forecast.generated_at).toLocaleString()
              : "N/A"}
          </p>
        </footer>
      </div>
    </main>
  );
}
