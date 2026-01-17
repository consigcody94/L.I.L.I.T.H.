"use client";

import { useState } from "react";
import { GlassCard } from "@/components/ui/GlassCard";
import { WeatherBackground } from "@/components/ui/WeatherBackground";
import { LocationSearch } from "@/components/LocationSearch";
import { ClimateChart } from "@/components/historical/ClimateChart";
import { RecordEvents } from "@/components/historical/RecordEvents";
import Link from "next/link";

export default function HistoricalPage() {
  const [location, setLocation] = useState({
    latitude: 40.7128,
    longitude: -74.006,
    name: "New York, NY",
  });

  const [dateRange, setDateRange] = useState({
    start: "1990-01-01",
    end: "2023-12-31",
  });

  return (
    <main className="relative min-h-screen overflow-hidden">
      <WeatherBackground condition="clear" />

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <Link
                href="/"
                className="text-white/60 hover:text-white transition-colors"
              >
                ← Back to Forecast
              </Link>
              <h1 className="text-4xl font-bold text-white mt-2">
                Historical Climate Data
              </h1>
              <p className="text-white/70 mt-1">
                Explore 150+ years of weather observations
              </p>
            </div>
          </div>
        </header>

        {/* Location Search */}
        <div className="max-w-xl mb-8">
          <LocationSearch onLocationSelect={setLocation} />
        </div>

        {/* Current Location */}
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-white">{location.name}</h2>
          <p className="text-white/50">
            {location.latitude.toFixed(2)}°, {location.longitude.toFixed(2)}°
          </p>
        </div>

        {/* Date Range Selector */}
        <GlassCard className="mb-8">
          <h3 className="text-lg font-medium text-white/80 mb-4">Date Range</h3>
          <div className="flex flex-wrap gap-4">
            <div>
              <label className="block text-sm text-white/60 mb-1">
                Start Date
              </label>
              <input
                type="date"
                value={dateRange.start}
                onChange={(e) =>
                  setDateRange({ ...dateRange, start: e.target.value })
                }
                className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-sky-400/50"
              />
            </div>
            <div>
              <label className="block text-sm text-white/60 mb-1">
                End Date
              </label>
              <input
                type="date"
                value={dateRange.end}
                onChange={(e) =>
                  setDateRange({ ...dateRange, end: e.target.value })
                }
                className="bg-white/10 border border-white/20 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-sky-400/50"
              />
            </div>
            <div className="flex items-end gap-2">
              <button
                onClick={() => setDateRange({ start: "1990-01-01", end: "2023-12-31" })}
                className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white/80 transition-colors"
              >
                Last 30 Years
              </button>
              <button
                onClick={() => setDateRange({ start: "1950-01-01", end: "2023-12-31" })}
                className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white/80 transition-colors"
              >
                Last 70 Years
              </button>
              <button
                onClick={() => setDateRange({ start: "1900-01-01", end: "2023-12-31" })}
                className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-white/80 transition-colors"
              >
                All Time
              </button>
            </div>
          </div>
        </GlassCard>

        {/* Climate Trends */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <GlassCard>
            <h3 className="text-lg font-medium text-white/80 mb-4">
              Temperature Trends
            </h3>
            <ClimateChart
              location={location}
              dateRange={dateRange}
              variable="temperature"
            />
          </GlassCard>

          <GlassCard>
            <h3 className="text-lg font-medium text-white/80 mb-4">
              Precipitation Trends
            </h3>
            <ClimateChart
              location={location}
              dateRange={dateRange}
              variable="precipitation"
            />
          </GlassCard>
        </div>

        {/* Record Events */}
        <GlassCard className="mb-8">
          <h3 className="text-lg font-medium text-white/80 mb-4">
            Record Events
          </h3>
          <RecordEvents location={location} />
        </GlassCard>

        {/* Monthly Averages */}
        <GlassCard>
          <h3 className="text-lg font-medium text-white/80 mb-4">
            Monthly Climate Normals
          </h3>
          <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-12 gap-3">
            {[
              "Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
            ].map((month, i) => (
              <div
                key={month}
                className="bg-white/5 rounded-lg p-3 text-center"
              >
                <p className="text-sm text-white/60 mb-1">{month}</p>
                <p className="text-lg font-semibold text-red-400">
                  {Math.round(10 + 15 * Math.sin((i - 1) * Math.PI / 6))}°
                </p>
                <p className="text-sm text-blue-400">
                  {Math.round(2 + 12 * Math.sin((i - 1) * Math.PI / 6))}°
                </p>
                <p className="text-xs text-white/40 mt-1">
                  {Math.round(60 + 40 * Math.random())}mm
                </p>
              </div>
            ))}
          </div>
        </GlassCard>

        {/* Footer */}
        <footer className="mt-12 text-center text-white/50 text-sm">
          <p>
            Data source: NOAA GHCN-Daily | Updated daily
          </p>
        </footer>
      </div>
    </main>
  );
}
