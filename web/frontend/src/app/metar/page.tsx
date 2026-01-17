"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";

interface MetarStation {
  icao: string;
  name: string;
  latitude: number;
  longitude: number;
  raw_metar: string | null;
  observation_time: string | null;
  is_flagged: boolean;
  is_missing: boolean;
  temperature_c: number | null;
  dewpoint_c: number | null;
  wind_speed_kt: number | null;
  wind_dir: number | null;
  visibility_sm: number | null;
  altimeter_inhg: number | null;
  weather: string | null;
  clouds: string | null;
  last_checked: string | null;
}

interface MetarResponse {
  generated_at: string;
  total_stations: number;
  flagged_count: number;
  missing_count: number;
  healthy_count: number;
  stations: MetarStation[];
  next_update_seconds: number;
}

function CountdownTimer({ seconds }: { seconds: number }) {
  const [timeLeft, setTimeLeft] = useState(seconds);

  useEffect(() => {
    setTimeLeft(seconds);
  }, [seconds]);

  useEffect(() => {
    const interval = setInterval(() => {
      setTimeLeft((prev) => Math.max(0, prev - 1));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const minutes = Math.floor(timeLeft / 60);
  const secs = timeLeft % 60;

  return (
    <div className="flex items-center gap-1 text-sm">
      <span className="bg-purple-500/20 text-purple-300 px-2 py-1 rounded-lg font-mono font-bold min-w-[2.5rem] text-center">
        {String(minutes).padStart(2, "0")}
      </span>
      <span className="text-white/30">:</span>
      <span className="bg-purple-500/20 text-purple-300 px-2 py-1 rounded-lg font-mono font-bold min-w-[2.5rem] text-center">
        {String(secs).padStart(2, "0")}
      </span>
    </div>
  );
}

function StationCard({ station }: { station: MetarStation }) {
  const getStatusColor = () => {
    if (station.is_flagged) return "border-red-500/50 bg-red-500/10";
    if (station.is_missing) return "border-yellow-500/50 bg-yellow-500/10";
    return "border-green-500/30 bg-green-500/5";
  };

  const getStatusBadge = () => {
    if (station.is_flagged) {
      return (
        <motion.span
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ repeat: Infinity, duration: 1 }}
          className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-red-500 text-white font-bold"
        >
          $ MAINTENANCE
        </motion.span>
      );
    }
    if (station.is_missing) {
      return (
        <motion.span
          animate={{ opacity: [1, 0.5, 1] }}
          transition={{ repeat: Infinity, duration: 1.5 }}
          className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-yellow-500 text-black font-bold"
        >
          STALE DATA
        </motion.span>
      );
    }
    return (
      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs bg-green-500/20 text-green-400">
        OPERATIONAL
      </span>
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className={`relative rounded-xl border-2 p-4 transition-all ${getStatusColor()} ${
        station.is_flagged ? "animate-pulse" : ""
      }`}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-3">
        <div>
          <h3 className="text-lg font-bold text-white">{station.icao}</h3>
          <p className="text-sm text-white/60">{station.name}</p>
        </div>
        {getStatusBadge()}
      </div>

      {/* Weather Data */}
      {station.raw_metar && (
        <>
          <div className="grid grid-cols-3 gap-2 mb-3">
            {station.temperature_c !== null && (
              <div className="text-center">
                <p className="text-xs text-white/50">Temp</p>
                <p className="text-lg font-bold text-cyan-400">
                  {station.temperature_c}°C
                </p>
              </div>
            )}
            {station.wind_speed_kt !== null && (
              <div className="text-center">
                <p className="text-xs text-white/50">Wind</p>
                <p className="text-lg font-bold text-white">
                  {station.wind_speed_kt}kt
                </p>
              </div>
            )}
            {station.visibility_sm !== null && (
              <div className="text-center">
                <p className="text-xs text-white/50">Vis</p>
                <p className="text-lg font-bold text-white">
                  {station.visibility_sm}SM
                </p>
              </div>
            )}
          </div>

          {/* Raw METAR */}
          <div className="bg-black/30 rounded-lg p-2 overflow-x-auto">
            <code className="text-xs text-white/70 whitespace-nowrap">
              {station.raw_metar}
            </code>
          </div>
        </>
      )}

      {!station.raw_metar && (
        <div className="text-center py-4 text-white/40">
          No METAR data available
        </div>
      )}

      {/* Observation Time */}
      {station.observation_time && (
        <p className="text-xs text-white/40 mt-2 text-right">
          Obs: {station.observation_time}
        </p>
      )}
    </motion.div>
  );
}

export default function MetarMonitorPage() {
  const [data, setData] = useState<MetarResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<"all" | "flagged" | "missing" | "healthy">("all");
  const [refreshing, setRefreshing] = useState(false);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const fetchMetar = useCallback(async () => {
    try {
      const response = await fetch(`${API_URL}/v1/metar`);
      if (!response.ok) throw new Error("Failed to fetch METAR data");
      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [API_URL]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await fetch(`${API_URL}/v1/metar/refresh`, { method: "POST" });
      await fetchMetar();
    } catch (err) {
      console.error("Refresh failed:", err);
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchMetar();
    const interval = setInterval(fetchMetar, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [fetchMetar]);

  const filteredStations = data?.stations.filter((s) => {
    switch (filter) {
      case "flagged":
        return s.is_flagged;
      case "missing":
        return s.is_missing;
      case "healthy":
        return !s.is_flagged && !s.is_missing;
      default:
        return true;
    }
  });

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500"></div>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 p-6">
      {/* Header */}
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
            >
              <svg
                className="w-6 h-6 text-white/70"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M10 19l-7-7m0 0l7-7m-7 7h18"
                />
              </svg>
            </Link>
            <div>
              <h1 className="text-3xl font-bold text-white">METAR Monitor</h1>
              <p className="text-white/60">
                Real-time METAR station monitoring
              </p>
            </div>
          </div>

          {/* Countdown */}
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-xs text-white/50">Next auto-update</p>
              {data && <CountdownTimer seconds={data.next_update_seconds} />}
            </div>
            <button
              onClick={handleRefresh}
              disabled={refreshing}
              className="px-4 py-2 bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 rounded-lg transition-colors disabled:opacity-50"
            >
              {refreshing ? "Refreshing..." : "Refresh Now"}
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        {data && (
          <div className="grid grid-cols-4 gap-4 mb-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white/5 rounded-xl p-4 border border-white/10"
            >
              <p className="text-sm text-white/50">Total Stations</p>
              <p className="text-3xl font-bold text-white">
                {data.total_stations}
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="bg-green-500/10 rounded-xl p-4 border border-green-500/30"
            >
              <p className="text-sm text-green-400/70">Healthy</p>
              <p className="text-3xl font-bold text-green-400">
                {data.healthy_count}
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-yellow-500/10 rounded-xl p-4 border border-yellow-500/30"
            >
              <p className="text-sm text-yellow-400/70">Stale Data</p>
              <p className="text-3xl font-bold text-yellow-400">
                {data.missing_count}
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className={`rounded-xl p-4 border ${
                data.flagged_count > 0
                  ? "bg-red-500/20 border-red-500/50 animate-pulse"
                  : "bg-red-500/10 border-red-500/30"
              }`}
            >
              <p className="text-sm text-red-400/70">Flagged ($)</p>
              <p className="text-3xl font-bold text-red-400">
                {data.flagged_count}
              </p>
            </motion.div>
          </div>
        )}

        {/* Filter Tabs */}
        <div className="flex gap-2 mb-6">
          {(["all", "healthy", "flagged", "missing"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                filter === f
                  ? "bg-purple-500 text-white"
                  : "bg-white/5 text-white/60 hover:bg-white/10"
              }`}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
              {f !== "all" && data && (
                <span className="ml-2 opacity-70">
                  (
                  {f === "healthy"
                    ? data.healthy_count
                    : f === "flagged"
                    ? data.flagged_count
                    : data.missing_count}
                  )
                </span>
              )}
            </button>
          ))}
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-xl p-4 mb-6">
            <p className="text-red-400">{error}</p>
          </div>
        )}

        {/* Station Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          <AnimatePresence>
            {filteredStations?.map((station) => (
              <StationCard key={station.icao} station={station} />
            ))}
          </AnimatePresence>
        </div>

        {/* Empty State */}
        {filteredStations?.length === 0 && (
          <div className="text-center py-12">
            <p className="text-white/50 text-lg">
              No stations match the current filter
            </p>
          </div>
        )}

        {/* Legend */}
        <div className="mt-8 p-4 bg-white/5 rounded-xl border border-white/10">
          <h3 className="text-sm font-medium text-white/70 mb-2">Legend</h3>
          <div className="flex flex-wrap gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-white/60">Operational - Normal METAR</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span className="text-white/60">Stale - Data older than 90 minutes</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse"></div>
              <span className="text-white/60">
                Flagged ($) - ASOS needs maintenance
              </span>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
