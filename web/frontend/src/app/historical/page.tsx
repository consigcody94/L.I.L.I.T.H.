"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import axios from "axios";
import Link from "next/link";
import { GlassCard } from "@/components/ui/GlassCard";
import { WeatherBackground } from "@/components/ui/WeatherBackground";

interface Observation {
  date: string;
  temperature_max: number | null;
  temperature_min: number | null;
  precipitation: number | null;
}
interface History {
  station_id: string;
  station_name: string;
  observations: Observation[];
  source_url: string;
  source_sha256: string;
  day_definition: string;
  latitude: number;
  longitude: number;
}

export default function HistoricalPage() {
  const [station, setStation] = useState("USW00094728");
  const [start, setStart] = useState("1980-01-01");
  const [end, setEnd] = useState(new Date().toISOString().slice(0, 10));
  const [request, setRequest] = useState<{ station_id: string; start_date: string; end_date: string } | null>(null);
  const { data, isFetching, isError } = useQuery({
    queryKey: ["station-history", request],
    enabled: request !== null,
    retry: 1,
    queryFn: async () => (await axios.post<History>(
      `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}/v1/historical`, request,
    )).data,
  });
  const years = new Map<string, Observation[]>();
  for (const row of data?.observations || []) {
    const year = row.date.slice(0, 4);
    years.set(year, [...(years.get(year) || []), row]);
  }
  const summary = Array.from(years.entries()).map(([year, rows]) => {
    const highs = rows.flatMap(r => r.temperature_max === null ? [] : [r.temperature_max]);
    const lows = rows.flatMap(r => r.temperature_min === null ? [] : [r.temperature_min]);
    const rain = rows.flatMap(r => r.precipitation === null ? [] : [r.precipitation]);
    const number = Number(year);
    const days = (number % 4 === 0 && (number % 100 !== 0 || number % 400 === 0)) ? 366 : 365;
    const mean = (values: number[]) => values.length ? (values.reduce((a, b) => a + b, 0) / values.length).toFixed(1) : "—";
    return { year, high: mean(highs), low: mean(lows), highCount: highs.length, lowCount: lows.length,
      rain: rain.length === days ? rain.reduce((a, b) => a + b, 0).toFixed(1) : "Incomplete", rainCount: rain.length, days };
  });
  return (
    <main className="relative min-h-screen">
      <WeatherBackground condition="clear" />
      <div className="relative z-10 container mx-auto px-4 py-8 max-w-5xl">
        <Link href="/" className="text-white/70">← Back to Forecast</Link>
        <h1 className="text-3xl font-bold text-white mt-4">Historical station observations</h1>
        <p className="text-white/70 mt-2 mb-6">
          Explore actual NOAA GHCN daily records. Available dates vary by station; many records extend before 1980.
        </p>
        <GlassCard className="mb-6">
          <form onSubmit={event => { event.preventDefault(); setRequest({ station_id: station.trim().toUpperCase(), start_date: start, end_date: end }); }} className="flex flex-wrap gap-4 items-end">
            <label className="text-white/70 text-sm">GHCN station ID
              <input className="block bg-slate-800 rounded-lg p-3 mt-1 text-white" required pattern="[A-Za-z0-9]{11}" value={station} onChange={e => setStation(e.target.value)} />
            </label>
            <label className="text-white/70 text-sm">From
              <input className="block bg-slate-800 rounded-lg p-3 mt-1 text-white" type="date" required value={start} onChange={e => setStart(e.target.value)} />
            </label>
            <label className="text-white/70 text-sm">Through
              <input className="block bg-slate-800 rounded-lg p-3 mt-1 text-white" type="date" required value={end} onChange={e => setEnd(e.target.value)} />
            </label>
            <button disabled={isFetching} className="bg-sky-600 text-white px-5 py-3 rounded-lg disabled:opacity-50">{isFetching ? "Loading records…" : "Load records"}</button>
          </form>
          <a className="text-sky-300 text-sm inline-block mt-4" href="https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt" target="_blank" rel="noreferrer">Find station IDs in NOAA’s global station catalog ↗</a>
        </GlassCard>
        {isError && <p role="alert" className="text-amber-300 mb-6">Station data could not be loaded. Check the station ID and date range, then try again.</p>}
        {data && <GlassCard>
          <h2 className="text-xl font-semibold text-white">{data.station_name} · {data.station_id}</h2>
          <p className="text-white/60 text-sm mt-2">{data.latitude}°, {data.longitude}° · Station reporting days</p>
          <p className="text-white/60 text-sm mt-2">Temperature means use available observations; counts show completeness. Rainfall totals appear only for complete calendar years. Missing or rejected values remain unavailable.</p>
          <div className="overflow-x-auto mt-6">
            <table className="w-full text-left text-sm text-white/80">
              <thead><tr className="border-b border-white/20"><th className="p-3">Year</th><th>Mean daily high °C</th><th>Mean daily low °C</th><th>Rainfall mm</th></tr></thead>
              <tbody>{summary.map(row => <tr key={row.year} className="border-b border-white/10">
                <td className="p-3">{row.year}</td>
                <td>{row.high} <span className="text-white/40">({row.highCount}/{row.days})</span></td>
                <td>{row.low} <span className="text-white/40">({row.lowCount}/{row.days})</span></td>
                <td>{row.rain} <span className="text-white/40">({row.rainCount}/{row.days})</span></td>
              </tr>)}</tbody>
            </table>
          </div>
          {!summary.length && <p className="text-white/70 mt-4">No station records are available for this period.</p>}
          <a href={data.source_url} className="text-sky-300 text-sm mt-5 inline-block">Original NOAA station file ↗</a>
          <p className="text-xs text-white/40 mt-2 break-all">Source SHA256: {data.source_sha256}</p>
        </GlassCard>}
      </div>
    </main>
  );
}
