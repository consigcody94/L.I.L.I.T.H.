"use client";

import { useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from "recharts";

interface Location {
  latitude: number;
  longitude: number;
  name: string;
}

interface DateRange {
  start: string;
  end: string;
}

interface ClimateChartProps {
  location: Location;
  dateRange: DateRange;
  variable: "temperature" | "precipitation";
}

export function ClimateChart({ location, dateRange, variable }: ClimateChartProps) {
  // Generate mock historical data
  // In production, this would fetch from the API
  const data = useMemo(() => {
    const startYear = parseInt(dateRange.start.split("-")[0]);
    const endYear = parseInt(dateRange.end.split("-")[0]);

    const years = [];
    for (let year = startYear; year <= endYear; year++) {
      if (variable === "temperature") {
        // Simulated warming trend
        const baseline = 12 + (year - 1950) * 0.02;
        const variation = Math.sin(year * 0.5) * 0.5;

        years.push({
          year,
          annual: baseline + variation + Math.random() * 0.5,
          summer: baseline + 10 + variation + Math.random() * 0.5,
          winter: baseline - 8 + variation + Math.random() * 0.5,
        });
      } else {
        // Simulated precipitation variability
        const baseline = 1000;
        const variation = Math.sin(year * 0.3) * 100;

        years.push({
          year,
          annual: baseline + variation + Math.random() * 200 - 100,
        });
      }
    }

    return years;
  }, [dateRange, variable]);

  // Calculate trend line
  const trendLine = useMemo(() => {
    if (data.length < 2) return null;

    const n = data.length;
    const sumX = data.reduce((sum, d) => sum + d.year, 0);
    const sumY = data.reduce((sum, d) => sum + d.annual, 0);
    const sumXY = data.reduce((sum, d) => sum + d.year * d.annual, 0);
    const sumX2 = data.reduce((sum, d) => sum + d.year * d.year, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return {
      slope,
      intercept,
      start: { year: data[0].year, value: slope * data[0].year + intercept },
      end: { year: data[n - 1].year, value: slope * data[n - 1].year + intercept },
    };
  }, [data]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const d = payload[0].payload;
      return (
        <div className="bg-slate-800/90 backdrop-blur-sm border border-white/20 rounded-lg p-3 shadow-xl">
          <p className="text-white font-medium mb-2">{d.year}</p>
          {variable === "temperature" ? (
            <div className="space-y-1">
              <p className="text-white/80">
                Annual: {d.annual.toFixed(1)}°C
              </p>
              <p className="text-red-400">
                Summer: {d.summer.toFixed(1)}°C
              </p>
              <p className="text-blue-400">
                Winter: {d.winter.toFixed(1)}°C
              </p>
            </div>
          ) : (
            <p className="text-blue-400">
              Total: {d.annual.toFixed(0)}mm
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />

          <XAxis
            dataKey="year"
            stroke="rgba(255,255,255,0.5)"
            fontSize={12}
            tickLine={false}
          />

          <YAxis
            stroke="rgba(255,255,255,0.5)"
            fontSize={12}
            tickLine={false}
            tickFormatter={(value) =>
              variable === "temperature" ? `${value}°` : `${value}mm`
            }
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            formatter={(value) => (
              <span className="text-white/70 text-sm">{value}</span>
            )}
          />

          {variable === "temperature" ? (
            <>
              <Line
                type="monotone"
                dataKey="annual"
                name="Annual Avg"
                stroke="#94a3b8"
                strokeWidth={2}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="summer"
                name="Summer"
                stroke="#ef4444"
                strokeWidth={1}
                dot={false}
                strokeDasharray="4 4"
              />
              <Line
                type="monotone"
                dataKey="winter"
                name="Winter"
                stroke="#3b82f6"
                strokeWidth={1}
                dot={false}
                strokeDasharray="4 4"
              />
            </>
          ) : (
            <Line
              type="monotone"
              dataKey="annual"
              name="Annual Total"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
            />
          )}

          {/* Trend line visualization */}
          {trendLine && (
            <ReferenceLine
              segment={[
                { x: trendLine.start.year, y: trendLine.start.value },
                { x: trendLine.end.year, y: trendLine.end.value },
              ]}
              stroke="#22c55e"
              strokeWidth={2}
              strokeDasharray="8 4"
            />
          )}
        </LineChart>
      </ResponsiveContainer>

      {/* Trend summary */}
      {trendLine && variable === "temperature" && (
        <div className="mt-2 text-sm text-white/60">
          Trend: {trendLine.slope > 0 ? "+" : ""}
          {(trendLine.slope * 10).toFixed(2)}°C per decade
          {trendLine.slope > 0 && (
            <span className="text-amber-400 ml-2">↑ Warming</span>
          )}
        </div>
      )}
    </div>
  );
}
