"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import { format, parseISO } from "date-fns";

interface ForecastData {
  date: string;
  temperature_max: number;
  temperature_min: number;
  temperature_max_lower?: number;
  temperature_max_upper?: number;
  temperature_min_lower?: number;
  temperature_min_upper?: number;
}

interface ForecastChartProps {
  data: ForecastData[];
  showUncertainty?: boolean;
}

export function ForecastChart({ data, showUncertainty = true }: ForecastChartProps) {
  // Transform data for the chart
  const chartData = data.map((d) => ({
    date: d.date,
    dateDisplay: format(parseISO(d.date), "MMM d"),
    high: d.temperature_max,
    low: d.temperature_min,
    highUpper: d.temperature_max_upper,
    highLower: d.temperature_max_lower,
    lowUpper: d.temperature_min_upper,
    lowLower: d.temperature_min_lower,
  }));

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-800/90 backdrop-blur-sm border border-white/20 rounded-lg p-3 shadow-xl">
          <p className="text-white font-medium mb-2">
            {format(parseISO(data.date), "EEEE, MMM d")}
          </p>
          <div className="space-y-1">
            <p className="text-red-400">
              High: {Math.round(data.high)}°
              {data.highLower && data.highUpper && (
                <span className="text-white/50 text-sm ml-1">
                  ({Math.round(data.highLower)}° - {Math.round(data.highUpper)}°)
                </span>
              )}
            </p>
            <p className="text-blue-400">
              Low: {Math.round(data.low)}°
              {data.lowLower && data.lowUpper && (
                <span className="text-white/50 text-sm ml-1">
                  ({Math.round(data.lowLower)}° - {Math.round(data.lowUpper)}°)
                </span>
              )}
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={chartData}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
          <defs>
            {/* High temperature gradient */}
            <linearGradient id="highGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>

            {/* Low temperature gradient */}
            <linearGradient id="lowGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>

            {/* Uncertainty band gradient */}
            <linearGradient id="uncertaintyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.2} />
              <stop offset="95%" stopColor="#94a3b8" stopOpacity={0.05} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />

          <XAxis
            dataKey="dateDisplay"
            stroke="rgba(255,255,255,0.5)"
            fontSize={12}
            tickLine={false}
            interval={Math.floor(data.length / 10)}
          />

          <YAxis
            stroke="rgba(255,255,255,0.5)"
            fontSize={12}
            tickLine={false}
            tickFormatter={(value) => `${value}°`}
            domain={["auto", "auto"]}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            wrapperStyle={{ paddingTop: "10px" }}
            formatter={(value) => (
              <span className="text-white/70 text-sm">{value}</span>
            )}
          />

          {/* Uncertainty bands (if available) */}
          {showUncertainty && chartData[0]?.highUpper && (
            <>
              <Area
                type="monotone"
                dataKey="highUpper"
                stroke="transparent"
                fill="url(#uncertaintyGradient)"
                fillOpacity={1}
              />
              <Area
                type="monotone"
                dataKey="highLower"
                stroke="transparent"
                fill="white"
                fillOpacity={1}
              />
            </>
          )}

          {/* High temperature */}
          <Area
            type="monotone"
            dataKey="high"
            name="High"
            stroke="#ef4444"
            strokeWidth={2}
            fill="url(#highGradient)"
            fillOpacity={1}
          />

          {/* Low temperature */}
          <Area
            type="monotone"
            dataKey="low"
            name="Low"
            stroke="#3b82f6"
            strokeWidth={2}
            fill="url(#lowGradient)"
            fillOpacity={1}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
