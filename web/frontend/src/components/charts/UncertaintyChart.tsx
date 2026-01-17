"use client";

import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
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

interface UncertaintyChartProps {
  data: ForecastData[];
  variable: "max" | "min";
}

export function UncertaintyChart({ data, variable }: UncertaintyChartProps) {
  const chartData = data.map((d) => {
    if (variable === "max") {
      return {
        date: d.date,
        dateDisplay: format(parseISO(d.date), "MMM d"),
        value: d.temperature_max,
        lower: d.temperature_max_lower ?? d.temperature_max - 2,
        upper: d.temperature_max_upper ?? d.temperature_max + 2,
        range: [
          d.temperature_max_lower ?? d.temperature_max - 2,
          d.temperature_max_upper ?? d.temperature_max + 2,
        ],
      };
    } else {
      return {
        date: d.date,
        dateDisplay: format(parseISO(d.date), "MMM d"),
        value: d.temperature_min,
        lower: d.temperature_min_lower ?? d.temperature_min - 2,
        upper: d.temperature_min_upper ?? d.temperature_min + 2,
        range: [
          d.temperature_min_lower ?? d.temperature_min - 2,
          d.temperature_min_upper ?? d.temperature_min + 2,
        ],
      };
    }
  });

  const color = variable === "max" ? "#ef4444" : "#3b82f6";
  const label = variable === "max" ? "High Temperature" : "Low Temperature";

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-800/90 backdrop-blur-sm border border-white/20 rounded-lg p-3 shadow-xl">
          <p className="text-white font-medium mb-2">
            {format(parseISO(data.date), "EEEE, MMM d")}
          </p>
          <div className="space-y-1">
            <p style={{ color }}>
              {label}: {Math.round(data.value)}°C
            </p>
            <p className="text-white/60 text-sm">
              95% CI: {Math.round(data.lower)}° - {Math.round(data.upper)}°
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
        <ComposedChart
          data={chartData}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
          <defs>
            <linearGradient id={`gradient-${variable}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.2} />
              <stop offset="95%" stopColor={color} stopOpacity={0.05} />
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
            formatter={() => (
              <span className="text-white/70 text-sm">{label}</span>
            )}
          />

          {/* Uncertainty band */}
          <Area
            type="monotone"
            dataKey="range"
            stroke="transparent"
            fill={`url(#gradient-${variable})`}
            fillOpacity={1}
            name="95% Confidence"
          />

          {/* Mean line */}
          <Line
            type="monotone"
            dataKey="value"
            stroke={color}
            strokeWidth={2}
            dot={false}
            name={label}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
