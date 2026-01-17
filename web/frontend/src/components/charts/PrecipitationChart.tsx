"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";
import { format, parseISO } from "date-fns";

interface PrecipitationData {
  date: string;
  precipitation: number;
  precipitation_probability: number;
}

interface PrecipitationChartProps {
  data: PrecipitationData[];
}

function getPrecipColor(prob: number): string {
  if (prob < 0.2) return "#94a3b8"; // slate-400
  if (prob < 0.4) return "#60a5fa"; // blue-400
  if (prob < 0.6) return "#3b82f6"; // blue-500
  if (prob < 0.8) return "#2563eb"; // blue-600
  return "#1d4ed8"; // blue-700
}

export function PrecipitationChart({ data }: PrecipitationChartProps) {
  const chartData = data.map((d) => ({
    date: d.date,
    dateDisplay: format(parseISO(d.date), "MMM d"),
    precipitation: d.precipitation,
    probability: d.precipitation_probability,
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-800/90 backdrop-blur-sm border border-white/20 rounded-lg p-3 shadow-xl">
          <p className="text-white font-medium mb-2">
            {format(parseISO(data.date), "EEEE, MMM d")}
          </p>
          <div className="space-y-1">
            <p className="text-blue-400">
              Precipitation: {data.precipitation.toFixed(1)} mm
            </p>
            <p className="text-white/70">
              Probability: {Math.round(data.probability * 100)}%
            </p>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
        >
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
            tickFormatter={(value) => `${value}mm`}
          />

          <Tooltip content={<CustomTooltip />} />

          <Bar dataKey="precipitation" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={getPrecipColor(entry.probability)}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
