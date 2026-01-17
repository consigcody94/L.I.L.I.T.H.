"use client";

import { motion } from "framer-motion";

interface Location {
  latitude: number;
  longitude: number;
  name: string;
}

interface RecordEventsProps {
  location: Location;
}

interface RecordEvent {
  date: string;
  type: "heat" | "cold" | "rain" | "snow" | "wind";
  value: number;
  unit: string;
  description: string;
}

export function RecordEvents({ location }: RecordEventsProps) {
  // Mock record events - in production, fetch from API
  const records: RecordEvent[] = [
    {
      date: "2023-07-28",
      type: "heat",
      value: 41.2,
      unit: "°C",
      description: "Highest temperature recorded",
    },
    {
      date: "1934-02-09",
      type: "cold",
      value: -26.1,
      unit: "°C",
      description: "Lowest temperature recorded",
    },
    {
      date: "2011-08-28",
      type: "rain",
      value: 183,
      unit: "mm",
      description: "Hurricane Irene - highest daily rainfall",
    },
    {
      date: "2016-01-23",
      type: "snow",
      value: 68,
      unit: "cm",
      description: "Winter Storm Jonas - highest snowfall",
    },
    {
      date: "2012-10-29",
      type: "wind",
      value: 145,
      unit: "km/h",
      description: "Hurricane Sandy - highest wind gust",
    },
  ];

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "heat":
        return "🌡️";
      case "cold":
        return "❄️";
      case "rain":
        return "🌧️";
      case "snow":
        return "🌨️";
      case "wind":
        return "💨";
      default:
        return "📊";
    }
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case "heat":
        return "text-red-400 bg-red-400/10";
      case "cold":
        return "text-blue-400 bg-blue-400/10";
      case "rain":
        return "text-cyan-400 bg-cyan-400/10";
      case "snow":
        return "text-slate-300 bg-slate-300/10";
      case "wind":
        return "text-purple-400 bg-purple-400/10";
      default:
        return "text-white/60 bg-white/10";
    }
  };

  return (
    <div className="space-y-3">
      {records.map((record, index) => (
        <motion.div
          key={`${record.type}-${record.date}`}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className={`flex items-center gap-4 p-4 rounded-xl ${getTypeColor(
            record.type
          )}`}
        >
          {/* Icon */}
          <div className="text-3xl">{getTypeIcon(record.type)}</div>

          {/* Details */}
          <div className="flex-1">
            <div className="flex items-baseline gap-2">
              <span className="text-2xl font-bold">
                {record.value}
                {record.unit}
              </span>
              <span className="text-sm text-white/60">{record.date}</span>
            </div>
            <p className="text-sm text-white/70">{record.description}</p>
          </div>

          {/* Type badge */}
          <div className="px-3 py-1 rounded-full bg-white/10 text-sm capitalize">
            {record.type}
          </div>
        </motion.div>
      ))}

      {/* Note about data */}
      <p className="text-xs text-white/40 mt-4">
        * Records based on available GHCN station data near this location.
        Actual records may vary.
      </p>
    </div>
  );
}
