"use client";

import { useQuery } from "@tanstack/react-query";
import axios from "axios";

interface Location {
  latitude: number;
  longitude: number;
  name?: string;
}

interface DailyForecast {
  date: string;
  temperature_max: number;
  temperature_min: number;
  precipitation: number;
  precipitation_probability: number;
  temperature_max_lower?: number;
  temperature_max_upper?: number;
  temperature_min_lower?: number;
  temperature_min_upper?: number;
}

interface ForecastResponse {
  location: {
    latitude: number;
    longitude: number;
  };
  generated_at: string;
  model_version: string;
  forecast_days: number;
  forecasts: DailyForecast[];
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchForecast(location: Location): Promise<ForecastResponse> {
  const response = await axios.post<ForecastResponse>(`${API_URL}/v1/forecast`, {
    latitude: location.latitude,
    longitude: location.longitude,
    days: 90,
    include_uncertainty: true,
  });

  return response.data;
}

export function useForecast(location: Location) {
  return useQuery({
    queryKey: ["forecast", location.latitude, location.longitude],
    queryFn: () => fetchForecast(location),
    staleTime: 30 * 60 * 1000, // 30 minutes
    gcTime: 60 * 60 * 1000, // 1 hour (formerly cacheTime)
    retry: 2,
    refetchOnWindowFocus: false,
  });
}

// Hook for batch forecasts
export function useBatchForecast(locations: Location[]) {
  return useQuery({
    queryKey: ["batch-forecast", locations.map((l) => `${l.latitude},${l.longitude}`).join("|")],
    queryFn: async () => {
      const response = await axios.post(`${API_URL}/v1/forecast/batch`, {
        locations: locations.map((l) => ({
          latitude: l.latitude,
          longitude: l.longitude,
        })),
        days: 90,
        include_uncertainty: true,
      });
      return response.data;
    },
    enabled: locations.length > 0,
    staleTime: 30 * 60 * 1000,
  });
}
