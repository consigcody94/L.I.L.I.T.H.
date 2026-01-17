"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface Location {
  latitude: number;
  longitude: number;
  name: string;
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

interface ForecastData {
  location: Location;
  generated_at: string;
  model_version: string;
  forecast_days: number;
  forecasts: DailyForecast[];
}

interface WeatherState {
  // Current location
  location: Location;
  setLocation: (location: Location) => void;

  // Saved locations
  savedLocations: Location[];
  addSavedLocation: (location: Location) => void;
  removeSavedLocation: (latitude: number, longitude: number) => void;

  // Forecast cache
  forecastCache: Record<string, ForecastData>;
  setForecast: (key: string, forecast: ForecastData) => void;
  getForecast: (key: string) => ForecastData | undefined;

  // UI preferences
  temperatureUnit: "C" | "F";
  setTemperatureUnit: (unit: "C" | "F") => void;

  showUncertainty: boolean;
  setShowUncertainty: (show: boolean) => void;

  theme: "auto" | "light" | "dark";
  setTheme: (theme: "auto" | "light" | "dark") => void;

  // Selected date for detailed view
  selectedDate: string | null;
  setSelectedDate: (date: string | null) => void;
}

const generateLocationKey = (lat: number, lon: number): string => {
  return `${lat.toFixed(2)},${lon.toFixed(2)}`;
};

export const useWeatherStore = create<WeatherState>()(
  persist(
    (set, get) => ({
      // Location
      location: {
        latitude: 40.7128,
        longitude: -74.006,
        name: "New York, NY",
      },
      setLocation: (location) => set({ location }),

      // Saved locations
      savedLocations: [],
      addSavedLocation: (location) =>
        set((state) => {
          // Don't add duplicates
          const exists = state.savedLocations.some(
            (l) =>
              l.latitude === location.latitude &&
              l.longitude === location.longitude
          );
          if (exists) return state;

          return {
            savedLocations: [...state.savedLocations, location].slice(0, 10), // Max 10
          };
        }),
      removeSavedLocation: (latitude, longitude) =>
        set((state) => ({
          savedLocations: state.savedLocations.filter(
            (l) => l.latitude !== latitude || l.longitude !== longitude
          ),
        })),

      // Forecast cache
      forecastCache: {},
      setForecast: (key, forecast) =>
        set((state) => ({
          forecastCache: {
            ...state.forecastCache,
            [key]: forecast,
          },
        })),
      getForecast: (key) => get().forecastCache[key],

      // UI preferences
      temperatureUnit: "C",
      setTemperatureUnit: (unit) => set({ temperatureUnit: unit }),

      showUncertainty: true,
      setShowUncertainty: (show) => set({ showUncertainty: show }),

      theme: "dark",
      setTheme: (theme) => set({ theme }),

      // Selected date
      selectedDate: null,
      setSelectedDate: (date) => set({ selectedDate: date }),
    }),
    {
      name: "lilith-weather-storage",
      partialize: (state) => ({
        savedLocations: state.savedLocations,
        temperatureUnit: state.temperatureUnit,
        showUncertainty: state.showUncertainty,
        theme: state.theme,
      }),
    }
  )
);

// Selectors
export const useCurrentLocation = () =>
  useWeatherStore((state) => state.location);

export const useSavedLocations = () =>
  useWeatherStore((state) => state.savedLocations);

export const useTemperatureUnit = () =>
  useWeatherStore((state) => state.temperatureUnit);

export const useShowUncertainty = () =>
  useWeatherStore((state) => state.showUncertainty);
