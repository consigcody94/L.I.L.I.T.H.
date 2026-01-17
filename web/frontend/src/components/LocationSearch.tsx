"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Location {
  latitude: number;
  longitude: number;
  name: string;
}

interface LocationSearchProps {
  onLocationSelect: (location: Location) => void;
}

// Popular cities for quick selection
const popularLocations: Location[] = [
  { latitude: 40.7128, longitude: -74.006, name: "New York, NY" },
  { latitude: 51.5074, longitude: -0.1278, name: "London, UK" },
  { latitude: 35.6762, longitude: 139.6503, name: "Tokyo, Japan" },
  { latitude: 48.8566, longitude: 2.3522, name: "Paris, France" },
  { latitude: -33.8688, longitude: 151.2093, name: "Sydney, Australia" },
  { latitude: 55.7558, longitude: 37.6173, name: "Moscow, Russia" },
];

export function LocationSearch({ onLocationSelect }: LocationSearchProps) {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<Location[]>([]);

  const handleSearch = useCallback(async (searchQuery: string) => {
    if (searchQuery.length < 2) {
      setResults([]);
      return;
    }

    setIsLoading(true);

    // Filter popular locations by query
    const filtered = popularLocations.filter((loc) =>
      loc.name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    // In production, this would call a geocoding API
    // For now, use filtered popular locations
    setTimeout(() => {
      setResults(filtered);
      setIsLoading(false);
    }, 200);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
    handleSearch(value);
  };

  const handleSelect = (location: Location) => {
    setQuery(location.name);
    setIsOpen(false);
    onLocationSelect(location);
  };

  const handleUseCurrentLocation = () => {
    if ("geolocation" in navigator) {
      setIsLoading(true);
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const location: Location = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            name: "Current Location",
          };
          setQuery(location.name);
          setIsOpen(false);
          setIsLoading(false);
          onLocationSelect(location);
        },
        (error) => {
          console.error("Geolocation error:", error);
          setIsLoading(false);
        }
      );
    }
  };

  return (
    <div className="relative">
      {/* Search input */}
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          placeholder="Search for a location..."
          className="w-full px-4 py-3 pl-12 bg-white/10 backdrop-blur-xl border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:ring-2 focus:ring-sky-400/50 focus:border-sky-400/50 transition-all"
        />

        {/* Search icon */}
        <svg
          className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-white/50"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
          />
        </svg>

        {/* Loading indicator */}
        {isLoading && (
          <div className="absolute right-4 top-1/2 -translate-y-1/2">
            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
          </div>
        )}
      </div>

      {/* Dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="absolute z-50 w-full mt-2 bg-slate-800/95 backdrop-blur-xl border border-white/20 rounded-xl shadow-2xl overflow-hidden"
          >
            {/* Use current location button */}
            <button
              onClick={handleUseCurrentLocation}
              className="w-full px-4 py-3 flex items-center gap-3 text-left hover:bg-white/10 transition-colors border-b border-white/10"
            >
              <svg
                className="w-5 h-5 text-sky-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 11a3 3 0 11-6 0 3 3 0 016 0z"
                />
              </svg>
              <span className="text-white">Use current location</span>
            </button>

            {/* Search results or popular locations */}
            <div className="max-h-64 overflow-y-auto">
              {(results.length > 0 ? results : popularLocations).map(
                (location, index) => (
                  <button
                    key={`${location.latitude}-${location.longitude}`}
                    onClick={() => handleSelect(location)}
                    className="w-full px-4 py-3 flex items-center gap-3 text-left hover:bg-white/10 transition-colors"
                  >
                    <svg
                      className="w-5 h-5 text-white/50"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z"
                      />
                    </svg>
                    <div>
                      <p className="text-white">{location.name}</p>
                      <p className="text-xs text-white/50">
                        {location.latitude.toFixed(2)}°,{" "}
                        {location.longitude.toFixed(2)}°
                      </p>
                    </div>
                  </button>
                )
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Click outside to close */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </div>
  );
}
