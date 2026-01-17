"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

interface Location {
  latitude: number;
  longitude: number;
  name: string;
}

interface LocationSearchProps {
  onLocationSelect: (location: Location) => void;
}

interface NominatimResult {
  lat: string;
  lon: string;
  display_name: string;
  address?: {
    city?: string;
    town?: string;
    village?: string;
    state?: string;
    country?: string;
  };
}

export function LocationSearch({ onLocationSelect }: LocationSearchProps) {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<Location[]>([]);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  // Geocode using OpenStreetMap Nominatim (free, no API key required)
  const geocodeLocation = useCallback(async (searchQuery: string) => {
    if (searchQuery.length < 2) {
      setResults([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(
          searchQuery
        )}&limit=8&addressdetails=1`,
        {
          headers: {
            "User-Agent": "LILITH Weather App",
          },
        }
      );

      if (!response.ok) {
        throw new Error("Geocoding failed");
      }

      const data: NominatimResult[] = await response.json();

      const locations: Location[] = data.map((result) => {
        // Build a cleaner name from address components
        const parts: string[] = [];
        if (result.address?.city || result.address?.town || result.address?.village) {
          parts.push(result.address.city || result.address.town || result.address.village || "");
        }
        if (result.address?.state) {
          parts.push(result.address.state);
        }
        if (result.address?.country) {
          parts.push(result.address.country);
        }

        const name = parts.length > 0 ? parts.join(", ") : result.display_name.split(",").slice(0, 3).join(",");

        return {
          latitude: parseFloat(result.lat),
          longitude: parseFloat(result.lon),
          name: name,
        };
      });

      setResults(locations);
    } catch (err) {
      console.error("Geocoding error:", err);
      setError("Failed to search locations. Please try again.");
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);

    // Debounce the geocoding request
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    debounceRef.current = setTimeout(() => {
      geocodeLocation(value);
    }, 300);
  };

  const handleSelect = (location: Location) => {
    setQuery(location.name);
    setIsOpen(false);
    setResults([]);
    onLocationSelect(location);
  };

  const handleUseCurrentLocation = () => {
    if ("geolocation" in navigator) {
      setIsLoading(true);
      setError(null);
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const { latitude, longitude } = position.coords;

          // Reverse geocode to get location name
          try {
            const response = await fetch(
              `https://nominatim.openstreetmap.org/reverse?format=json&lat=${latitude}&lon=${longitude}&addressdetails=1`,
              {
                headers: {
                  "User-Agent": "LILITH Weather App",
                },
              }
            );

            if (response.ok) {
              const data = await response.json();
              const parts: string[] = [];
              if (data.address?.city || data.address?.town || data.address?.village) {
                parts.push(data.address.city || data.address.town || data.address.village);
              }
              if (data.address?.state) {
                parts.push(data.address.state);
              }

              const location: Location = {
                latitude,
                longitude,
                name: parts.length > 0 ? parts.join(", ") : "Current Location",
              };

              setQuery(location.name);
              setIsOpen(false);
              onLocationSelect(location);
            } else {
              throw new Error("Reverse geocoding failed");
            }
          } catch {
            // Fallback if reverse geocoding fails
            const location: Location = {
              latitude,
              longitude,
              name: "Current Location",
            };
            setQuery(location.name);
            setIsOpen(false);
            onLocationSelect(location);
          }
          setIsLoading(false);
        },
        (error) => {
          console.error("Geolocation error:", error);
          setError("Unable to get your location. Please enable location services.");
          setIsLoading(false);
        },
        {
          enableHighAccuracy: true,
          timeout: 10000,
        }
      );
    } else {
      setError("Geolocation is not supported by your browser.");
    }
  };

  // Clean up debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  return (
    <div className="relative">
      {/* Search input */}
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          placeholder="Search any city, address, or place..."
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
              disabled={isLoading}
              className="w-full px-4 py-3 flex items-center gap-3 text-left hover:bg-white/10 transition-colors border-b border-white/10 disabled:opacity-50"
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
              <span className="text-white">Use my current location</span>
            </button>

            {/* Error message */}
            {error && (
              <div className="px-4 py-2 text-red-400 text-sm bg-red-400/10">
                {error}
              </div>
            )}

            {/* Search results */}
            <div className="max-h-64 overflow-y-auto">
              {results.length > 0 ? (
                results.map((location, index) => (
                  <button
                    key={`${location.latitude}-${location.longitude}-${index}`}
                    onClick={() => handleSelect(location)}
                    className="w-full px-4 py-3 flex items-center gap-3 text-left hover:bg-white/10 transition-colors"
                  >
                    <svg
                      className="w-5 h-5 text-white/50 flex-shrink-0"
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
                    <div className="min-w-0">
                      <p className="text-white truncate">{location.name}</p>
                      <p className="text-xs text-white/50">
                        {location.latitude.toFixed(4)}°, {location.longitude.toFixed(4)}°
                      </p>
                    </div>
                  </button>
                ))
              ) : query.length >= 2 && !isLoading ? (
                <div className="px-4 py-3 text-white/50 text-center">
                  No locations found. Try a different search.
                </div>
              ) : query.length < 2 ? (
                <div className="px-4 py-3 text-white/50 text-center">
                  Type at least 2 characters to search...
                </div>
              ) : null}
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
