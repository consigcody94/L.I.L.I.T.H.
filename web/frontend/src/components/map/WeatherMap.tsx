"use client";

import { useEffect, useRef, useState } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";

interface WeatherMapProps {
  latitude: number;
  longitude: number;
  onLocationSelect?: (lat: number, lon: number) => void;
  showTemperatureLayer?: boolean;
}

export function WeatherMap({
  latitude,
  longitude,
  onLocationSelect,
  showTemperatureLayer = false,
}: WeatherMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const marker = useRef<mapboxgl.Marker | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    const token = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;
    if (!token) {
      console.warn("Mapbox token not configured");
      return;
    }

    mapboxgl.accessToken = token;

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: "mapbox://styles/mapbox/dark-v11",
      center: [longitude, latitude],
      zoom: 6,
      attributionControl: false,
    });

    // Add navigation controls
    map.current.addControl(
      new mapboxgl.NavigationControl({ showCompass: false }),
      "top-right"
    );

    // Add marker
    marker.current = new mapboxgl.Marker({
      color: "#3b82f6",
      draggable: !!onLocationSelect,
    })
      .setLngLat([longitude, latitude])
      .addTo(map.current);

    // Handle marker drag
    if (onLocationSelect) {
      marker.current.on("dragend", () => {
        const lngLat = marker.current?.getLngLat();
        if (lngLat) {
          onLocationSelect(lngLat.lat, lngLat.lng);
        }
      });

      // Handle map click
      map.current.on("click", (e) => {
        marker.current?.setLngLat(e.lngLat);
        onLocationSelect(e.lngLat.lat, e.lngLat.lng);
      });
    }

    map.current.on("load", () => {
      setIsLoaded(true);
    });

    return () => {
      map.current?.remove();
      map.current = null;
    };
  }, []);

  // Update marker position when props change
  useEffect(() => {
    if (marker.current && map.current) {
      marker.current.setLngLat([longitude, latitude]);
      map.current.flyTo({
        center: [longitude, latitude],
        duration: 1000,
      });
    }
  }, [latitude, longitude]);

  // Add temperature layer
  useEffect(() => {
    if (!map.current || !isLoaded || !showTemperatureLayer) return;

    // Add a simple temperature visualization
    // In production, this would load actual temperature data
    if (!map.current.getSource("temperature")) {
      map.current.addSource("temperature", {
        type: "geojson",
        data: {
          type: "FeatureCollection",
          features: [],
        },
      });

      map.current.addLayer({
        id: "temperature-heat",
        type: "heatmap",
        source: "temperature",
        paint: {
          "heatmap-weight": ["get", "temperature"],
          "heatmap-intensity": 0.5,
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0,
            "rgba(0, 0, 255, 0)",
            0.2,
            "rgb(0, 170, 255)",
            0.4,
            "rgb(0, 255, 170)",
            0.6,
            "rgb(170, 255, 0)",
            0.8,
            "rgb(255, 170, 0)",
            1,
            "rgb(255, 0, 0)",
          ],
          "heatmap-radius": 30,
          "heatmap-opacity": 0.6,
        },
      });
    }
  }, [isLoaded, showTemperatureLayer]);

  return (
    <div className="relative w-full h-full min-h-[300px] rounded-xl overflow-hidden">
      <div ref={mapContainer} className="absolute inset-0" />

      {/* Loading overlay */}
      {!isLoaded && (
        <div className="absolute inset-0 bg-slate-800 flex items-center justify-center">
          <div className="w-8 h-8 border-2 border-white/30 border-t-white rounded-full animate-spin" />
        </div>
      )}

      {/* Coordinates display */}
      <div className="absolute bottom-4 left-4 bg-black/50 backdrop-blur-sm px-3 py-1.5 rounded-lg text-sm text-white/80">
        {latitude.toFixed(4)}°, {longitude.toFixed(4)}°
      </div>
    </div>
  );
}
