"use client";

import { motion } from "framer-motion";
import { format, parseISO } from "date-fns";
import type { AccuracyReportResponse, PredictionRecord } from "@/hooks/useForecast";

interface AccuracyTrackerProps {
  report: AccuracyReportResponse | undefined;
  isLoading: boolean;
  unit?: "C" | "F";
}

// Convert Celsius to Fahrenheit
function toFahrenheit(celsius: number): number {
  return celsius * 9 / 5 + 32;
}

// Convert temperature based on unit
function convertTemp(celsius: number, unit: "C" | "F"): number {
  return unit === "F" ? toFahrenheit(celsius) : celsius;
}

// Get accuracy color
function getAccuracyColor(mae: number): string {
  if (mae < 1.5) return "text-green-400";
  if (mae < 2.5) return "text-yellow-400";
  if (mae < 4) return "text-orange-400";
  return "text-red-400";
}

// Get error color
function getErrorColor(error: number | null): string {
  if (error === null) return "text-white/50";
  const absError = Math.abs(error);
  if (absError < 1) return "text-green-400";
  if (absError < 2) return "text-yellow-400";
  if (absError < 3) return "text-orange-400";
  return "text-red-400";
}

// Stat card component
function StatCard({ label, value, subValue, color }: { label: string; value: string; subValue?: string; color?: string }) {
  return (
    <div className="bg-white/5 rounded-lg p-4 text-center">
      <p className="text-xs text-white/50 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color || "text-white"}`}>{value}</p>
      {subValue && <p className="text-xs text-white/40 mt-1">{subValue}</p>}
    </div>
  );
}

// Prediction row component
function PredictionRow({ prediction, unit }: { prediction: PredictionRecord; unit: "C" | "F" }) {
  const isVerified = prediction.actual_temp_max !== null;
  const predictedMax = convertTemp(prediction.predicted_temp_max, unit);
  const predictedMin = convertTemp(prediction.predicted_temp_min, unit);
  const actualMax = prediction.actual_temp_max !== null ? convertTemp(prediction.actual_temp_max, unit) : null;
  const actualMin = prediction.actual_temp_min !== null ? convertTemp(prediction.actual_temp_min, unit) : null;

  // Convert error to the right unit scale
  const tempMaxError = prediction.temp_max_error !== null ? (unit === "F" ? prediction.temp_max_error * 9/5 : prediction.temp_max_error) : null;
  const tempMinError = prediction.temp_min_error !== null ? (unit === "F" ? prediction.temp_min_error * 9/5 : prediction.temp_min_error) : null;

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`grid grid-cols-7 gap-2 py-3 px-4 text-sm border-b border-white/5 hover:bg-white/5 transition-colors ${
        isVerified ? "" : "opacity-60"
      }`}
    >
      {/* Date */}
      <div className="text-white/80">
        {format(parseISO(prediction.target_date), "MMM d")}
      </div>

      {/* Lead days */}
      <div className="text-white/50 text-center">
        {prediction.lead_days}d
      </div>

      {/* Predicted */}
      <div className="text-center">
        <span className="text-red-400">{Math.round(predictedMax)}°</span>
        <span className="text-white/30 mx-1">/</span>
        <span className="text-blue-400">{Math.round(predictedMin)}°</span>
      </div>

      {/* Actual */}
      <div className="text-center">
        {isVerified ? (
          <>
            <span className="text-red-300">{actualMax !== null ? Math.round(actualMax) : "-"}°</span>
            <span className="text-white/30 mx-1">/</span>
            <span className="text-blue-300">{actualMin !== null ? Math.round(actualMin) : "-"}°</span>
          </>
        ) : (
          <span className="text-white/30">Pending...</span>
        )}
      </div>

      {/* Max Temp Error */}
      <div className={`text-center ${getErrorColor(tempMaxError)}`}>
        {tempMaxError !== null ? (
          <>
            {tempMaxError > 0 ? "+" : ""}
            {tempMaxError.toFixed(1)}°
          </>
        ) : (
          "-"
        )}
      </div>

      {/* Min Temp Error */}
      <div className={`text-center ${getErrorColor(tempMinError)}`}>
        {tempMinError !== null ? (
          <>
            {tempMinError > 0 ? "+" : ""}
            {tempMinError.toFixed(1)}°
          </>
        ) : (
          "-"
        )}
      </div>

      {/* Status */}
      <div className="text-center">
        {isVerified ? (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-green-500/20 text-green-400">
            Verified
          </span>
        ) : (
          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-yellow-500/20 text-yellow-400">
            Awaiting
          </span>
        )}
      </div>
    </motion.div>
  );
}

export function AccuracyTracker({ report, isLoading, unit = "C" }: AccuracyTrackerProps) {
  if (isLoading) {
    return (
      <div className="animate-pulse space-y-4">
        <div className="grid grid-cols-4 gap-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-20 bg-white/10 rounded-lg" />
          ))}
        </div>
        <div className="h-64 bg-white/10 rounded-lg" />
      </div>
    );
  }

  if (!report) {
    return (
      <div className="text-center py-8 text-white/50">
        <p>No accuracy data available yet.</p>
        <p className="text-sm mt-2">Predictions will be tracked and compared to labeled weather reference data.</p>
      </div>
    );
  }

  const { stats, recent_predictions } = report;
  const hasVerified = stats.verified_predictions > 0;

  // Convert MAE values for display based on unit
  const displayMaxMAE = unit === "F" ? stats.temp_max_mae * 9/5 : stats.temp_max_mae;
  const displayMinMAE = unit === "F" ? stats.temp_min_mae * 9/5 : stats.temp_min_mae;
  const displayMaxRMSE = unit === "F" ? stats.temp_max_rmse * 9/5 : stats.temp_max_rmse;
  const displayMinRMSE = unit === "F" ? stats.temp_min_rmse * 9/5 : stats.temp_min_rmse;

  return (
    <div className="space-y-6">
      <div className="rounded-xl p-4 border border-purple-500/20 bg-purple-500/10">
        <p className="text-sm font-semibold text-white">Forecast archive and reference comparison</p>
        <p className="text-xs text-white/60 mt-1">
          These comparisons use Open-Meteo modeled weather data. Independent station verification
          is not available yet. Completed daily periods are checked as reference data arrives.
        </p>
      </div>

      {/* Stats overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard
          label="Predictions Tracked"
          value={stats.total_predictions.toString()}
          subValue={`${stats.verified_predictions} compared`}
        />
        <StatCard
          label="High Temperature MAE"
          value={hasVerified ? `${displayMaxMAE.toFixed(1)}°${unit}` : "N/A"}
          subValue={hasVerified ? `RMSE: ${displayMaxRMSE.toFixed(1)}°` : "No data yet"}
          color={hasVerified ? getAccuracyColor(stats.temp_max_mae) : undefined}
        />
        <StatCard
          label="Low Temperature MAE"
          value={hasVerified ? `${displayMinMAE.toFixed(1)}°${unit}` : "N/A"}
          subValue={hasVerified ? `RMSE: ${displayMinRMSE.toFixed(1)}°` : "No data yet"}
          color={hasVerified ? getAccuracyColor(stats.temp_min_mae) : undefined}
        />
        <StatCard
          label="Wet/Dry Agreement"
          value={hasVerified ? `${stats.precip_accuracy.toFixed(0)}%` : "N/A"}
          subValue={hasVerified ? `MAE: ${stats.precip_mae.toFixed(1)}mm` : "No data yet"}
          color={hasVerified && stats.precip_accuracy > 70 ? "text-green-400" : undefined}
        />
      </div>

      {/* Accuracy by lead time chart */}
      {hasVerified && Object.keys(stats.accuracy_by_lead_day).length > 0 && (
        <div className="bg-white/5 rounded-xl p-4">
          <h4 className="text-sm font-medium text-white/70 mb-4">Accuracy by Forecast Lead Time</h4>
          <div className="flex items-end gap-2 h-32">
            {Array.from({ length: 14 }, (_, i) => i + 1).map((day) => {
              const dayStats = stats.accuracy_by_lead_day[day];
              if (!dayStats) return null;

              const maxHeight = 100;
              const mae = dayStats.temp_max_mae;
              // Scale: 0 MAE = full height, 5+ MAE = minimal height
              const height = Math.max(10, maxHeight - (mae / 5) * maxHeight);

              return (
                <div key={day} className="flex-1 flex flex-col items-center">
                  <motion.div
                    initial={{ height: 0 }}
                    animate={{ height }}
                    transition={{ delay: day * 0.05 }}
                    className={`w-full rounded-t ${getAccuracyColor(mae).replace("text-", "bg-")} bg-opacity-60`}
                    title={`Day ${day}: ±${mae.toFixed(1)}°C (${dayStats.count} predictions)`}
                  />
                  <span className="text-xs text-white/40 mt-1">{day}d</span>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-white/40 text-center mt-2">
            Taller bars = better accuracy (lower error)
          </p>
        </div>
      )}

      {/* Recent predictions table */}
      <div className="bg-white/5 rounded-xl overflow-hidden">
        <div className="px-4 py-3 border-b border-white/10">
          <h4 className="text-sm font-medium text-white/70">Archived Predictions vs Reference</h4>
        </div>

        {/* Table header */}
        <div className="grid grid-cols-7 gap-2 py-2 px-4 text-xs text-white/50 border-b border-white/10 bg-white/5">
          <div>Date</div>
          <div className="text-center">Lead</div>
          <div className="text-center">Predicted</div>
          <div className="text-center">Reference</div>
          <div className="text-center">High Err</div>
          <div className="text-center">Low Err</div>
          <div className="text-center">Status</div>
        </div>

        {/* Table body */}
        <div className="max-h-80 overflow-y-auto">
          {recent_predictions.length > 0 ? (
            recent_predictions.map((prediction) => (
              <PredictionRow key={prediction.id} prediction={prediction} unit={unit} />
            ))
          ) : (
            <div className="py-8 text-center text-white/40">
              <p>No predictions recorded yet.</p>
              <p className="text-sm mt-1">Forecasts will appear here as they are made.</p>
            </div>
          )}
        </div>
      </div>

      {/* Info footer */}
      <div className="text-xs text-white/40 text-center">
        <p>
          MAE = Mean Absolute Error | RMSE = Root Mean Square Error
        </p>
        <p className="mt-1">
          Model-reference agreement does not measure independent station accuracy. Missing comparisons remain pending.
        </p>
      </div>
    </div>
  );
}
