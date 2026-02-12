"use client";

import { motion } from "framer-motion";
import { GlassCard } from "./GlassCard";

export function Hero() {
    return (
        <div className="relative min-h-[60vh] flex items-center justify-center p-6">
            {/* Background Ambience */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-[100px] animate-pulse-slow" />
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-[100px] animate-pulse-slow" style={{ animationDelay: "1s" }} />
            </div>

            <div className="relative z-10 max-w-4xl w-full text-center">
                <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                    className="mb-8"
                >
                    <span className="inline-block py-1 px-3 rounded-full bg-white/10 border border-white/20 text-sm font-medium text-cyan-300 mb-6 backdrop-blur-md">
                        The Future of Weather Forecasting
                    </span>
                    <h1 className="text-5xl md:text-7xl font-bold bg-gradient-to-br from-white via-white to-white/50 bg-clip-text text-transparent mb-6 tracking-tight">
                        See the Weather <br />
                        <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400">
                            90 Days Ahead
                        </span>
                    </h1>
                    <p className="text-lg md:text-xl text-white/60 max-w-2xl mx-auto leading-relaxed">
                        LILITH uses advanced machine learning to predict long-range climate trends with unprecedented accuracy.
                        Open source, free, and powered by global historical data.
                    </p>
                </motion.div>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3, duration: 0.6 }}
                    className="flex flex-col sm:flex-row items-center justify-center gap-4"
                >
                    <button
                        onClick={() => document.getElementById('forecast-section')?.scrollIntoView({ behavior: 'smooth' })}
                        className="px-8 py-4 bg-white text-slate-900 rounded-xl font-bold hover:bg-white/90 transition-colors shadow-[0_0_20px_rgba(255,255,255,0.3)]"
                    >
                        View Forecast
                    </button>
                    <a
                        href="https://github.com/consigcody94/L.I.L.I.T.H."
                        target="_blank"
                        rel="noopener noreferrer"
                        className="px-8 py-4 bg-white/5 border border-white/10 rounded-xl font-bold text-white hover:bg-white/10 transition-all hover:border-white/20"
                    >
                        Explore Code
                    </a>
                </motion.div>

                {/* Stats Grid */}
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6, duration: 0.8 }}
                    className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-16 text-left"
                >
                    {[
                        { label: "Forecast Range", value: "90 Days" },
                        { label: "Global Stations", value: "100k+" },
                        { label: "Model Architecture", value: "Transformer" },
                        { label: "License", value: "Apache 2.0" }
                    ].map((stat, i) => (
                        <GlassCard key={i} className="p-4" variant="default">
                            <div className="text-sm text-white/40 mb-1">{stat.label}</div>
                            <div className="text-xl font-semibold text-white">{stat.value}</div>
                        </GlassCard>
                    ))}
                </motion.div>
            </div>
        </div>
    );
}
