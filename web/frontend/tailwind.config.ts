import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        // Ocean theme
        ocean: {
          deep: "#0a1929",
          mid: "#0d2847",
          light: "#1a3a5c",
        },
        sky: {
          clear: "#87ceeb",
          dusk: "#ff7f50",
          night: "#1a1a2e",
        },
        // Temperature colors
        temp: {
          freezing: "#0000ff",
          cold: "#00aaff",
          cool: "#00ffaa",
          mild: "#aaff00",
          warm: "#ffaa00",
          hot: "#ff6600",
          extreme: "#ff0000",
        },
        // Glass effect
        glass: {
          light: "rgba(255, 255, 255, 0.1)",
          medium: "rgba(255, 255, 255, 0.15)",
          dark: "rgba(0, 0, 0, 0.2)",
        },
      },
      fontFamily: {
        display: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      fontSize: {
        hero: ["4.5rem", { lineHeight: "1.1" }],
        "temp-large": ["6rem", { lineHeight: "1" }],
      },
      backdropBlur: {
        glass: "20px",
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-weather":
          "linear-gradient(180deg, var(--tw-gradient-stops))",
      },
      animation: {
        "fade-in": "fadeIn 0.5s ease-out",
        "slide-up": "slideUp 0.5s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        rain: "rain 1s linear infinite",
        snow: "snow 3s linear infinite",
      },
      keyframes: {
        fadeIn: {
          "0%": { opacity: "0" },
          "100%": { opacity: "1" },
        },
        slideUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        rain: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
        snow: {
          "0%": { transform: "translateY(-10%) rotate(0deg)" },
          "100%": { transform: "translateY(100vh) rotate(360deg)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
