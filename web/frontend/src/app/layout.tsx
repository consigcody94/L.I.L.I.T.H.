import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "LILITH - 90-Day Weather Forecast",
  description:
    "Long-range Intelligent Learning for Integrated Trend Hindcasting - Free, open-source 90-day weather forecasts powered by machine learning.",
  keywords: [
    "weather forecast",
    "90-day forecast",
    "long-range weather",
    "machine learning weather",
    "open source weather",
  ],
  authors: [{ name: "LILITH Project" }],
  openGraph: {
    title: "LILITH - 90-Day Weather Forecast",
    description: "Free, open-source 90-day weather forecasts powered by ML",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-display antialiased`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
