/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        vision: {
          bg: "#0a0a12",
          panel: "#12121d",
          border: "#242438",
          accent: "#6366f1",
          accent2: "#22d3ee",
          muted: "#8b8ba7",
        },
      },
      animation: {
        "pulse-slow": "pulse 2.5s cubic-bezier(0.4,0,0.6,1) infinite",
      },
    },
  },
  plugins: [],
};
