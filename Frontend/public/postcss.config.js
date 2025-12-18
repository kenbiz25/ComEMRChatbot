
/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  darkMode: ["class", '[data-theme="dark"]'],
  theme: {
    extend: {
      fontFamily: { sans: ['"Segoe UI"', 'system-ui', 'sans-serif'] },
      colors: {
        accent: "var(--accent)",
        bg: "var(--bg)",
        text: "var(--text)",
        border: "var(--border)",
        bubbleUser: "var(--bubble-user)",
        bubbleAssistant: "var(--bubble-assistant)",
      },
      boxShadow: { bubble: "0 6px 16px rgba(0,0,0,0.06)" },
      keyframes: {
        fadeIn: { "0%": { opacity: "0", transform: "translateY(4px)" }, "100%": { opacity: "1", transform: "translateY(0)" } }
      },
      animation: { fadeIn: "fadeIn 0.2s ease-in" },
    },
  },
  plugins: [
    require("@tailwindcss/forms"),
    require("@tailwindcss/typography"),
  ],
};
