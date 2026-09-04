/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{ts,tsx,html}"],
  theme: {
    extend: {
      colors: {
        safe: { DEFAULT: "#16a34a", foreground: "#052e16" },
        warn: { DEFAULT: "#d97706", foreground: "#431407" },
        danger: { DEFAULT: "#dc2626", foreground: "#450a0a" },
      },
      keyframes: {
        slidein: {
          "0%": { transform: "translateY(12px)", opacity: 0 },
          "100%": { transform: "translateY(0)", opacity: 1 },
        },
      },
      animation: { slidein: "slidein 180ms ease-out" },
    },
  },
  plugins: [],
};
