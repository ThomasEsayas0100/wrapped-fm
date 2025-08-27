/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      keyframes: {
        gradient: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)', opacity: '0.5' },
          '50%': { transform: 'translateY(-40px)', opacity: '0.2' },
        },
      },
      animation: {
        gradient: 'gradient 15s ease infinite',
        float: 'float 10s ease-in-out infinite',
      },
    },
  },
  plugins: [],
};
