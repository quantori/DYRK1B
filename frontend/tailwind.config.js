module.exports = {
  content: [
    './public/index.html',
    './src/**/*.tsx',
  ],
  darkMode: 'class',
  theme: {
    extend: {},
    colors: {
      success: '#00b04c',
      danger: '#ff5623',
      warning: '#feb31e',
      transparent: 'transparent',
      current: 'currentColor',
      white: '#ffffff',
      // https://tailwindcss.com/docs/customizing-colors#color-object-syntax
      primary: {
        DEFAULT: '#2977ff',
      },
      secondary: {
        DEFAULT: '#6b52fc',
      },
      dark: {
        DEFAULT: '#173348',
        500: '#667b93',
      },
    },
    fontFamily: {
      sans: ['"Roboto"', 'sans-serif'],
    },
  },
  plugins: [],
}
