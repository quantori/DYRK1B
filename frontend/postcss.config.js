/* eslint-disable @typescript-eslint/no-var-requires, import/no-extraneous-dependencies */
const presetEnv = require('postcss-preset-env')
const tailwindcss = require('tailwindcss')
const autoprefixer = require('autoprefixer')

module.exports = {
  plugins: [
    tailwindcss,
    autoprefixer,
    presetEnv({
      browsers: '>0.2%, not dead, not IE 11, not op_mini all',
    }),
  ],
}
