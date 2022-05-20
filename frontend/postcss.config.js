/* eslint-disable @typescript-eslint/no-var-requires, import/no-extraneous-dependencies */
const presetEnv = require('postcss-preset-env')

module.exports = {
  plugins: [
    presetEnv({
      browsers: '>0.2%, not dead, not IE 11, not op_mini all',
    }),
  ],
}
