// eslint-disable-next-line @typescript-eslint/no-var-requires
const TailwindConfig = require('../tailwind.config.js')

const lessOptionsModifyVars = {
  // https://github.com/ant-design/ant-design/blob/master/components/style/themes/default.less
  // Replace just theme independent variables
  '@primary-color': TailwindConfig.theme.colors.primary.DEFAULT,
  '@success-color': TailwindConfig.theme.colors.success,
  '@text-color': TailwindConfig.theme.colors.dark[500],
  '@line-height-base': 1.5,
}

module.exports = lessOptionsModifyVars
