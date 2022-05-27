/* eslint-disable @typescript-eslint/no-var-requires, import/no-extraneous-dependencies */
const path = require('path')
const webpack = require('webpack')
const HtmlWebpackPlugin = require('html-webpack-plugin')
const ForkTsCheckerWebpackPlugin = require('fork-ts-checker-webpack-plugin')
const StyleLintPlugin = require('stylelint-webpack-plugin')
const MiniCssExtractPlugin = require('mini-css-extract-plugin')
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin')
const TsconfigPathsPlugin = require('tsconfig-paths-webpack-plugin')
const CopyPlugin = require('copy-webpack-plugin')

// webpack
const lessOptionsModifyVars = require('./webpack/lessOptionsModifyVars.js')

module.exports = (webpackEnv, { mode }) => {
  const isEnvDevelopment = mode === 'development'
  const isEnvProduction = mode === 'production'

  const config = {
    mode: isEnvProduction ? 'production' : isEnvDevelopment && 'development',
    devtool: isEnvProduction ? 'hidden-source-map' : 'source-map',
    devServer: {
      open: true,
      compress: true,
      historyApiFallback: true,
      client: {
        logging: 'warn',
        overlay: { errors: true, warnings: true },
      },
    },
    entry: {
      main: path.resolve(__dirname, './src/index.tsx'),
    },
    output: {
      path: path.resolve(__dirname, './build'),
      clean: true,
      assetModuleFilename: 'static/media/[hash][ext]',
      filename: isEnvProduction
        ? 'static/js/[name].[contenthash:8].js'
        : isEnvDevelopment && 'static/js/[name].js',
    },
    optimization: {
      minimize: isEnvProduction,
      minimizer: [
        new CssMinimizerPlugin(),
      ],
      splitChunks: {
        chunks: 'all',
      },
    },
    resolve: {
      extensions: ['.tsx', '.ts', '.js'],
      plugins: [new TsconfigPathsPlugin({
        extensions: ['.ts', '.tsx', '.css', '.less', '.png', '.jpg', '.jpeg', '.svg'],
      })],
    },
    module: {
      rules: [
        {
          test: /\.(ts|js)x?$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: [
                '@babel/preset-env',
                '@babel/preset-react',
                '@babel/preset-typescript',
              ],
            },
          },
        },
        {
          test: /\.css$/,
          use: [
            isEnvDevelopment ? 'style-loader' : MiniCssExtractPlugin.loader,
            'css-loader',
            'postcss-loader',
          ],
        },
        {
          test: /\.less$/,
          use: [
            isEnvDevelopment ? 'style-loader' : MiniCssExtractPlugin.loader,
            'css-loader',
            'postcss-loader',
            {
              loader: 'less-loader',
              options: {
                lessOptions: {
                  modifyVars: lessOptionsModifyVars,
                  javascriptEnabled: true,
                },
              },
            },
          ],
        },
        {
          test: /\.(svg|gif|png|jpg|jpeg)$/,
          type: 'asset/resource',
        },
      ],
    },
    plugins: [
      new webpack.ProgressPlugin(),
      new HtmlWebpackPlugin({
        template: path.join(__dirname, './public/index.html'),
      }),
      new CopyPlugin({
        patterns: [
          {
            from: 'public',
            globOptions: {
              ignore: ['**/index.html'],
            },
          },
        ],
      }),
      new webpack.ProvidePlugin({
        React: 'react',
      }),
      new ForkTsCheckerWebpackPlugin({
        async: false,
      }),
      new StyleLintPlugin({
        configFile: path.resolve(__dirname, './stylelint.config.js'),
        context: path.resolve(__dirname, './src'),
        files: '**/*.(css|less)',
      }),
    ].concat(isEnvDevelopment ? [] : [
      new MiniCssExtractPlugin({
        filename: 'static/css/[name].[contenthash:8].css',
      }),
    ]),
  }

  return config
}
