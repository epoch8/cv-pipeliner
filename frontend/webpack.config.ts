import path from 'path';
import { Configuration } from 'webpack';
import { CleanWebpackPlugin } from 'clean-webpack-plugin';
import HtmlWebpackPlugin from 'html-webpack-plugin';
import MiniCssExtractPlugin from 'mini-css-extract-plugin';

const config: Configuration = {
  entry: {
    index: path.join(__dirname, './src/index.ts'),
  },
  output: {
    filename: 'bundle.[hash].js',
    path: path.join(__dirname, 'build'),
    sourceMapFilename: '[file].map.json',
  },
  mode: 'development',
  resolve: {
    extensions: ['.ts', '.js'],
  },
  module: {
    rules: [
      {
        loader: 'babel-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: [
          MiniCssExtractPlugin.loader,
          'css-loader',
          'postcss-loader'
        ],
      },
      {
        test: /\.html$/,
        loader: 'html-loader'
      }
    ],
  },
  plugins: [
    new CleanWebpackPlugin(),
    new HtmlWebpackPlugin({
      template: path.join(__dirname, 'public/index.html'),
    }),
    new MiniCssExtractPlugin({
      filename: '[name].[hash].css',
    }),
  ],
  devtool: 'source-map',
  devServer: {
    compress: true,
    contentBase: path.join(__dirname, 'public'),
    hot: true,
    port: 3000,
  },
};

module.exports = config;
