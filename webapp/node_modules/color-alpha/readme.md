# color-alpha [![stable](https://img.shields.io/badge/stability-stable-green.svg)](http://github.com/badges/stability-badges) [![build](https://travis-ci.org/colorjs/color-alpha.png)](https://travis-ci.org/colorjs/color-alpha) [![size](https://img.shields.io/bundlephobia/minzip/color-parse?label=size)](https://bundlephobia.com/result?p=color-alpha)

Change alpha of a color with mercy on _node_modules_.

[![npm install color-alpha](https://nodei.co/npm/color-alpha.png?mini=true)](https://npmjs.org/package/color-alpha/)

```js
import α from 'color-alpha'
α('white', .4); // 'rgba(255,255,255,.4)'
```

Input color is parsed with [color-parse](https://ghub.io/color-parse) package.
