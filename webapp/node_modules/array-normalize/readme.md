# array-normalize  [![experimental](https://img.shields.io/badge/stability-unstable-yellow.svg)](http://github.com/badges/stability-badges) [![Build Status](https://img.shields.io/travis/dfcreative/array-normalize.svg)](https://travis-ci.org/dfcreative/array-normalize)

Normalize array to unit length, that is 0..1 range. See [feature scaling](https://en.wikipedia.org/wiki/Feature_scaling).

[![npm install array-normalize](https://nodei.co/npm/array-normalize.png?mini=true)](https://npmjs.org/package/array-normalize/)

```js
const normalize = require('array-normalize')

normalize([0, 50, 100]) // [0, .5, 1]
normalize([0, 0, .1, .2, 1, 2], 2) // [0, 0, .1, .1, 1, 1]
normalize([0, .25, 1, .25], 2, [0, .5, 1, .5]) // [0, .5, 1, .5])
```

## API

### array = normalize(array, dimensions=1, bounds?)

Normalizes n-dimensional array in-place using `dimensions` as stride, ie. for 1d array the expected data layout is `[x, x, x, ...]` for 2d is `[x, y, x, y, ...]`, etc.

Every dimension is normalized independently, eg. 2d array is normalized to unit square `[0, 0, 1, 1]`.

Optional `bounds` box can predefine min/max to optimize calculations.
