'use strict'

let {float32, fract32} = require('./')
let a = require('assert')

let data = [0, 0.1, 0.2, 0.5, 1, 1.5]

a.deepEqual(float32(data), new Float32Array(data))
a.deepEqual(fract32(data), new Float32Array([ 0, -1.4901161415892261e-9,
  -2.9802322831784522e-9, 0, 0, 0 ]))
a.deepEqual(float32(.1), 0.10000000149011612);
a.deepEqual(fract32([.1]), new Float32Array([-1.4901161415892261e-9]))

let data_to_32 = float32(data);
a.deepEqual(fract32(data, data_to_32), new Float32Array([ 0, -1.4901161415892261e-9,
  -2.9802322831784522e-9, 0, 0, 0 ]))
