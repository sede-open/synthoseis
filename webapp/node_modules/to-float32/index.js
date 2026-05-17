/* @module to-float32 */

'use strict'

module.exports = float32
module.exports.float32 =
module.exports.float = float32
module.exports.fract32 =
module.exports.fract = fract32

var narr = new Float32Array(1)

// Returns fractional part of float32 array
function fract32 (arr, fract) {
	if (arr.length) {
		if (arr instanceof Float32Array) return new Float32Array(arr.length);
		if (!(fract instanceof Float32Array)) fract = float32(arr)
		for (var i = 0, l = fract.length; i < l; i++) {
			fract[i] = arr[i] - fract[i]
		}
		return fract
	}

	// number
	return float32(arr - float32(arr))
}

// Make sure data is float32 array
function float32 (arr) {
	if (arr.length) {
		if (arr instanceof Float32Array) return arr
		return new Float32Array(arr);
	}

	// number
	narr[0] = arr
	return narr[0]
}
