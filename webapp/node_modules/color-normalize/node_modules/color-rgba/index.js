/** @module  color-rgba */

'use strict'

var parse = require('color-parse')

module.exports = function rgba (color) {
	// template literals
	if (Array.isArray(color) && color.raw) color = String.raw.apply(null, arguments)

	var values, i, l

	//attempt to parse non-array arguments
	var parsed = parse(color)

	if (!parsed.space) return []

	var min = [0,0,0], max = parsed.space[0] === 'h' ? [360,100,100] : [255,255,255]

	values = Array(3)
	values[0] = Math.min(Math.max(parsed.values[0], min[0]), max[0])
	values[1] = Math.min(Math.max(parsed.values[1], min[1]), max[1])
	values[2] = Math.min(Math.max(parsed.values[2], min[2]), max[2])

	if (parsed.space[0] === 'h') values = hsl2rgb(values)

	values.push(Math.min(Math.max(parsed.alpha, 0), 1))

	return values
}


// excerpt from color-space/hsl
function hsl2rgb(hsl) {
	var h = hsl[0]/360, s = hsl[1]/100, l = hsl[2]/100, t1, t2, t3, rgb, val, i=0;

	if (s === 0) return val = l * 255, [val, val, val];

	t2 = l < 0.5 ? l * (1 + s) : l + s - l * s;
	t1 = 2 * l - t2;

	rgb = [0, 0, 0];
	for (;i<3;) {
		t3 = h + 1 / 3 * - (i - 1);
		t3 < 0 ? t3++ : t3 > 1 && t3--;
		val = 6 * t3 < 1 ? t1 + (t2 - t1) * 6 * t3 :
		2 * t3 < 1 ? t2 :
		3 * t3 < 2 ?  t1 + (t2 - t1) * (2 / 3 - t3) * 6 :
		t1;
		rgb[i++] = val * 255;
	}

	return rgb;
}
