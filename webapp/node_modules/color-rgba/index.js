/** @module  color-rgba */
import parse from 'color-parse'
import rgb from 'color-space/rgb.js'
import hsl from 'color-space/hsl.js'

export default function rgba(color) {
	// template literals
	if (Array.isArray(color) && color.raw) color = String.raw(...arguments)
	if (color instanceof Number) color = +color

	var values, i, l

	//attempt to parse non-array arguments
	var parsed = parse(color)

	if (!parsed.space) return []

	const min = parsed.space[0] === 'h' ? hsl.min : rgb.min
	const max = parsed.space[0] === 'h' ? hsl.max : rgb.max

	values = Array(3)
	values[0] = Math.min(Math.max(parsed.values[0], min[0]), max[0])
	values[1] = Math.min(Math.max(parsed.values[1], min[1]), max[1])
	values[2] = Math.min(Math.max(parsed.values[2], min[2]), max[2])

	if (parsed.space[0] === 'h') {
		values = hsl.rgb(values)
	}

	values.push(Math.min(Math.max(parsed.alpha, 0), 1))

	return values
}
