/**
 * @module color-parse
 */
import names from 'color-name'

export default parse

/**
 * Base hues
 * http://dev.w3.org/csswg/css-color/#typedef-named-hue
 */
//FIXME: use external hue detector
var baseHues = {
	red: 0,
	orange: 60,
	yellow: 120,
	green: 180,
	blue: 240,
	purple: 300
}

/**
 * Parse color from the string passed
 *
 * @return {Object} A space indicator `space`, an array `values` and `alpha`
 */
function parse(cstr) {
	var m, parts = [], alpha = 1, space

	//numeric case
	if (typeof cstr === 'number') {
		return { space: 'rgb', values: [cstr >>> 16, (cstr & 0x00ff00) >>> 8, cstr & 0x0000ff], alpha: 1 }
	}
	if (typeof cstr === 'number') return { space: 'rgb', values: [cstr >>> 16, (cstr & 0x00ff00) >>> 8, cstr & 0x0000ff], alpha: 1 }

	cstr = String(cstr).toLowerCase();

	//keyword
	if (names[cstr]) {
		parts = names[cstr].slice()
		space = 'rgb'
	}

	//reserved words
	else if (cstr === 'transparent') {
		alpha = 0
		space = 'rgb'
		parts = [0, 0, 0]
	}

	//hex
	else if (cstr[0] === '#') {
		var base = cstr.slice(1)
		var size = base.length
		var isShort = size <= 4
		alpha = 1

		if (isShort) {
			parts = [
				parseInt(base[0] + base[0], 16),
				parseInt(base[1] + base[1], 16),
				parseInt(base[2] + base[2], 16)
			]
			if (size === 4) {
				alpha = parseInt(base[3] + base[3], 16) / 255
			}
		}
		else {
			parts = [
				parseInt(base[0] + base[1], 16),
				parseInt(base[2] + base[3], 16),
				parseInt(base[4] + base[5], 16)
			]
			if (size === 8) {
				alpha = parseInt(base[6] + base[7], 16) / 255
			}
		}

		if (!parts[0]) parts[0] = 0
		if (!parts[1]) parts[1] = 0
		if (!parts[2]) parts[2] = 0

		space = 'rgb'
	}

	// color space
	else if (m = /^((?:rgba?|hs[lvb]a?|hwba?|cmyk?|xy[zy]|gray|lab|lchu?v?|[ly]uv|lms|oklch|oklab|color))\s*\(([^\)]*)\)/.exec(cstr)) {
		var name = m[1]
		space = name.replace(/a$/, '')
		var dims = space === 'cmyk' ? 4 : space === 'gray' ? 1 : 3
		parts = m[2].trim().split(/\s*[,\/]\s*|\s+/)

		// color(srgb-linear x x x) -> srgb-linear(x x x)
		if (space === 'color') space = parts.shift()

		parts = parts.map(function (x, i) {
			//<percentage>
			if (x[x.length - 1] === '%') {
				x = parseFloat(x) / 100
				// alpha -> 0..1
				if (i === 3) return x
				// rgb -> 0..255
				if (space === 'rgb') return x * 255
				// hsl, hwb H -> 0..100
				if (space[0] === 'h') return x * 100
				// lch, lab L -> 0..100
				if (space[0] === 'l' && !i) return x * 100
				// lab A B -> -125..125
				if (space === 'lab') return x * 125
				// lch C -> 0..150, H -> 0..360
				if (space === 'lch') return i < 2 ? x * 150 : x * 360
				// oklch/oklab L -> 0..1
				if (space[0] === 'o' && !i) return x
				// oklab A B -> -0.4..0.4
				if (space === 'oklab') return x * 0.4
				// oklch C -> 0..0.4, H -> 0..360
				if (space === 'oklch') return i < 2 ? x * 0.4 : x * 360
				// color(xxx) -> 0..1
				return x
			}

			//hue
			if (space[i] === 'h' || (i === 2 && space[space.length - 1] === 'h')) {
				//<base-hue>
				if (baseHues[x] !== undefined) return baseHues[x]
				//<deg>
				if (x.endsWith('deg')) return parseFloat(x)
				//<turn>
				if (x.endsWith('turn')) return parseFloat(x) * 360
				if (x.endsWith('grad')) return parseFloat(x) * 360 / 400
				if (x.endsWith('rad')) return parseFloat(x) * 180 / Math.PI
			}
			if (x === 'none') return 0
			return parseFloat(x)
		});

		alpha = parts.length > dims ? parts.pop() : 1
	}

	//named channels case
	else if (/[0-9](?:\s|\/|,)/.test(cstr)) {
		parts = cstr.match(/([0-9]+)/g).map(function (value) {
			return parseFloat(value)
		})

		space = cstr.match(/([a-z])/ig)?.join('')?.toLowerCase() || 'rgb'
	}

	return {
		space,
		values: parts,
		alpha
	}
}
