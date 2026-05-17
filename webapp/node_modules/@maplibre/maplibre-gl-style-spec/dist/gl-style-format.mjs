#!/usr/bin/env node
import fs from 'fs';

function getDefaultExportFromCjs (x) {
	return x && x.__esModule && Object.prototype.hasOwnProperty.call(x, 'default') ? x['default'] : x;
}

var minimist$1;
var hasRequiredMinimist;

function requireMinimist () {
	if (hasRequiredMinimist) return minimist$1;
	hasRequiredMinimist = 1;

	function hasKey(obj, keys) {
		var o = obj;
		keys.slice(0, -1).forEach(function (key) {
			o = o[key] || {};
		});

		var key = keys[keys.length - 1];
		return key in o;
	}

	function isNumber(x) {
		if (typeof x === 'number') { return true; }
		if ((/^0x[0-9a-f]+$/i).test(x)) { return true; }
		return (/^[-+]?(?:\d+(?:\.\d*)?|\.\d+)(e[-+]?\d+)?$/).test(x);
	}

	function isConstructorOrProto(obj, key) {
		return (key === 'constructor' && typeof obj[key] === 'function') || key === '__proto__';
	}

	minimist$1 = function (args, opts) {
		if (!opts) { opts = {}; }

		var flags = {
			bools: {},
			strings: {},
			unknownFn: null,
		};

		if (typeof opts.unknown === 'function') {
			flags.unknownFn = opts.unknown;
		}

		if (typeof opts.boolean === 'boolean' && opts.boolean) {
			flags.allBools = true;
		} else {
			[].concat(opts.boolean).filter(Boolean).forEach(function (key) {
				flags.bools[key] = true;
			});
		}

		var aliases = {};

		function aliasIsBoolean(key) {
			return aliases[key].some(function (x) {
				return flags.bools[x];
			});
		}

		Object.keys(opts.alias || {}).forEach(function (key) {
			aliases[key] = [].concat(opts.alias[key]);
			aliases[key].forEach(function (x) {
				aliases[x] = [key].concat(aliases[key].filter(function (y) {
					return x !== y;
				}));
			});
		});

		[].concat(opts.string).filter(Boolean).forEach(function (key) {
			flags.strings[key] = true;
			if (aliases[key]) {
				[].concat(aliases[key]).forEach(function (k) {
					flags.strings[k] = true;
				});
			}
		});

		var defaults = opts.default || {};

		var argv = { _: [] };

		function argDefined(key, arg) {
			return (flags.allBools && (/^--[^=]+$/).test(arg))
				|| flags.strings[key]
				|| flags.bools[key]
				|| aliases[key];
		}

		function setKey(obj, keys, value) {
			var o = obj;
			for (var i = 0; i < keys.length - 1; i++) {
				var key = keys[i];
				if (isConstructorOrProto(o, key)) { return; }
				if (o[key] === undefined) { o[key] = {}; }
				if (
					o[key] === Object.prototype
					|| o[key] === Number.prototype
					|| o[key] === String.prototype
				) {
					o[key] = {};
				}
				if (o[key] === Array.prototype) { o[key] = []; }
				o = o[key];
			}

			var lastKey = keys[keys.length - 1];
			if (isConstructorOrProto(o, lastKey)) { return; }
			if (
				o === Object.prototype
				|| o === Number.prototype
				|| o === String.prototype
			) {
				o = {};
			}
			if (o === Array.prototype) { o = []; }
			if (o[lastKey] === undefined || flags.bools[lastKey] || typeof o[lastKey] === 'boolean') {
				o[lastKey] = value;
			} else if (Array.isArray(o[lastKey])) {
				o[lastKey].push(value);
			} else {
				o[lastKey] = [o[lastKey], value];
			}
		}

		function setArg(key, val, arg) {
			if (arg && flags.unknownFn && !argDefined(key, arg)) {
				if (flags.unknownFn(arg) === false) { return; }
			}

			var value = !flags.strings[key] && isNumber(val)
				? Number(val)
				: val;
			setKey(argv, key.split('.'), value);

			(aliases[key] || []).forEach(function (x) {
				setKey(argv, x.split('.'), value);
			});
		}

		Object.keys(flags.bools).forEach(function (key) {
			setArg(key, defaults[key] === undefined ? false : defaults[key]);
		});

		var notFlags = [];

		if (args.indexOf('--') !== -1) {
			notFlags = args.slice(args.indexOf('--') + 1);
			args = args.slice(0, args.indexOf('--'));
		}

		for (var i = 0; i < args.length; i++) {
			var arg = args[i];
			var key;
			var next;

			if ((/^--.+=/).test(arg)) {
				// Using [\s\S] instead of . because js doesn't support the
				// 'dotall' regex modifier. See:
				// http://stackoverflow.com/a/1068308/13216
				var m = arg.match(/^--([^=]+)=([\s\S]*)$/);
				key = m[1];
				var value = m[2];
				if (flags.bools[key]) {
					value = value !== 'false';
				}
				setArg(key, value, arg);
			} else if ((/^--no-.+/).test(arg)) {
				key = arg.match(/^--no-(.+)/)[1];
				setArg(key, false, arg);
			} else if ((/^--.+/).test(arg)) {
				key = arg.match(/^--(.+)/)[1];
				next = args[i + 1];
				if (
					next !== undefined
					&& !(/^(-|--)[^-]/).test(next)
					&& !flags.bools[key]
					&& !flags.allBools
					&& (aliases[key] ? !aliasIsBoolean(key) : true)
				) {
					setArg(key, next, arg);
					i += 1;
				} else if ((/^(true|false)$/).test(next)) {
					setArg(key, next === 'true', arg);
					i += 1;
				} else {
					setArg(key, flags.strings[key] ? '' : true, arg);
				}
			} else if ((/^-[^-]+/).test(arg)) {
				var letters = arg.slice(1, -1).split('');

				var broken = false;
				for (var j = 0; j < letters.length; j++) {
					next = arg.slice(j + 2);

					if (next === '-') {
						setArg(letters[j], next, arg);
						continue;
					}

					if ((/[A-Za-z]/).test(letters[j]) && next[0] === '=') {
						setArg(letters[j], next.slice(1), arg);
						broken = true;
						break;
					}

					if (
						(/[A-Za-z]/).test(letters[j])
						&& (/-?\d+(\.\d*)?(e-?\d+)?$/).test(next)
					) {
						setArg(letters[j], next, arg);
						broken = true;
						break;
					}

					if (letters[j + 1] && letters[j + 1].match(/\W/)) {
						setArg(letters[j], arg.slice(j + 2), arg);
						broken = true;
						break;
					} else {
						setArg(letters[j], flags.strings[letters[j]] ? '' : true, arg);
					}
				}

				key = arg.slice(-1)[0];
				if (!broken && key !== '-') {
					if (
						args[i + 1]
						&& !(/^(-|--)[^-]/).test(args[i + 1])
						&& !flags.bools[key]
						&& (aliases[key] ? !aliasIsBoolean(key) : true)
					) {
						setArg(key, args[i + 1], arg);
						i += 1;
					} else if (args[i + 1] && (/^(true|false)$/).test(args[i + 1])) {
						setArg(key, args[i + 1] === 'true', arg);
						i += 1;
					} else {
						setArg(key, flags.strings[key] ? '' : true, arg);
					}
				}
			} else {
				if (!flags.unknownFn || flags.unknownFn(arg) !== false) {
					argv._.push(flags.strings._ || !isNumber(arg) ? arg : Number(arg));
				}
				if (opts.stopEarly) {
					argv._.push.apply(argv._, args.slice(i + 1));
					break;
				}
			}
		}

		Object.keys(defaults).forEach(function (k) {
			if (!hasKey(argv, k.split('.'))) {
				setKey(argv, k.split('.'), defaults[k]);

				(aliases[k] || []).forEach(function (x) {
					setKey(argv, x.split('.'), defaults[k]);
				});
			}
		});

		if (opts['--']) {
			argv['--'] = notFlags.slice();
		} else {
			notFlags.forEach(function (k) {
				argv._.push(k);
			});
		}

		return argv;
	};
	return minimist$1;
}

var minimistExports = requireMinimist();
var minimist = /*@__PURE__*/getDefaultExportFromCjs(minimistExports);

var $version = 8;
var $root = {
	version: {
		required: true,
		type: "enum",
		values: [
			8
		]
	},
	name: {
		type: "string"
	},
	metadata: {
		type: "*"
	},
	center: {
		type: "array",
		value: "number"
	},
	zoom: {
		type: "number"
	},
	bearing: {
		type: "number",
		"default": 0,
		period: 360,
		units: "degrees"
	},
	pitch: {
		type: "number",
		"default": 0,
		units: "degrees"
	},
	roll: {
		type: "number",
		"default": 0,
		units: "degrees"
	},
	light: {
		type: "light"
	},
	sky: {
		type: "sky"
	},
	projection: {
		type: "projection"
	},
	terrain: {
		type: "terrain"
	},
	sources: {
		required: true,
		type: "sources"
	},
	sprite: {
		type: "sprite"
	},
	glyphs: {
		type: "string"
	},
	transition: {
		type: "transition"
	},
	layers: {
		required: true,
		type: "array",
		value: "layer"
	}
};
var sources = {
	"*": {
		type: "source"
	}
};
var source = [
	"source_vector",
	"source_raster",
	"source_raster_dem",
	"source_geojson",
	"source_video",
	"source_image"
];
var source_vector = {
	type: {
		required: true,
		type: "enum",
		values: {
			vector: {
			}
		}
	},
	url: {
		type: "string"
	},
	tiles: {
		type: "array",
		value: "string"
	},
	bounds: {
		type: "array",
		value: "number",
		length: 4,
		"default": [
			-180,
			-85.051129,
			180,
			85.051129
		]
	},
	scheme: {
		type: "enum",
		values: {
			xyz: {
			},
			tms: {
			}
		},
		"default": "xyz"
	},
	minzoom: {
		type: "number",
		"default": 0
	},
	maxzoom: {
		type: "number",
		"default": 22
	},
	attribution: {
		type: "string"
	},
	promoteId: {
		type: "promoteId"
	},
	volatile: {
		type: "boolean",
		"default": false
	},
	"*": {
		type: "*"
	}
};
var source_raster = {
	type: {
		required: true,
		type: "enum",
		values: {
			raster: {
			}
		}
	},
	url: {
		type: "string"
	},
	tiles: {
		type: "array",
		value: "string"
	},
	bounds: {
		type: "array",
		value: "number",
		length: 4,
		"default": [
			-180,
			-85.051129,
			180,
			85.051129
		]
	},
	minzoom: {
		type: "number",
		"default": 0
	},
	maxzoom: {
		type: "number",
		"default": 22
	},
	tileSize: {
		type: "number",
		"default": 512,
		units: "pixels"
	},
	scheme: {
		type: "enum",
		values: {
			xyz: {
			},
			tms: {
			}
		},
		"default": "xyz"
	},
	attribution: {
		type: "string"
	},
	volatile: {
		type: "boolean",
		"default": false
	},
	"*": {
		type: "*"
	}
};
var source_raster_dem = {
	type: {
		required: true,
		type: "enum",
		values: {
			"raster-dem": {
			}
		}
	},
	url: {
		type: "string"
	},
	tiles: {
		type: "array",
		value: "string"
	},
	bounds: {
		type: "array",
		value: "number",
		length: 4,
		"default": [
			-180,
			-85.051129,
			180,
			85.051129
		]
	},
	minzoom: {
		type: "number",
		"default": 0
	},
	maxzoom: {
		type: "number",
		"default": 22
	},
	tileSize: {
		type: "number",
		"default": 512,
		units: "pixels"
	},
	attribution: {
		type: "string"
	},
	encoding: {
		type: "enum",
		values: {
			terrarium: {
			},
			mapbox: {
			},
			custom: {
			}
		},
		"default": "mapbox"
	},
	redFactor: {
		type: "number",
		"default": 1
	},
	blueFactor: {
		type: "number",
		"default": 1
	},
	greenFactor: {
		type: "number",
		"default": 1
	},
	baseShift: {
		type: "number",
		"default": 0
	},
	volatile: {
		type: "boolean",
		"default": false
	},
	"*": {
		type: "*"
	}
};
var source_geojson = {
	type: {
		required: true,
		type: "enum",
		values: {
			geojson: {
			}
		}
	},
	data: {
		required: true,
		type: "*"
	},
	maxzoom: {
		type: "number",
		"default": 18
	},
	attribution: {
		type: "string"
	},
	buffer: {
		type: "number",
		"default": 128,
		maximum: 512,
		minimum: 0
	},
	filter: {
		type: "*"
	},
	tolerance: {
		type: "number",
		"default": 0.375
	},
	cluster: {
		type: "boolean",
		"default": false
	},
	clusterRadius: {
		type: "number",
		"default": 50,
		minimum: 0
	},
	clusterMaxZoom: {
		type: "number"
	},
	clusterMinPoints: {
		type: "number"
	},
	clusterProperties: {
		type: "*"
	},
	lineMetrics: {
		type: "boolean",
		"default": false
	},
	generateId: {
		type: "boolean",
		"default": false
	},
	promoteId: {
		type: "promoteId"
	}
};
var source_video = {
	type: {
		required: true,
		type: "enum",
		values: {
			video: {
			}
		}
	},
	urls: {
		required: true,
		type: "array",
		value: "string"
	},
	coordinates: {
		required: true,
		type: "array",
		length: 4,
		value: {
			type: "array",
			length: 2,
			value: "number"
		}
	}
};
var source_image = {
	type: {
		required: true,
		type: "enum",
		values: {
			image: {
			}
		}
	},
	url: {
		required: true,
		type: "string"
	},
	coordinates: {
		required: true,
		type: "array",
		length: 4,
		value: {
			type: "array",
			length: 2,
			value: "number"
		}
	}
};
var layer = {
	id: {
		type: "string",
		required: true
	},
	type: {
		type: "enum",
		values: {
			fill: {
			},
			line: {
			},
			symbol: {
			},
			circle: {
			},
			heatmap: {
			},
			"fill-extrusion": {
			},
			raster: {
			},
			hillshade: {
			},
			background: {
			}
		},
		required: true
	},
	metadata: {
		type: "*"
	},
	source: {
		type: "string"
	},
	"source-layer": {
		type: "string"
	},
	minzoom: {
		type: "number",
		minimum: 0,
		maximum: 24
	},
	maxzoom: {
		type: "number",
		minimum: 0,
		maximum: 24
	},
	filter: {
		type: "filter"
	},
	layout: {
		type: "layout"
	},
	paint: {
		type: "paint"
	}
};
var layout = [
	"layout_fill",
	"layout_line",
	"layout_circle",
	"layout_heatmap",
	"layout_fill-extrusion",
	"layout_symbol",
	"layout_raster",
	"layout_hillshade",
	"layout_background"
];
var layout_background = {
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
};
var layout_fill = {
	"fill-sort-key": {
		type: "number",
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
};
var layout_circle = {
	"circle-sort-key": {
		type: "number",
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
};
var layout_heatmap = {
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
};
var layout_line = {
	"line-cap": {
		type: "enum",
		values: {
			butt: {
			},
			round: {
			},
			square: {
			}
		},
		"default": "butt",
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"line-join": {
		type: "enum",
		values: {
			bevel: {
			},
			round: {
			},
			miter: {
			}
		},
		"default": "miter",
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"line-miter-limit": {
		type: "number",
		"default": 2,
		requires: [
			{
				"line-join": "miter"
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"line-round-limit": {
		type: "number",
		"default": 1.05,
		requires: [
			{
				"line-join": "round"
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"line-sort-key": {
		type: "number",
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
};
var layout_symbol = {
	"symbol-placement": {
		type: "enum",
		values: {
			point: {
			},
			line: {
			},
			"line-center": {
			}
		},
		"default": "point",
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"symbol-spacing": {
		type: "number",
		"default": 250,
		minimum: 1,
		units: "pixels",
		requires: [
			{
				"symbol-placement": "line"
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"symbol-avoid-edges": {
		type: "boolean",
		"default": false,
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"symbol-sort-key": {
		type: "number",
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"symbol-z-order": {
		type: "enum",
		values: {
			auto: {
			},
			"viewport-y": {
			},
			source: {
			}
		},
		"default": "auto",
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-allow-overlap": {
		type: "boolean",
		"default": false,
		requires: [
			"icon-image",
			{
				"!": "icon-overlap"
			}
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-overlap": {
		type: "enum",
		values: {
			never: {
			},
			always: {
			},
			cooperative: {
			}
		},
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-ignore-placement": {
		type: "boolean",
		"default": false,
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-optional": {
		type: "boolean",
		"default": false,
		requires: [
			"icon-image",
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-rotation-alignment": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			},
			auto: {
			}
		},
		"default": "auto",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-size": {
		type: "number",
		"default": 1,
		minimum: 0,
		units: "factor of the original icon size",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"icon-text-fit": {
		type: "enum",
		values: {
			none: {
			},
			width: {
			},
			height: {
			},
			both: {
			}
		},
		"default": "none",
		requires: [
			"icon-image",
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-text-fit-padding": {
		type: "array",
		value: "number",
		length: 4,
		"default": [
			0,
			0,
			0,
			0
		],
		units: "pixels",
		requires: [
			"icon-image",
			"text-field",
			{
				"icon-text-fit": [
					"both",
					"width",
					"height"
				]
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-image": {
		type: "resolvedImage",
		tokens: true,
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"icon-rotate": {
		type: "number",
		"default": 0,
		period: 360,
		units: "degrees",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"icon-padding": {
		type: "padding",
		"default": [
			2
		],
		units: "pixels",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"icon-keep-upright": {
		type: "boolean",
		"default": false,
		requires: [
			"icon-image",
			{
				"icon-rotation-alignment": "map"
			},
			{
				"symbol-placement": [
					"line",
					"line-center"
				]
			}
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-offset": {
		type: "array",
		value: "number",
		length: 2,
		"default": [
			0,
			0
		],
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"icon-anchor": {
		type: "enum",
		values: {
			center: {
			},
			left: {
			},
			right: {
			},
			top: {
			},
			bottom: {
			},
			"top-left": {
			},
			"top-right": {
			},
			"bottom-left": {
			},
			"bottom-right": {
			}
		},
		"default": "center",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"icon-pitch-alignment": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			},
			auto: {
			}
		},
		"default": "auto",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-pitch-alignment": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			},
			auto: {
			}
		},
		"default": "auto",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-rotation-alignment": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			},
			"viewport-glyph": {
			},
			auto: {
			}
		},
		"default": "auto",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-field": {
		type: "formatted",
		"default": "",
		tokens: true,
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-font": {
		type: "array",
		value: "string",
		"default": [
			"Open Sans Regular",
			"Arial Unicode MS Regular"
		],
		requires: [
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-size": {
		type: "number",
		"default": 16,
		minimum: 0,
		units: "pixels",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-max-width": {
		type: "number",
		"default": 10,
		minimum: 0,
		units: "ems",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-line-height": {
		type: "number",
		"default": 1.2,
		units: "ems",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-letter-spacing": {
		type: "number",
		"default": 0,
		units: "ems",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-justify": {
		type: "enum",
		values: {
			auto: {
			},
			left: {
			},
			center: {
			},
			right: {
			}
		},
		"default": "center",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-radial-offset": {
		type: "number",
		units: "ems",
		"default": 0,
		requires: [
			"text-field"
		],
		"property-type": "data-driven",
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		}
	},
	"text-variable-anchor": {
		type: "array",
		value: "enum",
		values: {
			center: {
			},
			left: {
			},
			right: {
			},
			top: {
			},
			bottom: {
			},
			"top-left": {
			},
			"top-right": {
			},
			"bottom-left": {
			},
			"bottom-right": {
			}
		},
		requires: [
			"text-field",
			{
				"symbol-placement": [
					"point"
				]
			}
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-variable-anchor-offset": {
		type: "variableAnchorOffsetCollection",
		requires: [
			"text-field",
			{
				"symbol-placement": [
					"point"
				]
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-anchor": {
		type: "enum",
		values: {
			center: {
			},
			left: {
			},
			right: {
			},
			top: {
			},
			bottom: {
			},
			"top-left": {
			},
			"top-right": {
			},
			"bottom-left": {
			},
			"bottom-right": {
			}
		},
		"default": "center",
		requires: [
			"text-field",
			{
				"!": "text-variable-anchor"
			}
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-max-angle": {
		type: "number",
		"default": 45,
		units: "degrees",
		requires: [
			"text-field",
			{
				"symbol-placement": [
					"line",
					"line-center"
				]
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-writing-mode": {
		type: "array",
		value: "enum",
		values: {
			horizontal: {
			},
			vertical: {
			}
		},
		requires: [
			"text-field",
			{
				"symbol-placement": [
					"point"
				]
			}
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-rotate": {
		type: "number",
		"default": 0,
		period: 360,
		units: "degrees",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-padding": {
		type: "number",
		"default": 2,
		minimum: 0,
		units: "pixels",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-keep-upright": {
		type: "boolean",
		"default": true,
		requires: [
			"text-field",
			{
				"text-rotation-alignment": "map"
			},
			{
				"symbol-placement": [
					"line",
					"line-center"
				]
			}
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-transform": {
		type: "enum",
		values: {
			none: {
			},
			uppercase: {
			},
			lowercase: {
			}
		},
		"default": "none",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-offset": {
		type: "array",
		value: "number",
		units: "ems",
		length: 2,
		"default": [
			0,
			0
		],
		requires: [
			"text-field",
			{
				"!": "text-radial-offset"
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "data-driven"
	},
	"text-allow-overlap": {
		type: "boolean",
		"default": false,
		requires: [
			"text-field",
			{
				"!": "text-overlap"
			}
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-overlap": {
		type: "enum",
		values: {
			never: {
			},
			always: {
			},
			cooperative: {
			}
		},
		requires: [
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-ignore-placement": {
		type: "boolean",
		"default": false,
		requires: [
			"text-field"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-optional": {
		type: "boolean",
		"default": false,
		requires: [
			"text-field",
			"icon-image"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
};
var layout_raster = {
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
};
var layout_hillshade = {
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
};
var filter = {
	type: "array",
	value: "*"
};
var filter_operator = {
	type: "enum",
	values: {
		"==": {
		},
		"!=": {
		},
		">": {
		},
		">=": {
		},
		"<": {
		},
		"<=": {
		},
		"in": {
		},
		"!in": {
		},
		all: {
		},
		any: {
		},
		none: {
		},
		has: {
		},
		"!has": {
		}
	}
};
var geometry_type = {
	type: "enum",
	values: {
		Point: {
		},
		LineString: {
		},
		Polygon: {
		}
	}
};
var function_stop = {
	type: "array",
	minimum: 0,
	maximum: 24,
	value: [
		"number",
		"color"
	],
	length: 2
};
var expression = {
	type: "array",
	value: "*",
	minimum: 1
};
var light = {
	anchor: {
		type: "enum",
		"default": "viewport",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"property-type": "data-constant",
		transition: false,
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		}
	},
	position: {
		type: "array",
		"default": [
			1.15,
			210,
			30
		],
		length: 3,
		value: "number",
		"property-type": "data-constant",
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		}
	},
	color: {
		type: "color",
		"property-type": "data-constant",
		"default": "#ffffff",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	},
	intensity: {
		type: "number",
		"property-type": "data-constant",
		"default": 0.5,
		minimum: 0,
		maximum: 1,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	}
};
var sky = {
	"sky-color": {
		type: "color",
		"property-type": "data-constant",
		"default": "#88C6FC",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	},
	"horizon-color": {
		type: "color",
		"property-type": "data-constant",
		"default": "#ffffff",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	},
	"fog-color": {
		type: "color",
		"property-type": "data-constant",
		"default": "#ffffff",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	},
	"fog-ground-blend": {
		type: "number",
		"property-type": "data-constant",
		"default": 0.5,
		minimum: 0,
		maximum: 1,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	},
	"horizon-fog-blend": {
		type: "number",
		"property-type": "data-constant",
		"default": 0.8,
		minimum: 0,
		maximum: 1,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	},
	"sky-horizon-blend": {
		type: "number",
		"property-type": "data-constant",
		"default": 0.8,
		minimum: 0,
		maximum: 1,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	},
	"atmosphere-blend": {
		type: "number",
		"property-type": "data-constant",
		"default": 0.8,
		minimum: 0,
		maximum: 1,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		transition: true
	}
};
var terrain = {
	source: {
		type: "string",
		required: true
	},
	exaggeration: {
		type: "number",
		minimum: 0,
		"default": 1
	}
};
var projection = {
	type: {
		type: "enum",
		"default": "mercator",
		values: {
			mercator: {
			},
			globe: {
			}
		}
	}
};
var paint = [
	"paint_fill",
	"paint_line",
	"paint_circle",
	"paint_heatmap",
	"paint_fill-extrusion",
	"paint_symbol",
	"paint_raster",
	"paint_hillshade",
	"paint_background"
];
var paint_fill = {
	"fill-antialias": {
		type: "boolean",
		"default": true,
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"fill-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"fill-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		requires: [
			{
				"!": "fill-pattern"
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"fill-outline-color": {
		type: "color",
		transition: true,
		requires: [
			{
				"!": "fill-pattern"
			},
			{
				"fill-antialias": true
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"fill-translate": {
		type: "array",
		value: "number",
		length: 2,
		"default": [
			0,
			0
		],
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"fill-translate-anchor": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "map",
		requires: [
			"fill-translate"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"fill-pattern": {
		type: "resolvedImage",
		transition: true,
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "cross-faded-data-driven"
	}
};
var paint_line = {
	"line-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"line-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		requires: [
			{
				"!": "line-pattern"
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"line-translate": {
		type: "array",
		value: "number",
		length: 2,
		"default": [
			0,
			0
		],
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"line-translate-anchor": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "map",
		requires: [
			"line-translate"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"line-width": {
		type: "number",
		"default": 1,
		minimum: 0,
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"line-gap-width": {
		type: "number",
		"default": 0,
		minimum: 0,
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"line-offset": {
		type: "number",
		"default": 0,
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"line-blur": {
		type: "number",
		"default": 0,
		minimum: 0,
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"line-dasharray": {
		type: "array",
		value: "number",
		minimum: 0,
		transition: true,
		units: "line widths",
		requires: [
			{
				"!": "line-pattern"
			}
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "cross-faded"
	},
	"line-pattern": {
		type: "resolvedImage",
		transition: true,
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "cross-faded-data-driven"
	},
	"line-gradient": {
		type: "color",
		transition: false,
		requires: [
			{
				"!": "line-dasharray"
			},
			{
				"!": "line-pattern"
			},
			{
				source: "geojson",
				has: {
					lineMetrics: true
				}
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"line-progress"
			]
		},
		"property-type": "color-ramp"
	}
};
var paint_circle = {
	"circle-radius": {
		type: "number",
		"default": 5,
		minimum: 0,
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"circle-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"circle-blur": {
		type: "number",
		"default": 0,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"circle-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"circle-translate": {
		type: "array",
		value: "number",
		length: 2,
		"default": [
			0,
			0
		],
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"circle-translate-anchor": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "map",
		requires: [
			"circle-translate"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"circle-pitch-scale": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "map",
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"circle-pitch-alignment": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "viewport",
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"circle-stroke-width": {
		type: "number",
		"default": 0,
		minimum: 0,
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"circle-stroke-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"circle-stroke-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	}
};
var paint_heatmap = {
	"heatmap-radius": {
		type: "number",
		"default": 30,
		minimum: 1,
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"heatmap-weight": {
		type: "number",
		"default": 1,
		minimum: 0,
		transition: false,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"heatmap-intensity": {
		type: "number",
		"default": 1,
		minimum: 0,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"heatmap-color": {
		type: "color",
		"default": [
			"interpolate",
			[
				"linear"
			],
			[
				"heatmap-density"
			],
			0,
			"rgba(0, 0, 255, 0)",
			0.1,
			"royalblue",
			0.3,
			"cyan",
			0.5,
			"lime",
			0.7,
			"yellow",
			1,
			"red"
		],
		transition: false,
		expression: {
			interpolated: true,
			parameters: [
				"heatmap-density"
			]
		},
		"property-type": "color-ramp"
	},
	"heatmap-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	}
};
var paint_symbol = {
	"icon-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"icon-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"icon-halo-color": {
		type: "color",
		"default": "rgba(0, 0, 0, 0)",
		transition: true,
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"icon-halo-width": {
		type: "number",
		"default": 0,
		minimum: 0,
		transition: true,
		units: "pixels",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"icon-halo-blur": {
		type: "number",
		"default": 0,
		minimum: 0,
		transition: true,
		units: "pixels",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"icon-translate": {
		type: "array",
		value: "number",
		length: 2,
		"default": [
			0,
			0
		],
		transition: true,
		units: "pixels",
		requires: [
			"icon-image"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"icon-translate-anchor": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "map",
		requires: [
			"icon-image",
			"icon-translate"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"text-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		overridable: true,
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"text-halo-color": {
		type: "color",
		"default": "rgba(0, 0, 0, 0)",
		transition: true,
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"text-halo-width": {
		type: "number",
		"default": 0,
		minimum: 0,
		transition: true,
		units: "pixels",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"text-halo-blur": {
		type: "number",
		"default": 0,
		minimum: 0,
		transition: true,
		units: "pixels",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"text-translate": {
		type: "array",
		value: "number",
		length: 2,
		"default": [
			0,
			0
		],
		transition: true,
		units: "pixels",
		requires: [
			"text-field"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"text-translate-anchor": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "map",
		requires: [
			"text-field",
			"text-translate"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	}
};
var paint_raster = {
	"raster-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"raster-hue-rotate": {
		type: "number",
		"default": 0,
		period: 360,
		transition: true,
		units: "degrees",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"raster-brightness-min": {
		type: "number",
		"default": 0,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"raster-brightness-max": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"raster-saturation": {
		type: "number",
		"default": 0,
		minimum: -1,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"raster-contrast": {
		type: "number",
		"default": 0,
		minimum: -1,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"raster-resampling": {
		type: "enum",
		values: {
			linear: {
			},
			nearest: {
			}
		},
		"default": "linear",
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"raster-fade-duration": {
		type: "number",
		"default": 300,
		minimum: 0,
		transition: false,
		units: "milliseconds",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	}
};
var paint_hillshade = {
	"hillshade-illumination-direction": {
		type: "number",
		"default": 335,
		minimum: 0,
		maximum: 359,
		transition: false,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"hillshade-illumination-anchor": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "viewport",
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"hillshade-exaggeration": {
		type: "number",
		"default": 0.5,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"hillshade-shadow-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"hillshade-highlight-color": {
		type: "color",
		"default": "#FFFFFF",
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"hillshade-accent-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	}
};
var paint_background = {
	"background-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		requires: [
			{
				"!": "background-pattern"
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"background-pattern": {
		type: "resolvedImage",
		transition: true,
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "cross-faded"
	},
	"background-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	}
};
var transition = {
	duration: {
		type: "number",
		"default": 300,
		minimum: 0,
		units: "milliseconds"
	},
	delay: {
		type: "number",
		"default": 0,
		minimum: 0,
		units: "milliseconds"
	}
};
var promoteId = {
	"*": {
		type: "string"
	}
};
var spec = {
	$version: $version,
	$root: $root,
	sources: sources,
	source: source,
	source_vector: source_vector,
	source_raster: source_raster,
	source_raster_dem: source_raster_dem,
	source_geojson: source_geojson,
	source_video: source_video,
	source_image: source_image,
	layer: layer,
	layout: layout,
	layout_background: layout_background,
	layout_fill: layout_fill,
	layout_circle: layout_circle,
	layout_heatmap: layout_heatmap,
	"layout_fill-extrusion": {
	visibility: {
		type: "enum",
		values: {
			visible: {
			},
			none: {
			}
		},
		"default": "visible",
		"property-type": "constant"
	}
},
	layout_line: layout_line,
	layout_symbol: layout_symbol,
	layout_raster: layout_raster,
	layout_hillshade: layout_hillshade,
	filter: filter,
	filter_operator: filter_operator,
	geometry_type: geometry_type,
	"function": {
	expression: {
		type: "expression"
	},
	stops: {
		type: "array",
		value: "function_stop"
	},
	base: {
		type: "number",
		"default": 1,
		minimum: 0
	},
	property: {
		type: "string",
		"default": "$zoom"
	},
	type: {
		type: "enum",
		values: {
			identity: {
			},
			exponential: {
			},
			interval: {
			},
			categorical: {
			}
		},
		"default": "exponential"
	},
	colorSpace: {
		type: "enum",
		values: {
			rgb: {
			},
			lab: {
			},
			hcl: {
			}
		},
		"default": "rgb"
	},
	"default": {
		type: "*",
		required: false
	}
},
	function_stop: function_stop,
	expression: expression,
	light: light,
	sky: sky,
	terrain: terrain,
	projection: projection,
	paint: paint,
	paint_fill: paint_fill,
	"paint_fill-extrusion": {
	"fill-extrusion-opacity": {
		type: "number",
		"default": 1,
		minimum: 0,
		maximum: 1,
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"fill-extrusion-color": {
		type: "color",
		"default": "#000000",
		transition: true,
		requires: [
			{
				"!": "fill-extrusion-pattern"
			}
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"fill-extrusion-translate": {
		type: "array",
		value: "number",
		length: 2,
		"default": [
			0,
			0
		],
		transition: true,
		units: "pixels",
		expression: {
			interpolated: true,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"fill-extrusion-translate-anchor": {
		type: "enum",
		values: {
			map: {
			},
			viewport: {
			}
		},
		"default": "map",
		requires: [
			"fill-extrusion-translate"
		],
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	},
	"fill-extrusion-pattern": {
		type: "resolvedImage",
		transition: true,
		expression: {
			interpolated: false,
			parameters: [
				"zoom",
				"feature"
			]
		},
		"property-type": "cross-faded-data-driven"
	},
	"fill-extrusion-height": {
		type: "number",
		"default": 0,
		minimum: 0,
		units: "meters",
		transition: true,
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"fill-extrusion-base": {
		type: "number",
		"default": 0,
		minimum: 0,
		units: "meters",
		transition: true,
		requires: [
			"fill-extrusion-height"
		],
		expression: {
			interpolated: true,
			parameters: [
				"zoom",
				"feature",
				"feature-state"
			]
		},
		"property-type": "data-driven"
	},
	"fill-extrusion-vertical-gradient": {
		type: "boolean",
		"default": true,
		transition: false,
		expression: {
			interpolated: false,
			parameters: [
				"zoom"
			]
		},
		"property-type": "data-constant"
	}
},
	paint_line: paint_line,
	paint_circle: paint_circle,
	paint_heatmap: paint_heatmap,
	paint_symbol: paint_symbol,
	paint_raster: paint_raster,
	paint_hillshade: paint_hillshade,
	paint_background: paint_background,
	transition: transition,
	"property-type": {
	"data-driven": {
		type: "property-type"
	},
	"cross-faded": {
		type: "property-type"
	},
	"cross-faded-data-driven": {
		type: "property-type"
	},
	"color-ramp": {
		type: "property-type"
	},
	"data-constant": {
		type: "property-type"
	},
	constant: {
		type: "property-type"
	}
},
	promoteId: promoteId
};

// Note: This regex matches even invalid JSON strings, but since we’re
// working on the output of `JSON.stringify` we know that only valid strings
// are present (unless the user supplied a weird `options.indent` but in
// that case we don’t care since the output would be invalid anyway).
const stringOrChar = /("(?:[^\\"]|\\.)*")|[:,]/g;

function stringify(passedObj, options = {}) {
  const indent = JSON.stringify(
    [1],
    undefined,
    options.indent === undefined ? 2 : options.indent
  ).slice(2, -3);

  const maxLength =
    indent === ""
      ? Infinity
      : options.maxLength === undefined
      ? 80
      : options.maxLength;

  let { replacer } = options;

  return (function _stringify(obj, currentIndent, reserved) {
    if (obj && typeof obj.toJSON === "function") {
      obj = obj.toJSON();
    }

    const string = JSON.stringify(obj, replacer);

    if (string === undefined) {
      return string;
    }

    const length = maxLength - currentIndent.length - reserved;

    if (string.length <= length) {
      const prettified = string.replace(
        stringOrChar,
        (match, stringLiteral) => {
          return stringLiteral || `${match} `;
        }
      );
      if (prettified.length <= length) {
        return prettified;
      }
    }

    if (replacer != null) {
      obj = JSON.parse(string);
      replacer = undefined;
    }

    if (typeof obj === "object" && obj !== null) {
      const nextIndent = currentIndent + indent;
      const items = [];
      let index = 0;
      let start;
      let end;

      if (Array.isArray(obj)) {
        start = "[";
        end = "]";
        const { length } = obj;
        for (; index < length; index++) {
          items.push(
            _stringify(obj[index], nextIndent, index === length - 1 ? 0 : 1) ||
              "null"
          );
        }
      } else {
        start = "{";
        end = "}";
        const keys = Object.keys(obj);
        const { length } = keys;
        for (; index < length; index++) {
          const key = keys[index];
          const keyPart = `${JSON.stringify(key)}: `;
          const value = _stringify(
            obj[key],
            nextIndent,
            keyPart.length + (index === length - 1 ? 0 : 1)
          );
          if (value !== undefined) {
            items.push(keyPart + value);
          }
        }
      }

      if (items.length > 0) {
        return [start, indent + items.join(`,\n${nextIndent}`), end].join(
          `\n${currentIndent}`
        );
      }
    }

    return string;
  })(passedObj, "", 0);
}

function sortKeysBy(obj, reference) {
    const result = {};
    for (const key in reference) {
        if (obj[key] !== undefined) {
            result[key] = obj[key];
        }
    }
    for (const key in obj) {
        if (result[key] === undefined) {
            result[key] = obj[key];
        }
    }
    return result;
}
/**
 * Format a MapLibre Style.  Returns a stringified style with its keys
 * sorted in the same order as the reference style.
 *
 * The optional `space` argument is passed to
 * [`JSON.stringify`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/JSON/stringify)
 * to generate formatted output.
 *
 * If `space` is unspecified, a default of `2` spaces will be used.
 *
 * @private
 * @param {Object} style a MapLibre Style
 * @param {number} [space] space argument to pass to `JSON.stringify`
 * @returns {string} stringified formatted JSON
 * @example
 * var fs = require('fs');
 * var format = require('maplibre-gl-style-spec').format;
 * var style = fs.readFileSync('./source.json', 'utf8');
 * fs.writeFileSync('./dest.json', format(style));
 * fs.writeFileSync('./dest.min.json', format(style, 0));
 */
function format(style, space = 2) {
    style = sortKeysBy(style, spec.$root);
    if (style.layers) {
        style.layers = style.layers.map((layer) => sortKeysBy(layer, spec.layer));
    }
    return stringify(style, { indent: space });
}

const argv = minimist(process.argv.slice(2));

if (argv.help || argv.h || (!argv._.length && process.stdin.isTTY)) {
    help();
} else {
    console.log(format(JSON.parse(fs.readFileSync(argv._[0]).toString()), argv.space));
}

function help() {
    console.log('usage:');
    console.log('  gl-style-format source.json > destination.json');
    console.log('');
    console.log('options:');
    console.log('  --space <num>');
    console.log('     Number of spaces in output (default "2")');
    console.log('     Pass "0" for minified output.');
}
//# sourceMappingURL=gl-style-format.mjs.map
