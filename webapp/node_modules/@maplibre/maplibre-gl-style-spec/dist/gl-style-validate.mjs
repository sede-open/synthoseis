#!/usr/bin/env node
import require$$0 from 'fs';

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

var rw$1 = {};

var dash = {};

var decode;
var hasRequiredDecode;

function requireDecode () {
	if (hasRequiredDecode) return decode;
	hasRequiredDecode = 1;
	decode = function(options) {
	  if (options) {
	    if (typeof options === "string") return encoding(options);
	    if (options.encoding !== null) return encoding(options.encoding);
	  }
	  return identity();
	};

	function identity() {
	  var chunks = [];
	  return {
	    push: function(chunk) { chunks.push(chunk); },
	    value: function() { return Buffer.concat(chunks); }
	  };
	}

	function encoding(encoding) {
	  var chunks = [];
	  return {
	    push: function(chunk) { chunks.push(chunk); },
	    value: function() { return Buffer.concat(chunks).toString(encoding); }
	  };
	}
	return decode;
}

var readFile;
var hasRequiredReadFile;

function requireReadFile () {
	if (hasRequiredReadFile) return readFile;
	hasRequiredReadFile = 1;
	var fs = require$$0,
	    decode = requireDecode();

	readFile = function(path, options, callback) {
	  if (arguments.length < 3) callback = options, options = null;

	  switch (path) {
	    case "/dev/stdin": return readStream(process.stdin, options, callback);
	  }

	  fs.stat(path, function(error, stat) {
	    if (error) return callback(error);
	    if (stat.isFile()) return fs.readFile(path, options, callback);
	    readStream(fs.createReadStream(path, options ? {flags: options.flag || "r"} : {}), options, callback); // N.B. flag / flags
	  });
	};

	function readStream(stream, options, callback) {
	  var decoder = decode(options);
	  stream.on("error", callback);
	  stream.on("data", function(d) { decoder.push(d); });
	  stream.on("end", function() { callback(null, decoder.value()); });
	}
	return readFile;
}

var readFileSync;
var hasRequiredReadFileSync;

function requireReadFileSync () {
	if (hasRequiredReadFileSync) return readFileSync;
	hasRequiredReadFileSync = 1;
	var fs = require$$0,
	    decode = requireDecode();

	readFileSync = function(filename, options) {
	  if (fs.statSync(filename).isFile()) {
	    return fs.readFileSync(filename, options);
	  } else {
	    var fd = fs.openSync(filename, options && options.flag || "r"),
	        decoder = decode(options);

	    while (true) { // eslint-disable-line no-constant-condition
	      try {
	        var buffer = new Buffer(bufferSize),
	            bytesRead = fs.readSync(fd, buffer, 0, bufferSize);
	      } catch (e) {
	        if (e.code === "EOF") break;
	        fs.closeSync(fd);
	        throw e;
	      }
	      if (bytesRead === 0) break;
	      decoder.push(buffer.slice(0, bytesRead));
	    }

	    fs.closeSync(fd);
	    return decoder.value();
	  }
	};

	var bufferSize = 1 << 16;
	return readFileSync;
}

var encode;
var hasRequiredEncode;

function requireEncode () {
	if (hasRequiredEncode) return encode;
	hasRequiredEncode = 1;
	encode = function(data, options) {
	  return typeof data === "string"
	      ? new Buffer(data, typeof options === "string" ? options
	          : options && options.encoding !== null ? options.encoding
	          : "utf8")
	      : data;
	};
	return encode;
}

var writeFile;
var hasRequiredWriteFile;

function requireWriteFile () {
	if (hasRequiredWriteFile) return writeFile;
	hasRequiredWriteFile = 1;
	var fs = require$$0,
	    encode = requireEncode();

	writeFile = function(path, data, options, callback) {
	  if (arguments.length < 4) callback = options, options = null;

	  switch (path) {
	    case "/dev/stdout": return writeStream(process.stdout, "write", data, options, callback);
	    case "/dev/stderr": return writeStream(process.stderr, "write", data, options, callback);
	  }

	  fs.stat(path, function(error, stat) {
	    if (error && error.code !== "ENOENT") return callback(error);
	    if (stat && stat.isFile()) return fs.writeFile(path, data, options, callback);
	    writeStream(fs.createWriteStream(path, options ? {flags: options.flag || "w"} : {}), "end", data, options, callback); // N.B. flag / flags
	  });
	};

	function writeStream(stream, send, data, options, callback) {
	  stream.on("error", function(error) { callback(error.code === "EPIPE" ? null : error); }); // ignore broken pipe, e.g., | head
	  stream[send](encode(data, options), function(error) { callback(error && error.code === "EPIPE" ? null : error); });
	}
	return writeFile;
}

var writeFileSync;
var hasRequiredWriteFileSync;

function requireWriteFileSync () {
	if (hasRequiredWriteFileSync) return writeFileSync;
	hasRequiredWriteFileSync = 1;
	var fs = require$$0,
	    encode = requireEncode();

	writeFileSync = function(filename, data, options) {
	  var stat;

	  try {
	    stat = fs.statSync(filename);
	  } catch (error) {
	    if (error.code !== "ENOENT") throw error;
	  }

	  if (!stat || stat.isFile()) {
	    fs.writeFileSync(filename, data, options);
	  } else {
	    var fd = fs.openSync(filename, options && options.flag || "w"),
	        bytesWritten = 0,
	        bytesTotal = (data = encode(data, options)).length;

	    while (bytesWritten < bytesTotal) {
	      try {
	        bytesWritten += fs.writeSync(fd, data, bytesWritten, bytesTotal - bytesWritten, null);
	      } catch (error) {
	        if (error.code === "EPIPE") break; // ignore broken pipe, e.g., | head
	        fs.closeSync(fd);
	        throw error;
	      }
	    }

	    fs.closeSync(fd);
	  }
	};
	return writeFileSync;
}

var hasRequiredDash;

function requireDash () {
	if (hasRequiredDash) return dash;
	hasRequiredDash = 1;
	var slice = Array.prototype.slice;

	function dashify(method, file) {
	  return function(path) {
	    var argv = arguments;
	    if (path == "-") (argv = slice.call(argv)).splice(0, 1, file);
	    return method.apply(null, argv);
	  };
	}

	dash.readFile = dashify(requireReadFile(), "/dev/stdin");
	dash.readFileSync = dashify(requireReadFileSync(), "/dev/stdin");
	dash.writeFile = dashify(requireWriteFile(), "/dev/stdout");
	dash.writeFileSync = dashify(requireWriteFileSync(), "/dev/stdout");
	return dash;
}

var hasRequiredRw;

function requireRw () {
	if (hasRequiredRw) return rw$1;
	hasRequiredRw = 1;
	rw$1.dash = requireDash();
	rw$1.readFile = requireReadFile();
	rw$1.readFileSync = requireReadFileSync();
	rw$1.writeFile = requireWriteFile();
	rw$1.writeFileSync = requireWriteFileSync();
	return rw$1;
}

var rwExports = requireRw();
var rw = /*@__PURE__*/getDefaultExportFromCjs(rwExports);

// Note: Do not inherit from Error. It breaks when transpiling to ES5.
class ValidationError {
    constructor(key, value, message, identifier) {
        this.message = (key ? `${key}: ` : '') + message;
        if (identifier)
            this.identifier = identifier;
        if (value !== null && value !== undefined && value.__line__) {
            this.line = value.__line__;
        }
    }
}

function validateConstants(options) {
    const key = options.key;
    const constants = options.value;
    if (constants) {
        return [new ValidationError(key, constants, 'constants have been deprecated as of v8')];
    }
    else {
        return [];
    }
}

function extendBy(output, ...inputs) {
    for (const input of inputs) {
        for (const k in input) {
            output[k] = input[k];
        }
    }
    return output;
}

// Turn jsonlint-lines-primitives objects into primitive objects
function unbundle(value) {
    if (value instanceof Number || value instanceof String || value instanceof Boolean) {
        return value.valueOf();
    }
    else {
        return value;
    }
}
function deepUnbundle(value) {
    if (Array.isArray(value)) {
        return value.map(deepUnbundle);
    }
    else if (value instanceof Object && !(value instanceof Number || value instanceof String || value instanceof Boolean)) {
        const unbundledValue = {};
        for (const key in value) {
            unbundledValue[key] = deepUnbundle(value[key]);
        }
        return unbundledValue;
    }
    return unbundle(value);
}

class ExpressionParsingError extends Error {
    constructor(key, message) {
        super(message);
        this.message = message;
        this.key = key;
    }
}

/**
 * Tracks `let` bindings during expression parsing.
 * @private
 */
class Scope {
    constructor(parent, bindings = []) {
        this.parent = parent;
        this.bindings = {};
        for (const [name, expression] of bindings) {
            this.bindings[name] = expression;
        }
    }
    concat(bindings) {
        return new Scope(this, bindings);
    }
    get(name) {
        if (this.bindings[name]) {
            return this.bindings[name];
        }
        if (this.parent) {
            return this.parent.get(name);
        }
        throw new Error(`${name} not found in scope.`);
    }
    has(name) {
        if (this.bindings[name])
            return true;
        return this.parent ? this.parent.has(name) : false;
    }
}

const NullType = { kind: 'null' };
const NumberType = { kind: 'number' };
const StringType = { kind: 'string' };
const BooleanType = { kind: 'boolean' };
const ColorType = { kind: 'color' };
const ObjectType = { kind: 'object' };
const ValueType = { kind: 'value' };
const ErrorType = { kind: 'error' };
const CollatorType = { kind: 'collator' };
const FormattedType = { kind: 'formatted' };
const PaddingType = { kind: 'padding' };
const ResolvedImageType = { kind: 'resolvedImage' };
const VariableAnchorOffsetCollectionType = { kind: 'variableAnchorOffsetCollection' };
function array$1(itemType, N) {
    return {
        kind: 'array',
        itemType,
        N
    };
}
function toString$1(type) {
    if (type.kind === 'array') {
        const itemType = toString$1(type.itemType);
        return typeof type.N === 'number' ?
            `array<${itemType}, ${type.N}>` :
            type.itemType.kind === 'value' ? 'array' : `array<${itemType}>`;
    }
    else {
        return type.kind;
    }
}
const valueMemberTypes = [
    NullType,
    NumberType,
    StringType,
    BooleanType,
    ColorType,
    FormattedType,
    ObjectType,
    array$1(ValueType),
    PaddingType,
    ResolvedImageType,
    VariableAnchorOffsetCollectionType
];
/**
 * Returns null if `t` is a subtype of `expected`; otherwise returns an
 * error message.
 * @private
 */
function checkSubtype(expected, t) {
    if (t.kind === 'error') {
        // Error is a subtype of every type
        return null;
    }
    else if (expected.kind === 'array') {
        if (t.kind === 'array' &&
            ((t.N === 0 && t.itemType.kind === 'value') || !checkSubtype(expected.itemType, t.itemType)) &&
            (typeof expected.N !== 'number' || expected.N === t.N)) {
            return null;
        }
    }
    else if (expected.kind === t.kind) {
        return null;
    }
    else if (expected.kind === 'value') {
        for (const memberType of valueMemberTypes) {
            if (!checkSubtype(memberType, t)) {
                return null;
            }
        }
    }
    return `Expected ${toString$1(expected)} but found ${toString$1(t)} instead.`;
}
function isValidType(provided, allowedTypes) {
    return allowedTypes.some(t => t.kind === provided.kind);
}
function isValidNativeType(provided, allowedTypes) {
    return allowedTypes.some(t => {
        if (t === 'null') {
            return provided === null;
        }
        else if (t === 'array') {
            return Array.isArray(provided);
        }
        else if (t === 'object') {
            return provided && !Array.isArray(provided) && typeof provided === 'object';
        }
        else {
            return t === typeof provided;
        }
    });
}
/**
 * Verify whether the specified type is of the same type as the specified sample.
 *
 * @param provided Type to verify
 * @param sample Sample type to reference
 * @returns `true` if both objects are of the same type, `false` otherwise
 * @example basic types
 * if (verifyType(outputType, ValueType)) {
 *     // type narrowed to:
 *     outputType.kind; // 'value'
 * }
 * @example array types
 * if (verifyType(outputType, array(NumberType))) {
 *     // type narrowed to:
 *     outputType.kind; // 'array'
 *     outputType.itemType; // NumberTypeT
 *     outputType.itemType.kind; // 'number'
 * }
 */
function verifyType(provided, sample) {
    if (provided.kind === 'array' && sample.kind === 'array') {
        return provided.itemType.kind === sample.itemType.kind && typeof provided.N === 'number';
    }
    return provided.kind === sample.kind;
}

// See https://observablehq.com/@mbostock/lab-and-rgb
const Xn = 0.96422, Yn = 1, Zn = 0.82521, t0 = 4 / 29, t1 = 6 / 29, t2 = 3 * t1 * t1, t3 = t1 * t1 * t1, deg2rad = Math.PI / 180, rad2deg = 180 / Math.PI;
function constrainAngle(angle) {
    angle = angle % 360;
    if (angle < 0) {
        angle += 360;
    }
    return angle;
}
function rgbToLab([r, g, b, alpha]) {
    r = rgb2xyz(r);
    g = rgb2xyz(g);
    b = rgb2xyz(b);
    let x, z;
    const y = xyz2lab((0.2225045 * r + 0.7168786 * g + 0.0606169 * b) / Yn);
    if (r === g && g === b) {
        x = z = y;
    }
    else {
        x = xyz2lab((0.4360747 * r + 0.3850649 * g + 0.1430804 * b) / Xn);
        z = xyz2lab((0.0139322 * r + 0.0971045 * g + 0.7141733 * b) / Zn);
    }
    const l = 116 * y - 16;
    return [(l < 0) ? 0 : l, 500 * (x - y), 200 * (y - z), alpha];
}
function rgb2xyz(x) {
    return (x <= 0.04045) ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}
function xyz2lab(t) {
    return (t > t3) ? Math.pow(t, 1 / 3) : t / t2 + t0;
}
function labToRgb([l, a, b, alpha]) {
    let y = (l + 16) / 116, x = isNaN(a) ? y : y + a / 500, z = isNaN(b) ? y : y - b / 200;
    y = Yn * lab2xyz(y);
    x = Xn * lab2xyz(x);
    z = Zn * lab2xyz(z);
    return [
        xyz2rgb(3.1338561 * x - 1.6168667 * y - 0.4906146 * z), // D50 -> sRGB
        xyz2rgb(-0.9787684 * x + 1.9161415 * y + 0.0334540 * z),
        xyz2rgb(0.0719453 * x - 0.2289914 * y + 1.4052427 * z),
        alpha,
    ];
}
function xyz2rgb(x) {
    x = (x <= 0.00304) ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
    return (x < 0) ? 0 : (x > 1) ? 1 : x; // clip to 0..1 range
}
function lab2xyz(t) {
    return (t > t1) ? t * t * t : t2 * (t - t0);
}
function rgbToHcl(rgbColor) {
    const [l, a, b, alpha] = rgbToLab(rgbColor);
    const c = Math.sqrt(a * a + b * b);
    const h = Math.round(c * 10000) ? constrainAngle(Math.atan2(b, a) * rad2deg) : NaN;
    return [h, c, l, alpha];
}
function hclToRgb([h, c, l, alpha]) {
    h = isNaN(h) ? 0 : h * deg2rad;
    return labToRgb([l, Math.cos(h) * c, Math.sin(h) * c, alpha]);
}
// https://drafts.csswg.org/css-color-4/#hsl-to-rgb
function hslToRgb([h, s, l, alpha]) {
    h = constrainAngle(h);
    s /= 100;
    l /= 100;
    function f(n) {
        const k = (n + h / 30) % 12;
        const a = s * Math.min(l, 1 - l);
        return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    }
    return [f(0), f(8), f(4), alpha];
}

/**
 * CSS color parser compliant with CSS Color 4 Specification.
 * Supports: named colors, `transparent` keyword, all rgb hex notations,
 * rgb(), rgba(), hsl() and hsla() functions.
 * Does not round the parsed values to integers from the range 0..255.
 *
 * Syntax:
 *
 * <alpha-value> = <number> | <percentage>
 *         <hue> = <number> | <angle>
 *
 *         rgb() = rgb( <percentage>{3} [ / <alpha-value> ]? ) | rgb( <number>{3} [ / <alpha-value> ]? )
 *         rgb() = rgb( <percentage>#{3} , <alpha-value>? )    | rgb( <number>#{3} , <alpha-value>? )
 *
 *         hsl() = hsl( <hue> <percentage> <percentage> [ / <alpha-value> ]? )
 *         hsl() = hsl( <hue>, <percentage>, <percentage>, <alpha-value>? )
 *
 * Caveats:
 *   - <angle> - <number> with optional `deg` suffix; `grad`, `rad`, `turn` are not supported
 *   - `none` keyword is not supported
 *   - comments inside rgb()/hsl() are not supported
 *   - legacy color syntax rgba() is supported with an identical grammar and behavior to rgb()
 *   - legacy color syntax hsla() is supported with an identical grammar and behavior to hsl()
 *
 * @param input CSS color string to parse.
 * @returns Color in sRGB color space, with `red`, `green`, `blue`
 * and `alpha` channels normalized to the range 0..1,
 * or `undefined` if the input is not a valid color string.
 */
function parseCssColor(input) {
    input = input.toLowerCase().trim();
    if (input === 'transparent') {
        return [0, 0, 0, 0];
    }
    // 'white', 'black', 'blue'
    const namedColorsMatch = namedColors[input];
    if (namedColorsMatch) {
        const [r, g, b] = namedColorsMatch;
        return [r / 255, g / 255, b / 255, 1];
    }
    // #f0c, #f0cf, #ff00cc, #ff00ccff
    if (input.startsWith('#')) {
        const hexRegexp = /^#(?:[0-9a-f]{3,4}|[0-9a-f]{6}|[0-9a-f]{8})$/;
        if (hexRegexp.test(input)) {
            const step = input.length < 6 ? 1 : 2;
            let i = 1;
            return [
                parseHex(input.slice(i, i += step)),
                parseHex(input.slice(i, i += step)),
                parseHex(input.slice(i, i += step)),
                parseHex(input.slice(i, i + step) || 'ff'),
            ];
        }
    }
    // rgb(128 0 0), rgb(50% 0% 0%), rgba(255,0,255,0.6), rgb(255 0 255 / 60%), rgb(100% 0% 100% /.6)
    if (input.startsWith('rgb')) {
        const rgbRegExp = /^rgba?\(\s*([\de.+-]+)(%)?(?:\s+|\s*(,)\s*)([\de.+-]+)(%)?(?:\s+|\s*(,)\s*)([\de.+-]+)(%)?(?:\s*([,\/])\s*([\de.+-]+)(%)?)?\s*\)$/;
        const rgbMatch = input.match(rgbRegExp);
        if (rgbMatch) {
            const [_, // eslint-disable-line @typescript-eslint/no-unused-vars
            r, // <numeric>
            rp, // %         (optional)
            f1, // ,         (optional)
            g, // <numeric>
            gp, // %         (optional)
            f2, // ,         (optional)
            b, // <numeric>
            bp, // %         (optional)
            f3, // ,|/       (optional)
            a, // <numeric> (optional)
            ap, // %         (optional)
            ] = rgbMatch;
            const argFormat = [f1 || ' ', f2 || ' ', f3].join('');
            if (argFormat === '  ' ||
                argFormat === '  /' ||
                argFormat === ',,' ||
                argFormat === ',,,') {
                const valFormat = [rp, gp, bp].join('');
                const maxValue = (valFormat === '%%%') ? 100 :
                    (valFormat === '') ? 255 : 0;
                if (maxValue) {
                    const rgba = [
                        clamp(+r / maxValue, 0, 1),
                        clamp(+g / maxValue, 0, 1),
                        clamp(+b / maxValue, 0, 1),
                        a ? parseAlpha(+a, ap) : 1,
                    ];
                    if (validateNumbers(rgba)) {
                        return rgba;
                    }
                    // invalid numbers
                }
                // values must be all numbers or all percentages
            }
            return; // comma optional syntax requires no commas at all
        }
    }
    // hsl(120 50% 80%), hsla(120deg,50%,80%,.9), hsl(12e1 50% 80% / 90%)
    const hslRegExp = /^hsla?\(\s*([\de.+-]+)(?:deg)?(?:\s+|\s*(,)\s*)([\de.+-]+)%(?:\s+|\s*(,)\s*)([\de.+-]+)%(?:\s*([,\/])\s*([\de.+-]+)(%)?)?\s*\)$/;
    const hslMatch = input.match(hslRegExp);
    if (hslMatch) {
        const [_, // eslint-disable-line @typescript-eslint/no-unused-vars
        h, // <numeric>
        f1, // ,         (optional)
        s, // <numeric>
        f2, // ,         (optional)
        l, // <numeric>
        f3, // ,|/       (optional)
        a, // <numeric> (optional)
        ap, // %         (optional)
        ] = hslMatch;
        const argFormat = [f1 || ' ', f2 || ' ', f3].join('');
        if (argFormat === '  ' ||
            argFormat === '  /' ||
            argFormat === ',,' ||
            argFormat === ',,,') {
            const hsla = [
                +h,
                clamp(+s, 0, 100),
                clamp(+l, 0, 100),
                a ? parseAlpha(+a, ap) : 1,
            ];
            if (validateNumbers(hsla)) {
                return hslToRgb(hsla);
            }
            // invalid numbers
        }
        // comma optional syntax requires no commas at all
    }
}
function parseHex(hex) {
    return parseInt(hex.padEnd(2, hex), 16) / 255;
}
function parseAlpha(a, asPercentage) {
    return clamp(asPercentage ? (a / 100) : a, 0, 1);
}
function clamp(n, min, max) {
    return Math.min(Math.max(min, n), max);
}
/**
 * The regular expression for numeric values is not super specific, and it may
 * happen that it will accept a value that is not a valid number. In order to
 * detect and eliminate such values this function exists.
 *
 * @param array Array of uncertain numbers.
 * @returns `true` if the specified array contains only valid numbers, `false` otherwise.
 */
function validateNumbers(array) {
    return !array.some(Number.isNaN);
}
/**
 * To generate:
 * - visit {@link https://www.w3.org/TR/css-color-4/#named-colors}
 * - run in the console:
 * @example
 * copy(`{\n${[...document.querySelector('.named-color-table tbody').children].map((tr) => `${tr.cells[2].textContent.trim()}: [${tr.cells[4].textContent.trim().split(/\s+/).join(', ')}],`).join('\n')}\n}`);
 */
const namedColors = {
    aliceblue: [240, 248, 255],
    antiquewhite: [250, 235, 215],
    aqua: [0, 255, 255],
    aquamarine: [127, 255, 212],
    azure: [240, 255, 255],
    beige: [245, 245, 220],
    bisque: [255, 228, 196],
    black: [0, 0, 0],
    blanchedalmond: [255, 235, 205],
    blue: [0, 0, 255],
    blueviolet: [138, 43, 226],
    brown: [165, 42, 42],
    burlywood: [222, 184, 135],
    cadetblue: [95, 158, 160],
    chartreuse: [127, 255, 0],
    chocolate: [210, 105, 30],
    coral: [255, 127, 80],
    cornflowerblue: [100, 149, 237],
    cornsilk: [255, 248, 220],
    crimson: [220, 20, 60],
    cyan: [0, 255, 255],
    darkblue: [0, 0, 139],
    darkcyan: [0, 139, 139],
    darkgoldenrod: [184, 134, 11],
    darkgray: [169, 169, 169],
    darkgreen: [0, 100, 0],
    darkgrey: [169, 169, 169],
    darkkhaki: [189, 183, 107],
    darkmagenta: [139, 0, 139],
    darkolivegreen: [85, 107, 47],
    darkorange: [255, 140, 0],
    darkorchid: [153, 50, 204],
    darkred: [139, 0, 0],
    darksalmon: [233, 150, 122],
    darkseagreen: [143, 188, 143],
    darkslateblue: [72, 61, 139],
    darkslategray: [47, 79, 79],
    darkslategrey: [47, 79, 79],
    darkturquoise: [0, 206, 209],
    darkviolet: [148, 0, 211],
    deeppink: [255, 20, 147],
    deepskyblue: [0, 191, 255],
    dimgray: [105, 105, 105],
    dimgrey: [105, 105, 105],
    dodgerblue: [30, 144, 255],
    firebrick: [178, 34, 34],
    floralwhite: [255, 250, 240],
    forestgreen: [34, 139, 34],
    fuchsia: [255, 0, 255],
    gainsboro: [220, 220, 220],
    ghostwhite: [248, 248, 255],
    gold: [255, 215, 0],
    goldenrod: [218, 165, 32],
    gray: [128, 128, 128],
    green: [0, 128, 0],
    greenyellow: [173, 255, 47],
    grey: [128, 128, 128],
    honeydew: [240, 255, 240],
    hotpink: [255, 105, 180],
    indianred: [205, 92, 92],
    indigo: [75, 0, 130],
    ivory: [255, 255, 240],
    khaki: [240, 230, 140],
    lavender: [230, 230, 250],
    lavenderblush: [255, 240, 245],
    lawngreen: [124, 252, 0],
    lemonchiffon: [255, 250, 205],
    lightblue: [173, 216, 230],
    lightcoral: [240, 128, 128],
    lightcyan: [224, 255, 255],
    lightgoldenrodyellow: [250, 250, 210],
    lightgray: [211, 211, 211],
    lightgreen: [144, 238, 144],
    lightgrey: [211, 211, 211],
    lightpink: [255, 182, 193],
    lightsalmon: [255, 160, 122],
    lightseagreen: [32, 178, 170],
    lightskyblue: [135, 206, 250],
    lightslategray: [119, 136, 153],
    lightslategrey: [119, 136, 153],
    lightsteelblue: [176, 196, 222],
    lightyellow: [255, 255, 224],
    lime: [0, 255, 0],
    limegreen: [50, 205, 50],
    linen: [250, 240, 230],
    magenta: [255, 0, 255],
    maroon: [128, 0, 0],
    mediumaquamarine: [102, 205, 170],
    mediumblue: [0, 0, 205],
    mediumorchid: [186, 85, 211],
    mediumpurple: [147, 112, 219],
    mediumseagreen: [60, 179, 113],
    mediumslateblue: [123, 104, 238],
    mediumspringgreen: [0, 250, 154],
    mediumturquoise: [72, 209, 204],
    mediumvioletred: [199, 21, 133],
    midnightblue: [25, 25, 112],
    mintcream: [245, 255, 250],
    mistyrose: [255, 228, 225],
    moccasin: [255, 228, 181],
    navajowhite: [255, 222, 173],
    navy: [0, 0, 128],
    oldlace: [253, 245, 230],
    olive: [128, 128, 0],
    olivedrab: [107, 142, 35],
    orange: [255, 165, 0],
    orangered: [255, 69, 0],
    orchid: [218, 112, 214],
    palegoldenrod: [238, 232, 170],
    palegreen: [152, 251, 152],
    paleturquoise: [175, 238, 238],
    palevioletred: [219, 112, 147],
    papayawhip: [255, 239, 213],
    peachpuff: [255, 218, 185],
    peru: [205, 133, 63],
    pink: [255, 192, 203],
    plum: [221, 160, 221],
    powderblue: [176, 224, 230],
    purple: [128, 0, 128],
    rebeccapurple: [102, 51, 153],
    red: [255, 0, 0],
    rosybrown: [188, 143, 143],
    royalblue: [65, 105, 225],
    saddlebrown: [139, 69, 19],
    salmon: [250, 128, 114],
    sandybrown: [244, 164, 96],
    seagreen: [46, 139, 87],
    seashell: [255, 245, 238],
    sienna: [160, 82, 45],
    silver: [192, 192, 192],
    skyblue: [135, 206, 235],
    slateblue: [106, 90, 205],
    slategray: [112, 128, 144],
    slategrey: [112, 128, 144],
    snow: [255, 250, 250],
    springgreen: [0, 255, 127],
    steelblue: [70, 130, 180],
    tan: [210, 180, 140],
    teal: [0, 128, 128],
    thistle: [216, 191, 216],
    tomato: [255, 99, 71],
    turquoise: [64, 224, 208],
    violet: [238, 130, 238],
    wheat: [245, 222, 179],
    white: [255, 255, 255],
    whitesmoke: [245, 245, 245],
    yellow: [255, 255, 0],
    yellowgreen: [154, 205, 50],
};

/**
 * Color representation used by WebGL.
 * Defined in sRGB color space and pre-blended with alpha.
 * @private
 */
class Color {
    /**
     * @param r Red component premultiplied by `alpha` 0..1
     * @param g Green component premultiplied by `alpha` 0..1
     * @param b Blue component premultiplied by `alpha` 0..1
     * @param [alpha=1] Alpha component 0..1
     * @param [premultiplied=true] Whether the `r`, `g` and `b` values have already
     * been multiplied by alpha. If `true` nothing happens if `false` then they will
     * be multiplied automatically.
     */
    constructor(r, g, b, alpha = 1, premultiplied = true) {
        this.r = r;
        this.g = g;
        this.b = b;
        this.a = alpha;
        if (!premultiplied) {
            this.r *= alpha;
            this.g *= alpha;
            this.b *= alpha;
            if (!alpha) {
                // alpha = 0 erases completely rgb channels. This behavior is not desirable
                // if this particular color is later used in color interpolation.
                // Because of that, a reference to original color is saved.
                this.overwriteGetter('rgb', [r, g, b, alpha]);
            }
        }
    }
    /**
     * Parses CSS color strings and converts colors to sRGB color space if needed.
     * Officially supported color formats:
     * - keyword, e.g. 'aquamarine' or 'steelblue'
     * - hex (with 3, 4, 6 or 8 digits), e.g. '#f0f' or '#e9bebea9'
     * - rgb and rgba, e.g. 'rgb(0,240,120)' or 'rgba(0%,94%,47%,0.1)' or 'rgb(0 240 120 / .3)'
     * - hsl and hsla, e.g. 'hsl(0,0%,83%)' or 'hsla(0,0%,83%,.5)' or 'hsl(0 0% 83% / 20%)'
     *
     * @param input CSS color string to parse.
     * @returns A `Color` instance, or `undefined` if the input is not a valid color string.
     */
    static parse(input) {
        // in zoom-and-property function input could be an instance of Color class
        if (input instanceof Color) {
            return input;
        }
        if (typeof input !== 'string') {
            return;
        }
        const rgba = parseCssColor(input);
        if (rgba) {
            return new Color(...rgba, false);
        }
    }
    /**
     * Used in color interpolation and by 'to-rgba' expression.
     *
     * @returns Gien color, with reversed alpha blending, in sRGB color space.
     */
    get rgb() {
        const { r, g, b, a } = this;
        const f = a || Infinity; // reverse alpha blending factor
        return this.overwriteGetter('rgb', [r / f, g / f, b / f, a]);
    }
    /**
     * Used in color interpolation.
     *
     * @returns Gien color, with reversed alpha blending, in HCL color space.
     */
    get hcl() {
        return this.overwriteGetter('hcl', rgbToHcl(this.rgb));
    }
    /**
     * Used in color interpolation.
     *
     * @returns Gien color, with reversed alpha blending, in LAB color space.
     */
    get lab() {
        return this.overwriteGetter('lab', rgbToLab(this.rgb));
    }
    /**
     * Lazy getter pattern. When getter is called for the first time lazy value
     * is calculated and then overwrites getter function in given object instance.
     *
     * @example:
     * const redColor = Color.parse('red');
     * let x = redColor.hcl; // this will invoke `get hcl()`, which will calculate
     * // the value of red in HCL space and invoke this `overwriteGetter` function
     * // which in turn will set a field with a key 'hcl' in the `redColor` object.
     * // In other words it will override `get hcl()` from its `Color` prototype
     * // with its own property: hcl = [calculated red value in hcl].
     * let y = redColor.hcl; // next call will no longer invoke getter but simply
     * // return the previously calculated value
     * x === y; // true - `x` is exactly the same object as `y`
     *
     * @param getterKey Getter key
     * @param lazyValue Lazily calculated value to be memoized by current instance
     * @private
     */
    overwriteGetter(getterKey, lazyValue) {
        Object.defineProperty(this, getterKey, { value: lazyValue });
        return lazyValue;
    }
    /**
     * Used by 'to-string' expression.
     *
     * @returns Serialized color in format `rgba(r,g,b,a)`
     * where r,g,b are numbers within 0..255 and alpha is number within 1..0
     *
     * @example
     * var purple = new Color.parse('purple');
     * purple.toString; // = "rgba(128,0,128,1)"
     * var translucentGreen = new Color.parse('rgba(26, 207, 26, .73)');
     * translucentGreen.toString(); // = "rgba(26,207,26,0.73)"
     */
    toString() {
        const [r, g, b, a] = this.rgb;
        return `rgba(${[r, g, b].map(n => Math.round(n * 255)).join(',')},${a})`;
    }
}
Color.black = new Color(0, 0, 0, 1);
Color.white = new Color(1, 1, 1, 1);
Color.transparent = new Color(0, 0, 0, 0);
Color.red = new Color(1, 0, 0, 1);

// Flow type declarations for Intl cribbed from
// https://github.com/facebook/flow/issues/1270
class Collator {
    constructor(caseSensitive, diacriticSensitive, locale) {
        if (caseSensitive)
            this.sensitivity = diacriticSensitive ? 'variant' : 'case';
        else
            this.sensitivity = diacriticSensitive ? 'accent' : 'base';
        this.locale = locale;
        this.collator = new Intl.Collator(this.locale ? this.locale : [], { sensitivity: this.sensitivity, usage: 'search' });
    }
    compare(lhs, rhs) {
        return this.collator.compare(lhs, rhs);
    }
    resolvedLocale() {
        // We create a Collator without "usage: search" because we don't want
        // the search options encoded in our result (e.g. "en-u-co-search")
        return new Intl.Collator(this.locale ? this.locale : [])
            .resolvedOptions().locale;
    }
}

class FormattedSection {
    constructor(text, image, scale, fontStack, textColor) {
        this.text = text;
        this.image = image;
        this.scale = scale;
        this.fontStack = fontStack;
        this.textColor = textColor;
    }
}
class Formatted {
    constructor(sections) {
        this.sections = sections;
    }
    static fromString(unformatted) {
        return new Formatted([new FormattedSection(unformatted, null, null, null, null)]);
    }
    isEmpty() {
        if (this.sections.length === 0)
            return true;
        return !this.sections.some(section => section.text.length !== 0 ||
            (section.image && section.image.name.length !== 0));
    }
    static factory(text) {
        if (text instanceof Formatted) {
            return text;
        }
        else {
            return Formatted.fromString(text);
        }
    }
    toString() {
        if (this.sections.length === 0)
            return '';
        return this.sections.map(section => section.text).join('');
    }
}

/**
 * A set of four numbers representing padding around a box. Create instances from
 * bare arrays or numeric values using the static method `Padding.parse`.
 * @private
 */
class Padding {
    constructor(values) {
        this.values = values.slice();
    }
    /**
     * Numeric padding values
     * @param input A padding value
     * @returns A `Padding` instance, or `undefined` if the input is not a valid padding value.
     */
    static parse(input) {
        if (input instanceof Padding) {
            return input;
        }
        // Backwards compatibility: bare number is treated the same as array with single value.
        // Padding applies to all four sides.
        if (typeof input === 'number') {
            return new Padding([input, input, input, input]);
        }
        if (!Array.isArray(input)) {
            return undefined;
        }
        if (input.length < 1 || input.length > 4) {
            return undefined;
        }
        for (const val of input) {
            if (typeof val !== 'number') {
                return undefined;
            }
        }
        // Expand shortcut properties into explicit 4-sided values
        switch (input.length) {
            case 1:
                input = [input[0], input[0], input[0], input[0]];
                break;
            case 2:
                input = [input[0], input[1], input[0], input[1]];
                break;
            case 3:
                input = [input[0], input[1], input[2], input[1]];
                break;
        }
        return new Padding(input);
    }
    toString() {
        return JSON.stringify(this.values);
    }
}

/** Set of valid anchor positions, as a set for validation */
const anchors = new Set(['center', 'left', 'right', 'top', 'bottom', 'top-left', 'top-right', 'bottom-left', 'bottom-right']);
/**
 * Utility class to assist managing values for text-variable-anchor-offset property. Create instances from
 * bare arrays using the static method `VariableAnchorOffsetCollection.parse`.
 * @private
 */
class VariableAnchorOffsetCollection {
    constructor(values) {
        this.values = values.slice();
    }
    static parse(input) {
        if (input instanceof VariableAnchorOffsetCollection) {
            return input;
        }
        if (!Array.isArray(input) ||
            input.length < 1 ||
            input.length % 2 !== 0) {
            return undefined;
        }
        for (let i = 0; i < input.length; i += 2) {
            // Elements in even positions should be anchor positions; Elements in odd positions should be offset values
            const anchorValue = input[i];
            const offsetValue = input[i + 1];
            if (typeof anchorValue !== 'string' || !anchors.has(anchorValue)) {
                return undefined;
            }
            if (!Array.isArray(offsetValue) || offsetValue.length !== 2 || typeof offsetValue[0] !== 'number' || typeof offsetValue[1] !== 'number') {
                return undefined;
            }
        }
        return new VariableAnchorOffsetCollection(input);
    }
    toString() {
        return JSON.stringify(this.values);
    }
}

class ResolvedImage {
    constructor(options) {
        this.name = options.name;
        this.available = options.available;
    }
    toString() {
        return this.name;
    }
    static fromString(name) {
        if (!name)
            return null; // treat empty values as no image
        return new ResolvedImage({ name, available: false });
    }
}

function validateRGBA(r, g, b, a) {
    if (!(typeof r === 'number' && r >= 0 && r <= 255 &&
        typeof g === 'number' && g >= 0 && g <= 255 &&
        typeof b === 'number' && b >= 0 && b <= 255)) {
        const value = typeof a === 'number' ? [r, g, b, a] : [r, g, b];
        return `Invalid rgba value [${value.join(', ')}]: 'r', 'g', and 'b' must be between 0 and 255.`;
    }
    if (!(typeof a === 'undefined' || (typeof a === 'number' && a >= 0 && a <= 1))) {
        return `Invalid rgba value [${[r, g, b, a].join(', ')}]: 'a' must be between 0 and 1.`;
    }
    return null;
}
function isValue(mixed) {
    if (mixed === null ||
        typeof mixed === 'string' ||
        typeof mixed === 'boolean' ||
        typeof mixed === 'number' ||
        mixed instanceof Color ||
        mixed instanceof Collator ||
        mixed instanceof Formatted ||
        mixed instanceof Padding ||
        mixed instanceof VariableAnchorOffsetCollection ||
        mixed instanceof ResolvedImage) {
        return true;
    }
    else if (Array.isArray(mixed)) {
        for (const item of mixed) {
            if (!isValue(item)) {
                return false;
            }
        }
        return true;
    }
    else if (typeof mixed === 'object') {
        for (const key in mixed) {
            if (!isValue(mixed[key])) {
                return false;
            }
        }
        return true;
    }
    else {
        return false;
    }
}
function typeOf(value) {
    if (value === null) {
        return NullType;
    }
    else if (typeof value === 'string') {
        return StringType;
    }
    else if (typeof value === 'boolean') {
        return BooleanType;
    }
    else if (typeof value === 'number') {
        return NumberType;
    }
    else if (value instanceof Color) {
        return ColorType;
    }
    else if (value instanceof Collator) {
        return CollatorType;
    }
    else if (value instanceof Formatted) {
        return FormattedType;
    }
    else if (value instanceof Padding) {
        return PaddingType;
    }
    else if (value instanceof VariableAnchorOffsetCollection) {
        return VariableAnchorOffsetCollectionType;
    }
    else if (value instanceof ResolvedImage) {
        return ResolvedImageType;
    }
    else if (Array.isArray(value)) {
        const length = value.length;
        let itemType;
        for (const item of value) {
            const t = typeOf(item);
            if (!itemType) {
                itemType = t;
            }
            else if (itemType === t) {
                continue;
            }
            else {
                itemType = ValueType;
                break;
            }
        }
        return array$1(itemType || ValueType, length);
    }
    else {
        return ObjectType;
    }
}
function toString(value) {
    const type = typeof value;
    if (value === null) {
        return '';
    }
    else if (type === 'string' || type === 'number' || type === 'boolean') {
        return String(value);
    }
    else if (value instanceof Color || value instanceof Formatted || value instanceof Padding || value instanceof VariableAnchorOffsetCollection || value instanceof ResolvedImage) {
        return value.toString();
    }
    else {
        return JSON.stringify(value);
    }
}

class Literal {
    constructor(type, value) {
        this.type = type;
        this.value = value;
    }
    static parse(args, context) {
        if (args.length !== 2)
            return context.error(`'literal' expression requires exactly one argument, but found ${args.length - 1} instead.`);
        if (!isValue(args[1]))
            return context.error('invalid value');
        const value = args[1];
        let type = typeOf(value);
        // special case: infer the item type if possible for zero-length arrays
        const expected = context.expectedType;
        if (type.kind === 'array' &&
            type.N === 0 &&
            expected &&
            expected.kind === 'array' &&
            (typeof expected.N !== 'number' || expected.N === 0)) {
            type = expected;
        }
        return new Literal(type, value);
    }
    evaluate() {
        return this.value;
    }
    eachChild() { }
    outputDefined() {
        return true;
    }
}

class RuntimeError {
    constructor(message) {
        this.name = 'ExpressionEvaluationError';
        this.message = message;
    }
    toJSON() {
        return this.message;
    }
}

const types$1 = {
    string: StringType,
    number: NumberType,
    boolean: BooleanType,
    object: ObjectType
};
class Assertion {
    constructor(type, args) {
        this.type = type;
        this.args = args;
    }
    static parse(args, context) {
        if (args.length < 2)
            return context.error('Expected at least one argument.');
        let i = 1;
        let type;
        const name = args[0];
        if (name === 'array') {
            let itemType;
            if (args.length > 2) {
                const type = args[1];
                if (typeof type !== 'string' || !(type in types$1) || type === 'object')
                    return context.error('The item type argument of "array" must be one of string, number, boolean', 1);
                itemType = types$1[type];
                i++;
            }
            else {
                itemType = ValueType;
            }
            let N;
            if (args.length > 3) {
                if (args[2] !== null &&
                    (typeof args[2] !== 'number' ||
                        args[2] < 0 ||
                        args[2] !== Math.floor(args[2]))) {
                    return context.error('The length argument to "array" must be a positive integer literal', 2);
                }
                N = args[2];
                i++;
            }
            type = array$1(itemType, N);
        }
        else {
            if (!types$1[name])
                throw new Error(`Types doesn't contain name = ${name}`);
            type = types$1[name];
        }
        const parsed = [];
        for (; i < args.length; i++) {
            const input = context.parse(args[i], i, ValueType);
            if (!input)
                return null;
            parsed.push(input);
        }
        return new Assertion(type, parsed);
    }
    evaluate(ctx) {
        for (let i = 0; i < this.args.length; i++) {
            const value = this.args[i].evaluate(ctx);
            const error = checkSubtype(this.type, typeOf(value));
            if (!error) {
                return value;
            }
            else if (i === this.args.length - 1) {
                throw new RuntimeError(`Expected value to be of type ${toString$1(this.type)}, but found ${toString$1(typeOf(value))} instead.`);
            }
        }
        throw new Error();
    }
    eachChild(fn) {
        this.args.forEach(fn);
    }
    outputDefined() {
        return this.args.every(arg => arg.outputDefined());
    }
}

const types = {
    'to-boolean': BooleanType,
    'to-color': ColorType,
    'to-number': NumberType,
    'to-string': StringType
};
/**
 * Special form for error-coalescing coercion expressions "to-number",
 * "to-color".  Since these coercions can fail at runtime, they accept multiple
 * arguments, only evaluating one at a time until one succeeds.
 *
 * @private
 */
class Coercion {
    constructor(type, args) {
        this.type = type;
        this.args = args;
    }
    static parse(args, context) {
        if (args.length < 2)
            return context.error('Expected at least one argument.');
        const name = args[0];
        if (!types[name])
            throw new Error(`Can't parse ${name} as it is not part of the known types`);
        if ((name === 'to-boolean' || name === 'to-string') && args.length !== 2)
            return context.error('Expected one argument.');
        const type = types[name];
        const parsed = [];
        for (let i = 1; i < args.length; i++) {
            const input = context.parse(args[i], i, ValueType);
            if (!input)
                return null;
            parsed.push(input);
        }
        return new Coercion(type, parsed);
    }
    evaluate(ctx) {
        switch (this.type.kind) {
            case 'boolean':
                return Boolean(this.args[0].evaluate(ctx));
            case 'color': {
                let input;
                let error;
                for (const arg of this.args) {
                    input = arg.evaluate(ctx);
                    error = null;
                    if (input instanceof Color) {
                        return input;
                    }
                    else if (typeof input === 'string') {
                        const c = ctx.parseColor(input);
                        if (c)
                            return c;
                    }
                    else if (Array.isArray(input)) {
                        if (input.length < 3 || input.length > 4) {
                            error = `Invalid rgba value ${JSON.stringify(input)}: expected an array containing either three or four numeric values.`;
                        }
                        else {
                            error = validateRGBA(input[0], input[1], input[2], input[3]);
                        }
                        if (!error) {
                            return new Color(input[0] / 255, input[1] / 255, input[2] / 255, input[3]);
                        }
                    }
                }
                throw new RuntimeError(error || `Could not parse color from value '${typeof input === 'string' ? input : JSON.stringify(input)}'`);
            }
            case 'padding': {
                let input;
                for (const arg of this.args) {
                    input = arg.evaluate(ctx);
                    const pad = Padding.parse(input);
                    if (pad) {
                        return pad;
                    }
                }
                throw new RuntimeError(`Could not parse padding from value '${typeof input === 'string' ? input : JSON.stringify(input)}'`);
            }
            case 'variableAnchorOffsetCollection': {
                let input;
                for (const arg of this.args) {
                    input = arg.evaluate(ctx);
                    const coll = VariableAnchorOffsetCollection.parse(input);
                    if (coll) {
                        return coll;
                    }
                }
                throw new RuntimeError(`Could not parse variableAnchorOffsetCollection from value '${typeof input === 'string' ? input : JSON.stringify(input)}'`);
            }
            case 'number': {
                let value = null;
                for (const arg of this.args) {
                    value = arg.evaluate(ctx);
                    if (value === null)
                        return 0;
                    const num = Number(value);
                    if (isNaN(num))
                        continue;
                    return num;
                }
                throw new RuntimeError(`Could not convert ${JSON.stringify(value)} to number.`);
            }
            case 'formatted':
                // There is no explicit 'to-formatted' but this coercion can be implicitly
                // created by properties that expect the 'formatted' type.
                return Formatted.fromString(toString(this.args[0].evaluate(ctx)));
            case 'resolvedImage':
                return ResolvedImage.fromString(toString(this.args[0].evaluate(ctx)));
            default:
                return toString(this.args[0].evaluate(ctx));
        }
    }
    eachChild(fn) {
        this.args.forEach(fn);
    }
    outputDefined() {
        return this.args.every(arg => arg.outputDefined());
    }
}

const geometryTypes = ['Unknown', 'Point', 'LineString', 'Polygon'];
class EvaluationContext {
    constructor() {
        this.globals = null;
        this.feature = null;
        this.featureState = null;
        this.formattedSection = null;
        this._parseColorCache = {};
        this.availableImages = null;
        this.canonical = null;
    }
    id() {
        return this.feature && 'id' in this.feature ? this.feature.id : null;
    }
    geometryType() {
        return this.feature ? typeof this.feature.type === 'number' ? geometryTypes[this.feature.type] : this.feature.type : null;
    }
    geometry() {
        return this.feature && 'geometry' in this.feature ? this.feature.geometry : null;
    }
    canonicalID() {
        return this.canonical;
    }
    properties() {
        return this.feature && this.feature.properties || {};
    }
    parseColor(input) {
        let cached = this._parseColorCache[input];
        if (!cached) {
            cached = this._parseColorCache[input] = Color.parse(input);
        }
        return cached;
    }
}

/**
 * State associated parsing at a given point in an expression tree.
 * @private
 */
class ParsingContext {
    constructor(registry, isConstantFunc, path = [], expectedType, scope = new Scope(), errors = []) {
        this.registry = registry;
        this.path = path;
        this.key = path.map(part => `[${part}]`).join('');
        this.scope = scope;
        this.errors = errors;
        this.expectedType = expectedType;
        this._isConstant = isConstantFunc;
    }
    /**
     * @param expr the JSON expression to parse
     * @param index the optional argument index if this expression is an argument of a parent expression that's being parsed
     * @param options
     * @param options.omitTypeAnnotations set true to omit inferred type annotations.  Caller beware: with this option set, the parsed expression's type will NOT satisfy `expectedType` if it would normally be wrapped in an inferred annotation.
     * @private
     */
    parse(expr, index, expectedType, bindings, options = {}) {
        if (index) {
            return this.concat(index, expectedType, bindings)._parse(expr, options);
        }
        return this._parse(expr, options);
    }
    _parse(expr, options) {
        if (expr === null || typeof expr === 'string' || typeof expr === 'boolean' || typeof expr === 'number') {
            expr = ['literal', expr];
        }
        function annotate(parsed, type, typeAnnotation) {
            if (typeAnnotation === 'assert') {
                return new Assertion(type, [parsed]);
            }
            else if (typeAnnotation === 'coerce') {
                return new Coercion(type, [parsed]);
            }
            else {
                return parsed;
            }
        }
        if (Array.isArray(expr)) {
            if (expr.length === 0) {
                return this.error('Expected an array with at least one element. If you wanted a literal array, use ["literal", []].');
            }
            const op = expr[0];
            if (typeof op !== 'string') {
                this.error(`Expression name must be a string, but found ${typeof op} instead. If you wanted a literal array, use ["literal", [...]].`, 0);
                return null;
            }
            const Expr = this.registry[op];
            if (Expr) {
                let parsed = Expr.parse(expr, this);
                if (!parsed)
                    return null;
                if (this.expectedType) {
                    const expected = this.expectedType;
                    const actual = parsed.type;
                    // When we expect a number, string, boolean, or array but have a value, wrap it in an assertion.
                    // When we expect a color or formatted string, but have a string or value, wrap it in a coercion.
                    // Otherwise, we do static type-checking.
                    //
                    // These behaviors are overridable for:
                    //   * The "coalesce" operator, which needs to omit type annotations.
                    //   * String-valued properties (e.g. `text-field`), where coercion is more convenient than assertion.
                    //
                    if ((expected.kind === 'string' || expected.kind === 'number' || expected.kind === 'boolean' || expected.kind === 'object' || expected.kind === 'array') && actual.kind === 'value') {
                        parsed = annotate(parsed, expected, options.typeAnnotation || 'assert');
                    }
                    else if ((expected.kind === 'color' || expected.kind === 'formatted' || expected.kind === 'resolvedImage') && (actual.kind === 'value' || actual.kind === 'string')) {
                        parsed = annotate(parsed, expected, options.typeAnnotation || 'coerce');
                    }
                    else if (expected.kind === 'padding' && (actual.kind === 'value' || actual.kind === 'number' || actual.kind === 'array')) {
                        parsed = annotate(parsed, expected, options.typeAnnotation || 'coerce');
                    }
                    else if (expected.kind === 'variableAnchorOffsetCollection' && (actual.kind === 'value' || actual.kind === 'array')) {
                        parsed = annotate(parsed, expected, options.typeAnnotation || 'coerce');
                    }
                    else if (this.checkSubtype(expected, actual)) {
                        return null;
                    }
                }
                // If an expression's arguments are all literals, we can evaluate
                // it immediately and replace it with a literal value in the
                // parsed/compiled result. Expressions that expect an image should
                // not be resolved here so we can later get the available images.
                if (!(parsed instanceof Literal) && (parsed.type.kind !== 'resolvedImage') && this._isConstant(parsed)) {
                    const ec = new EvaluationContext();
                    try {
                        parsed = new Literal(parsed.type, parsed.evaluate(ec));
                    }
                    catch (e) {
                        this.error(e.message);
                        return null;
                    }
                }
                return parsed;
            }
            return this.error(`Unknown expression "${op}". If you wanted a literal array, use ["literal", [...]].`, 0);
        }
        else if (typeof expr === 'undefined') {
            return this.error('\'undefined\' value invalid. Use null instead.');
        }
        else if (typeof expr === 'object') {
            return this.error('Bare objects invalid. Use ["literal", {...}] instead.');
        }
        else {
            return this.error(`Expected an array, but found ${typeof expr} instead.`);
        }
    }
    /**
     * Returns a copy of this context suitable for parsing the subexpression at
     * index `index`, optionally appending to 'let' binding map.
     *
     * Note that `errors` property, intended for collecting errors while
     * parsing, is copied by reference rather than cloned.
     * @private
     */
    concat(index, expectedType, bindings) {
        const path = typeof index === 'number' ? this.path.concat(index) : this.path;
        const scope = bindings ? this.scope.concat(bindings) : this.scope;
        return new ParsingContext(this.registry, this._isConstant, path, expectedType || null, scope, this.errors);
    }
    /**
     * Push a parsing (or type checking) error into the `this.errors`
     * @param error The message
     * @param keys Optionally specify the source of the error at a child
     * of the current expression at `this.key`.
     * @private
     */
    error(error, ...keys) {
        const key = `${this.key}${keys.map(k => `[${k}]`).join('')}`;
        this.errors.push(new ExpressionParsingError(key, error));
    }
    /**
     * Returns null if `t` is a subtype of `expected`; otherwise returns an
     * error message and also pushes it to `this.errors`.
     * @param expected The expected type
     * @param t The actual type
     * @returns null if `t` is a subtype of `expected`; otherwise returns an error message
     */
    checkSubtype(expected, t) {
        const error = checkSubtype(expected, t);
        if (error)
            this.error(error);
        return error;
    }
}

class Let {
    constructor(bindings, result) {
        this.type = result.type;
        this.bindings = [].concat(bindings);
        this.result = result;
    }
    evaluate(ctx) {
        return this.result.evaluate(ctx);
    }
    eachChild(fn) {
        for (const binding of this.bindings) {
            fn(binding[1]);
        }
        fn(this.result);
    }
    static parse(args, context) {
        if (args.length < 4)
            return context.error(`Expected at least 3 arguments, but found ${args.length - 1} instead.`);
        const bindings = [];
        for (let i = 1; i < args.length - 1; i += 2) {
            const name = args[i];
            if (typeof name !== 'string') {
                return context.error(`Expected string, but found ${typeof name} instead.`, i);
            }
            if (/[^a-zA-Z0-9_]/.test(name)) {
                return context.error('Variable names must contain only alphanumeric characters or \'_\'.', i);
            }
            const value = context.parse(args[i + 1], i + 1);
            if (!value)
                return null;
            bindings.push([name, value]);
        }
        const result = context.parse(args[args.length - 1], args.length - 1, context.expectedType, bindings);
        if (!result)
            return null;
        return new Let(bindings, result);
    }
    outputDefined() {
        return this.result.outputDefined();
    }
}

class Var {
    constructor(name, boundExpression) {
        this.type = boundExpression.type;
        this.name = name;
        this.boundExpression = boundExpression;
    }
    static parse(args, context) {
        if (args.length !== 2 || typeof args[1] !== 'string')
            return context.error('\'var\' expression requires exactly one string literal argument.');
        const name = args[1];
        if (!context.scope.has(name)) {
            return context.error(`Unknown variable "${name}". Make sure "${name}" has been bound in an enclosing "let" expression before using it.`, 1);
        }
        return new Var(name, context.scope.get(name));
    }
    evaluate(ctx) {
        return this.boundExpression.evaluate(ctx);
    }
    eachChild() { }
    outputDefined() {
        return false;
    }
}

class At {
    constructor(type, index, input) {
        this.type = type;
        this.index = index;
        this.input = input;
    }
    static parse(args, context) {
        if (args.length !== 3)
            return context.error(`Expected 2 arguments, but found ${args.length - 1} instead.`);
        const index = context.parse(args[1], 1, NumberType);
        const input = context.parse(args[2], 2, array$1(context.expectedType || ValueType));
        if (!index || !input)
            return null;
        const t = input.type;
        return new At(t.itemType, index, input);
    }
    evaluate(ctx) {
        const index = this.index.evaluate(ctx);
        const array = this.input.evaluate(ctx);
        if (index < 0) {
            throw new RuntimeError(`Array index out of bounds: ${index} < 0.`);
        }
        if (index >= array.length) {
            throw new RuntimeError(`Array index out of bounds: ${index} > ${array.length - 1}.`);
        }
        if (index !== Math.floor(index)) {
            throw new RuntimeError(`Array index must be an integer, but found ${index} instead.`);
        }
        return array[index];
    }
    eachChild(fn) {
        fn(this.index);
        fn(this.input);
    }
    outputDefined() {
        return false;
    }
}

class In {
    constructor(needle, haystack) {
        this.type = BooleanType;
        this.needle = needle;
        this.haystack = haystack;
    }
    static parse(args, context) {
        if (args.length !== 3) {
            return context.error(`Expected 2 arguments, but found ${args.length - 1} instead.`);
        }
        const needle = context.parse(args[1], 1, ValueType);
        const haystack = context.parse(args[2], 2, ValueType);
        if (!needle || !haystack)
            return null;
        if (!isValidType(needle.type, [BooleanType, StringType, NumberType, NullType, ValueType])) {
            return context.error(`Expected first argument to be of type boolean, string, number or null, but found ${toString$1(needle.type)} instead`);
        }
        return new In(needle, haystack);
    }
    evaluate(ctx) {
        const needle = this.needle.evaluate(ctx);
        const haystack = this.haystack.evaluate(ctx);
        if (!haystack)
            return false;
        if (!isValidNativeType(needle, ['boolean', 'string', 'number', 'null'])) {
            throw new RuntimeError(`Expected first argument to be of type boolean, string, number or null, but found ${toString$1(typeOf(needle))} instead.`);
        }
        if (!isValidNativeType(haystack, ['string', 'array'])) {
            throw new RuntimeError(`Expected second argument to be of type array or string, but found ${toString$1(typeOf(haystack))} instead.`);
        }
        return haystack.indexOf(needle) >= 0;
    }
    eachChild(fn) {
        fn(this.needle);
        fn(this.haystack);
    }
    outputDefined() {
        return true;
    }
}

class IndexOf {
    constructor(needle, haystack, fromIndex) {
        this.type = NumberType;
        this.needle = needle;
        this.haystack = haystack;
        this.fromIndex = fromIndex;
    }
    static parse(args, context) {
        if (args.length <= 2 || args.length >= 5) {
            return context.error(`Expected 3 or 4 arguments, but found ${args.length - 1} instead.`);
        }
        const needle = context.parse(args[1], 1, ValueType);
        const haystack = context.parse(args[2], 2, ValueType);
        if (!needle || !haystack)
            return null;
        if (!isValidType(needle.type, [BooleanType, StringType, NumberType, NullType, ValueType])) {
            return context.error(`Expected first argument to be of type boolean, string, number or null, but found ${toString$1(needle.type)} instead`);
        }
        if (args.length === 4) {
            const fromIndex = context.parse(args[3], 3, NumberType);
            if (!fromIndex)
                return null;
            return new IndexOf(needle, haystack, fromIndex);
        }
        else {
            return new IndexOf(needle, haystack);
        }
    }
    evaluate(ctx) {
        const needle = this.needle.evaluate(ctx);
        const haystack = this.haystack.evaluate(ctx);
        if (!isValidNativeType(needle, ['boolean', 'string', 'number', 'null'])) {
            throw new RuntimeError(`Expected first argument to be of type boolean, string, number or null, but found ${toString$1(typeOf(needle))} instead.`);
        }
        let fromIndex;
        if (this.fromIndex) {
            fromIndex = this.fromIndex.evaluate(ctx);
        }
        if (isValidNativeType(haystack, ['string'])) {
            const rawIndex = haystack.indexOf(needle, fromIndex);
            if (rawIndex === -1) {
                return -1;
            }
            else {
                // The index may be affected by surrogate pairs, so get the length of the preceding substring.
                return [...haystack.slice(0, rawIndex)].length;
            }
        }
        else if (isValidNativeType(haystack, ['array'])) {
            return haystack.indexOf(needle, fromIndex);
        }
        else {
            throw new RuntimeError(`Expected second argument to be of type array or string, but found ${toString$1(typeOf(haystack))} instead.`);
        }
    }
    eachChild(fn) {
        fn(this.needle);
        fn(this.haystack);
        if (this.fromIndex) {
            fn(this.fromIndex);
        }
    }
    outputDefined() {
        return false;
    }
}

class Match {
    constructor(inputType, outputType, input, cases, outputs, otherwise) {
        this.inputType = inputType;
        this.type = outputType;
        this.input = input;
        this.cases = cases;
        this.outputs = outputs;
        this.otherwise = otherwise;
    }
    static parse(args, context) {
        if (args.length < 5)
            return context.error(`Expected at least 4 arguments, but found only ${args.length - 1}.`);
        if (args.length % 2 !== 1)
            return context.error('Expected an even number of arguments.');
        let inputType;
        let outputType;
        if (context.expectedType && context.expectedType.kind !== 'value') {
            outputType = context.expectedType;
        }
        const cases = {};
        const outputs = [];
        for (let i = 2; i < args.length - 1; i += 2) {
            let labels = args[i];
            const value = args[i + 1];
            if (!Array.isArray(labels)) {
                labels = [labels];
            }
            const labelContext = context.concat(i);
            if (labels.length === 0) {
                return labelContext.error('Expected at least one branch label.');
            }
            for (const label of labels) {
                if (typeof label !== 'number' && typeof label !== 'string') {
                    return labelContext.error('Branch labels must be numbers or strings.');
                }
                else if (typeof label === 'number' && Math.abs(label) > Number.MAX_SAFE_INTEGER) {
                    return labelContext.error(`Branch labels must be integers no larger than ${Number.MAX_SAFE_INTEGER}.`);
                }
                else if (typeof label === 'number' && Math.floor(label) !== label) {
                    return labelContext.error('Numeric branch labels must be integer values.');
                }
                else if (!inputType) {
                    inputType = typeOf(label);
                }
                else if (labelContext.checkSubtype(inputType, typeOf(label))) {
                    return null;
                }
                if (typeof cases[String(label)] !== 'undefined') {
                    return labelContext.error('Branch labels must be unique.');
                }
                cases[String(label)] = outputs.length;
            }
            const result = context.parse(value, i, outputType);
            if (!result)
                return null;
            outputType = outputType || result.type;
            outputs.push(result);
        }
        const input = context.parse(args[1], 1, ValueType);
        if (!input)
            return null;
        const otherwise = context.parse(args[args.length - 1], args.length - 1, outputType);
        if (!otherwise)
            return null;
        if (input.type.kind !== 'value' && context.concat(1).checkSubtype(inputType, input.type)) {
            return null;
        }
        return new Match(inputType, outputType, input, cases, outputs, otherwise);
    }
    evaluate(ctx) {
        const input = this.input.evaluate(ctx);
        const output = (typeOf(input) === this.inputType && this.outputs[this.cases[input]]) || this.otherwise;
        return output.evaluate(ctx);
    }
    eachChild(fn) {
        fn(this.input);
        this.outputs.forEach(fn);
        fn(this.otherwise);
    }
    outputDefined() {
        return this.outputs.every(out => out.outputDefined()) && this.otherwise.outputDefined();
    }
}

class Case {
    constructor(type, branches, otherwise) {
        this.type = type;
        this.branches = branches;
        this.otherwise = otherwise;
    }
    static parse(args, context) {
        if (args.length < 4)
            return context.error(`Expected at least 3 arguments, but found only ${args.length - 1}.`);
        if (args.length % 2 !== 0)
            return context.error('Expected an odd number of arguments.');
        let outputType;
        if (context.expectedType && context.expectedType.kind !== 'value') {
            outputType = context.expectedType;
        }
        const branches = [];
        for (let i = 1; i < args.length - 1; i += 2) {
            const test = context.parse(args[i], i, BooleanType);
            if (!test)
                return null;
            const result = context.parse(args[i + 1], i + 1, outputType);
            if (!result)
                return null;
            branches.push([test, result]);
            outputType = outputType || result.type;
        }
        const otherwise = context.parse(args[args.length - 1], args.length - 1, outputType);
        if (!otherwise)
            return null;
        if (!outputType)
            throw new Error('Can\'t infer output type');
        return new Case(outputType, branches, otherwise);
    }
    evaluate(ctx) {
        for (const [test, expression] of this.branches) {
            if (test.evaluate(ctx)) {
                return expression.evaluate(ctx);
            }
        }
        return this.otherwise.evaluate(ctx);
    }
    eachChild(fn) {
        for (const [test, expression] of this.branches) {
            fn(test);
            fn(expression);
        }
        fn(this.otherwise);
    }
    outputDefined() {
        return this.branches.every(([_, out]) => out.outputDefined()) && this.otherwise.outputDefined();
    }
}

class Slice {
    constructor(type, input, beginIndex, endIndex) {
        this.type = type;
        this.input = input;
        this.beginIndex = beginIndex;
        this.endIndex = endIndex;
    }
    static parse(args, context) {
        if (args.length <= 2 || args.length >= 5) {
            return context.error(`Expected 3 or 4 arguments, but found ${args.length - 1} instead.`);
        }
        const input = context.parse(args[1], 1, ValueType);
        const beginIndex = context.parse(args[2], 2, NumberType);
        if (!input || !beginIndex)
            return null;
        if (!isValidType(input.type, [array$1(ValueType), StringType, ValueType])) {
            return context.error(`Expected first argument to be of type array or string, but found ${toString$1(input.type)} instead`);
        }
        if (args.length === 4) {
            const endIndex = context.parse(args[3], 3, NumberType);
            if (!endIndex)
                return null;
            return new Slice(input.type, input, beginIndex, endIndex);
        }
        else {
            return new Slice(input.type, input, beginIndex);
        }
    }
    evaluate(ctx) {
        const input = this.input.evaluate(ctx);
        const beginIndex = this.beginIndex.evaluate(ctx);
        let endIndex;
        if (this.endIndex) {
            endIndex = this.endIndex.evaluate(ctx);
        }
        if (isValidNativeType(input, ['string'])) {
            // Indices may be affected by surrogate pairs.
            return [...input].slice(beginIndex, endIndex).join('');
        }
        else if (isValidNativeType(input, ['array'])) {
            return input.slice(beginIndex, endIndex);
        }
        else {
            throw new RuntimeError(`Expected first argument to be of type array or string, but found ${toString$1(typeOf(input))} instead.`);
        }
    }
    eachChild(fn) {
        fn(this.input);
        fn(this.beginIndex);
        if (this.endIndex) {
            fn(this.endIndex);
        }
    }
    outputDefined() {
        return false;
    }
}

/**
 * Returns the index of the last stop <= input, or 0 if it doesn't exist.
 * @private
 */
function findStopLessThanOrEqualTo(stops, input) {
    const lastIndex = stops.length - 1;
    let lowerIndex = 0;
    let upperIndex = lastIndex;
    let currentIndex = 0;
    let currentValue, nextValue;
    while (lowerIndex <= upperIndex) {
        currentIndex = Math.floor((lowerIndex + upperIndex) / 2);
        currentValue = stops[currentIndex];
        nextValue = stops[currentIndex + 1];
        if (currentValue <= input) {
            if (currentIndex === lastIndex || input < nextValue) { // Search complete
                return currentIndex;
            }
            lowerIndex = currentIndex + 1;
        }
        else if (currentValue > input) {
            upperIndex = currentIndex - 1;
        }
        else {
            throw new RuntimeError('Input is not a number.');
        }
    }
    return 0;
}

class Step {
    constructor(type, input, stops) {
        this.type = type;
        this.input = input;
        this.labels = [];
        this.outputs = [];
        for (const [label, expression] of stops) {
            this.labels.push(label);
            this.outputs.push(expression);
        }
    }
    static parse(args, context) {
        if (args.length - 1 < 4) {
            return context.error(`Expected at least 4 arguments, but found only ${args.length - 1}.`);
        }
        if ((args.length - 1) % 2 !== 0) {
            return context.error('Expected an even number of arguments.');
        }
        const input = context.parse(args[1], 1, NumberType);
        if (!input)
            return null;
        const stops = [];
        let outputType = null;
        if (context.expectedType && context.expectedType.kind !== 'value') {
            outputType = context.expectedType;
        }
        for (let i = 1; i < args.length; i += 2) {
            const label = i === 1 ? -Infinity : args[i];
            const value = args[i + 1];
            const labelKey = i;
            const valueKey = i + 1;
            if (typeof label !== 'number') {
                return context.error('Input/output pairs for "step" expressions must be defined using literal numeric values (not computed expressions) for the input values.', labelKey);
            }
            if (stops.length && stops[stops.length - 1][0] >= label) {
                return context.error('Input/output pairs for "step" expressions must be arranged with input values in strictly ascending order.', labelKey);
            }
            const parsed = context.parse(value, valueKey, outputType);
            if (!parsed)
                return null;
            outputType = outputType || parsed.type;
            stops.push([label, parsed]);
        }
        return new Step(outputType, input, stops);
    }
    evaluate(ctx) {
        const labels = this.labels;
        const outputs = this.outputs;
        if (labels.length === 1) {
            return outputs[0].evaluate(ctx);
        }
        const value = this.input.evaluate(ctx);
        if (value <= labels[0]) {
            return outputs[0].evaluate(ctx);
        }
        const stopCount = labels.length;
        if (value >= labels[stopCount - 1]) {
            return outputs[stopCount - 1].evaluate(ctx);
        }
        const index = findStopLessThanOrEqualTo(labels, value);
        return outputs[index].evaluate(ctx);
    }
    eachChild(fn) {
        fn(this.input);
        for (const expression of this.outputs) {
            fn(expression);
        }
    }
    outputDefined() {
        return this.outputs.every(out => out.outputDefined());
    }
}

var unitbezier;
var hasRequiredUnitbezier;

function requireUnitbezier () {
	if (hasRequiredUnitbezier) return unitbezier;
	hasRequiredUnitbezier = 1;

	unitbezier = UnitBezier;

	function UnitBezier(p1x, p1y, p2x, p2y) {
	    // Calculate the polynomial coefficients, implicit first and last control points are (0,0) and (1,1).
	    this.cx = 3.0 * p1x;
	    this.bx = 3.0 * (p2x - p1x) - this.cx;
	    this.ax = 1.0 - this.cx - this.bx;

	    this.cy = 3.0 * p1y;
	    this.by = 3.0 * (p2y - p1y) - this.cy;
	    this.ay = 1.0 - this.cy - this.by;

	    this.p1x = p1x;
	    this.p1y = p1y;
	    this.p2x = p2x;
	    this.p2y = p2y;
	}

	UnitBezier.prototype = {
	    sampleCurveX: function (t) {
	        // `ax t^3 + bx t^2 + cx t' expanded using Horner's rule.
	        return ((this.ax * t + this.bx) * t + this.cx) * t;
	    },

	    sampleCurveY: function (t) {
	        return ((this.ay * t + this.by) * t + this.cy) * t;
	    },

	    sampleCurveDerivativeX: function (t) {
	        return (3.0 * this.ax * t + 2.0 * this.bx) * t + this.cx;
	    },

	    solveCurveX: function (x, epsilon) {
	        if (epsilon === undefined) epsilon = 1e-6;

	        if (x < 0.0) return 0.0;
	        if (x > 1.0) return 1.0;

	        var t = x;

	        // First try a few iterations of Newton's method - normally very fast.
	        for (var i = 0; i < 8; i++) {
	            var x2 = this.sampleCurveX(t) - x;
	            if (Math.abs(x2) < epsilon) return t;

	            var d2 = this.sampleCurveDerivativeX(t);
	            if (Math.abs(d2) < 1e-6) break;

	            t = t - x2 / d2;
	        }

	        // Fall back to the bisection method for reliability.
	        var t0 = 0.0;
	        var t1 = 1.0;
	        t = x;

	        for (i = 0; i < 20; i++) {
	            x2 = this.sampleCurveX(t);
	            if (Math.abs(x2 - x) < epsilon) break;

	            if (x > x2) {
	                t0 = t;
	            } else {
	                t1 = t;
	            }

	            t = (t1 - t0) * 0.5 + t0;
	        }

	        return t;
	    },

	    solve: function (x, epsilon) {
	        return this.sampleCurveY(this.solveCurveX(x, epsilon));
	    }
	};
	return unitbezier;
}

var unitbezierExports = requireUnitbezier();
var UnitBezier = /*@__PURE__*/getDefaultExportFromCjs(unitbezierExports);

function number(from, to, t) {
    return from + t * (to - from);
}
function color(from, to, t, spaceKey = 'rgb') {
    switch (spaceKey) {
        case 'rgb': {
            const [r, g, b, alpha] = array(from.rgb, to.rgb, t);
            return new Color(r, g, b, alpha, false);
        }
        case 'hcl': {
            const [hue0, chroma0, light0, alphaF] = from.hcl;
            const [hue1, chroma1, light1, alphaT] = to.hcl;
            // https://github.com/gka/chroma.js/blob/cd1b3c0926c7a85cbdc3b1453b3a94006de91a92/src/interpolator/_hsx.js
            let hue, chroma;
            if (!isNaN(hue0) && !isNaN(hue1)) {
                let dh = hue1 - hue0;
                if (hue1 > hue0 && dh > 180) {
                    dh -= 360;
                }
                else if (hue1 < hue0 && hue0 - hue1 > 180) {
                    dh += 360;
                }
                hue = hue0 + t * dh;
            }
            else if (!isNaN(hue0)) {
                hue = hue0;
                if (light1 === 1 || light1 === 0)
                    chroma = chroma0;
            }
            else if (!isNaN(hue1)) {
                hue = hue1;
                if (light0 === 1 || light0 === 0)
                    chroma = chroma1;
            }
            else {
                hue = NaN;
            }
            const [r, g, b, alpha] = hclToRgb([
                hue,
                chroma !== null && chroma !== void 0 ? chroma : number(chroma0, chroma1, t),
                number(light0, light1, t),
                number(alphaF, alphaT, t),
            ]);
            return new Color(r, g, b, alpha, false);
        }
        case 'lab': {
            const [r, g, b, alpha] = labToRgb(array(from.lab, to.lab, t));
            return new Color(r, g, b, alpha, false);
        }
    }
}
function array(from, to, t) {
    return from.map((d, i) => {
        return number(d, to[i], t);
    });
}
function padding(from, to, t) {
    return new Padding(array(from.values, to.values, t));
}
function variableAnchorOffsetCollection(from, to, t) {
    const fromValues = from.values;
    const toValues = to.values;
    if (fromValues.length !== toValues.length) {
        throw new RuntimeError(`Cannot interpolate values of different length. from: ${from.toString()}, to: ${to.toString()}`);
    }
    const output = [];
    for (let i = 0; i < fromValues.length; i += 2) {
        // Anchor entries must match
        if (fromValues[i] !== toValues[i]) {
            throw new RuntimeError(`Cannot interpolate values containing mismatched anchors. from[${i}]: ${fromValues[i]}, to[${i}]: ${toValues[i]}`);
        }
        output.push(fromValues[i]);
        // Interpolate the offset values for each anchor
        const [fx, fy] = fromValues[i + 1];
        const [tx, ty] = toValues[i + 1];
        output.push([number(fx, tx, t), number(fy, ty, t)]);
    }
    return new VariableAnchorOffsetCollection(output);
}
const interpolate = {
    number,
    color,
    array,
    padding,
    variableAnchorOffsetCollection
};

class Interpolate {
    constructor(type, operator, interpolation, input, stops) {
        this.type = type;
        this.operator = operator;
        this.interpolation = interpolation;
        this.input = input;
        this.labels = [];
        this.outputs = [];
        for (const [label, expression] of stops) {
            this.labels.push(label);
            this.outputs.push(expression);
        }
    }
    static interpolationFactor(interpolation, input, lower, upper) {
        let t = 0;
        if (interpolation.name === 'exponential') {
            t = exponentialInterpolation(input, interpolation.base, lower, upper);
        }
        else if (interpolation.name === 'linear') {
            t = exponentialInterpolation(input, 1, lower, upper);
        }
        else if (interpolation.name === 'cubic-bezier') {
            const c = interpolation.controlPoints;
            const ub = new UnitBezier(c[0], c[1], c[2], c[3]);
            t = ub.solve(exponentialInterpolation(input, 1, lower, upper));
        }
        return t;
    }
    static parse(args, context) {
        let [operator, interpolation, input, ...rest] = args;
        if (!Array.isArray(interpolation) || interpolation.length === 0) {
            return context.error('Expected an interpolation type expression.', 1);
        }
        if (interpolation[0] === 'linear') {
            interpolation = { name: 'linear' };
        }
        else if (interpolation[0] === 'exponential') {
            const base = interpolation[1];
            if (typeof base !== 'number')
                return context.error('Exponential interpolation requires a numeric base.', 1, 1);
            interpolation = {
                name: 'exponential',
                base
            };
        }
        else if (interpolation[0] === 'cubic-bezier') {
            const controlPoints = interpolation.slice(1);
            if (controlPoints.length !== 4 ||
                controlPoints.some(t => typeof t !== 'number' || t < 0 || t > 1)) {
                return context.error('Cubic bezier interpolation requires four numeric arguments with values between 0 and 1.', 1);
            }
            interpolation = {
                name: 'cubic-bezier',
                controlPoints: controlPoints
            };
        }
        else {
            return context.error(`Unknown interpolation type ${String(interpolation[0])}`, 1, 0);
        }
        if (args.length - 1 < 4) {
            return context.error(`Expected at least 4 arguments, but found only ${args.length - 1}.`);
        }
        if ((args.length - 1) % 2 !== 0) {
            return context.error('Expected an even number of arguments.');
        }
        input = context.parse(input, 2, NumberType);
        if (!input)
            return null;
        const stops = [];
        let outputType = null;
        if (operator === 'interpolate-hcl' || operator === 'interpolate-lab') {
            outputType = ColorType;
        }
        else if (context.expectedType && context.expectedType.kind !== 'value') {
            outputType = context.expectedType;
        }
        for (let i = 0; i < rest.length; i += 2) {
            const label = rest[i];
            const value = rest[i + 1];
            const labelKey = i + 3;
            const valueKey = i + 4;
            if (typeof label !== 'number') {
                return context.error('Input/output pairs for "interpolate" expressions must be defined using literal numeric values (not computed expressions) for the input values.', labelKey);
            }
            if (stops.length && stops[stops.length - 1][0] >= label) {
                return context.error('Input/output pairs for "interpolate" expressions must be arranged with input values in strictly ascending order.', labelKey);
            }
            const parsed = context.parse(value, valueKey, outputType);
            if (!parsed)
                return null;
            outputType = outputType || parsed.type;
            stops.push([label, parsed]);
        }
        if (!verifyType(outputType, NumberType) &&
            !verifyType(outputType, ColorType) &&
            !verifyType(outputType, PaddingType) &&
            !verifyType(outputType, VariableAnchorOffsetCollectionType) &&
            !verifyType(outputType, array$1(NumberType))) {
            return context.error(`Type ${toString$1(outputType)} is not interpolatable.`);
        }
        return new Interpolate(outputType, operator, interpolation, input, stops);
    }
    evaluate(ctx) {
        const labels = this.labels;
        const outputs = this.outputs;
        if (labels.length === 1) {
            return outputs[0].evaluate(ctx);
        }
        const value = this.input.evaluate(ctx);
        if (value <= labels[0]) {
            return outputs[0].evaluate(ctx);
        }
        const stopCount = labels.length;
        if (value >= labels[stopCount - 1]) {
            return outputs[stopCount - 1].evaluate(ctx);
        }
        const index = findStopLessThanOrEqualTo(labels, value);
        const lower = labels[index];
        const upper = labels[index + 1];
        const t = Interpolate.interpolationFactor(this.interpolation, value, lower, upper);
        const outputLower = outputs[index].evaluate(ctx);
        const outputUpper = outputs[index + 1].evaluate(ctx);
        switch (this.operator) {
            case 'interpolate':
                return interpolate[this.type.kind](outputLower, outputUpper, t);
            case 'interpolate-hcl':
                return interpolate.color(outputLower, outputUpper, t, 'hcl');
            case 'interpolate-lab':
                return interpolate.color(outputLower, outputUpper, t, 'lab');
        }
    }
    eachChild(fn) {
        fn(this.input);
        for (const expression of this.outputs) {
            fn(expression);
        }
    }
    outputDefined() {
        return this.outputs.every(out => out.outputDefined());
    }
}
/**
 * Returns a ratio that can be used to interpolate between exponential function
 * stops.
 * How it works: Two consecutive stop values define a (scaled and shifted) exponential function `f(x) = a * base^x + b`, where `base` is the user-specified base,
 * and `a` and `b` are constants affording sufficient degrees of freedom to fit
 * the function to the given stops.
 *
 * Here's a bit of algebra that lets us compute `f(x)` directly from the stop
 * values without explicitly solving for `a` and `b`:
 *
 * First stop value: `f(x0) = y0 = a * base^x0 + b`
 * Second stop value: `f(x1) = y1 = a * base^x1 + b`
 * => `y1 - y0 = a(base^x1 - base^x0)`
 * => `a = (y1 - y0)/(base^x1 - base^x0)`
 *
 * Desired value: `f(x) = y = a * base^x + b`
 * => `f(x) = y0 + a * (base^x - base^x0)`
 *
 * From the above, we can replace the `a` in `a * (base^x - base^x0)` and do a
 * little algebra:
 * ```
 * a * (base^x - base^x0) = (y1 - y0)/(base^x1 - base^x0) * (base^x - base^x0)
 *                     = (y1 - y0) * (base^x - base^x0) / (base^x1 - base^x0)
 * ```
 *
 * If we let `(base^x - base^x0) / (base^x1 base^x0)`, then we have
 * `f(x) = y0 + (y1 - y0) * ratio`.  In other words, `ratio` may be treated as
 * an interpolation factor between the two stops' output values.
 *
 * (Note: a slightly different form for `ratio`,
 * `(base^(x-x0) - 1) / (base^(x1-x0) - 1) `, is equivalent, but requires fewer
 * expensive `Math.pow()` operations.)
 *
 * @private
*/
function exponentialInterpolation(input, base, lowerValue, upperValue) {
    const difference = upperValue - lowerValue;
    const progress = input - lowerValue;
    if (difference === 0) {
        return 0;
    }
    else if (base === 1) {
        return progress / difference;
    }
    else {
        return (Math.pow(base, progress) - 1) / (Math.pow(base, difference) - 1);
    }
}

class Coalesce {
    constructor(type, args) {
        this.type = type;
        this.args = args;
    }
    static parse(args, context) {
        if (args.length < 2) {
            return context.error('Expected at least one argument.');
        }
        let outputType = null;
        const expectedType = context.expectedType;
        if (expectedType && expectedType.kind !== 'value') {
            outputType = expectedType;
        }
        const parsedArgs = [];
        for (const arg of args.slice(1)) {
            const parsed = context.parse(arg, 1 + parsedArgs.length, outputType, undefined, { typeAnnotation: 'omit' });
            if (!parsed)
                return null;
            outputType = outputType || parsed.type;
            parsedArgs.push(parsed);
        }
        if (!outputType)
            throw new Error('No output type');
        // Above, we parse arguments without inferred type annotation so that
        // they don't produce a runtime error for `null` input, which would
        // preempt the desired null-coalescing behavior.
        // Thus, if any of our arguments would have needed an annotation, we
        // need to wrap the enclosing coalesce expression with it instead.
        const needsAnnotation = expectedType &&
            parsedArgs.some(arg => checkSubtype(expectedType, arg.type));
        return needsAnnotation ?
            new Coalesce(ValueType, parsedArgs) :
            new Coalesce(outputType, parsedArgs);
    }
    evaluate(ctx) {
        let result = null;
        let argCount = 0;
        let requestedImageName;
        for (const arg of this.args) {
            argCount++;
            result = arg.evaluate(ctx);
            // we need to keep track of the first requested image in a coalesce statement
            // if coalesce can't find a valid image, we return the first image name so styleimagemissing can fire
            if (result && result instanceof ResolvedImage && !result.available) {
                if (!requestedImageName) {
                    requestedImageName = result.name;
                }
                result = null;
                if (argCount === this.args.length) {
                    result = requestedImageName;
                }
            }
            if (result !== null)
                break;
        }
        return result;
    }
    eachChild(fn) {
        this.args.forEach(fn);
    }
    outputDefined() {
        return this.args.every(arg => arg.outputDefined());
    }
}

function isComparableType(op, type) {
    if (op === '==' || op === '!=') {
        // equality operator
        return type.kind === 'boolean' ||
            type.kind === 'string' ||
            type.kind === 'number' ||
            type.kind === 'null' ||
            type.kind === 'value';
    }
    else {
        // ordering operator
        return type.kind === 'string' ||
            type.kind === 'number' ||
            type.kind === 'value';
    }
}
function eq(ctx, a, b) { return a === b; }
function neq(ctx, a, b) { return a !== b; }
function lt(ctx, a, b) { return a < b; }
function gt(ctx, a, b) { return a > b; }
function lteq(ctx, a, b) { return a <= b; }
function gteq(ctx, a, b) { return a >= b; }
function eqCollate(ctx, a, b, c) { return c.compare(a, b) === 0; }
function neqCollate(ctx, a, b, c) { return !eqCollate(ctx, a, b, c); }
function ltCollate(ctx, a, b, c) { return c.compare(a, b) < 0; }
function gtCollate(ctx, a, b, c) { return c.compare(a, b) > 0; }
function lteqCollate(ctx, a, b, c) { return c.compare(a, b) <= 0; }
function gteqCollate(ctx, a, b, c) { return c.compare(a, b) >= 0; }
/**
 * Special form for comparison operators, implementing the signatures:
 * - (T, T, ?Collator) => boolean
 * - (T, value, ?Collator) => boolean
 * - (value, T, ?Collator) => boolean
 *
 * For inequalities, T must be either value, string, or number. For ==/!=, it
 * can also be boolean or null.
 *
 * Equality semantics are equivalent to Javascript's strict equality (===/!==)
 * -- i.e., when the arguments' types don't match, == evaluates to false, != to
 * true.
 *
 * When types don't match in an ordering comparison, a runtime error is thrown.
 *
 * @private
 */
function makeComparison(op, compareBasic, compareWithCollator) {
    const isOrderComparison = op !== '==' && op !== '!=';
    return class Comparison {
        constructor(lhs, rhs, collator) {
            this.type = BooleanType;
            this.lhs = lhs;
            this.rhs = rhs;
            this.collator = collator;
            this.hasUntypedArgument = lhs.type.kind === 'value' || rhs.type.kind === 'value';
        }
        static parse(args, context) {
            if (args.length !== 3 && args.length !== 4)
                return context.error('Expected two or three arguments.');
            const op = args[0];
            let lhs = context.parse(args[1], 1, ValueType);
            if (!lhs)
                return null;
            if (!isComparableType(op, lhs.type)) {
                return context.concat(1).error(`"${op}" comparisons are not supported for type '${toString$1(lhs.type)}'.`);
            }
            let rhs = context.parse(args[2], 2, ValueType);
            if (!rhs)
                return null;
            if (!isComparableType(op, rhs.type)) {
                return context.concat(2).error(`"${op}" comparisons are not supported for type '${toString$1(rhs.type)}'.`);
            }
            if (lhs.type.kind !== rhs.type.kind &&
                lhs.type.kind !== 'value' &&
                rhs.type.kind !== 'value') {
                return context.error(`Cannot compare types '${toString$1(lhs.type)}' and '${toString$1(rhs.type)}'.`);
            }
            if (isOrderComparison) {
                // typing rules specific to less/greater than operators
                if (lhs.type.kind === 'value' && rhs.type.kind !== 'value') {
                    // (value, T)
                    lhs = new Assertion(rhs.type, [lhs]);
                }
                else if (lhs.type.kind !== 'value' && rhs.type.kind === 'value') {
                    // (T, value)
                    rhs = new Assertion(lhs.type, [rhs]);
                }
            }
            let collator = null;
            if (args.length === 4) {
                if (lhs.type.kind !== 'string' &&
                    rhs.type.kind !== 'string' &&
                    lhs.type.kind !== 'value' &&
                    rhs.type.kind !== 'value') {
                    return context.error('Cannot use collator to compare non-string types.');
                }
                collator = context.parse(args[3], 3, CollatorType);
                if (!collator)
                    return null;
            }
            return new Comparison(lhs, rhs, collator);
        }
        evaluate(ctx) {
            const lhs = this.lhs.evaluate(ctx);
            const rhs = this.rhs.evaluate(ctx);
            if (isOrderComparison && this.hasUntypedArgument) {
                const lt = typeOf(lhs);
                const rt = typeOf(rhs);
                // check that type is string or number, and equal
                if (lt.kind !== rt.kind || !(lt.kind === 'string' || lt.kind === 'number')) {
                    throw new RuntimeError(`Expected arguments for "${op}" to be (string, string) or (number, number), but found (${lt.kind}, ${rt.kind}) instead.`);
                }
            }
            if (this.collator && !isOrderComparison && this.hasUntypedArgument) {
                const lt = typeOf(lhs);
                const rt = typeOf(rhs);
                if (lt.kind !== 'string' || rt.kind !== 'string') {
                    return compareBasic(ctx, lhs, rhs);
                }
            }
            return this.collator ?
                compareWithCollator(ctx, lhs, rhs, this.collator.evaluate(ctx)) :
                compareBasic(ctx, lhs, rhs);
        }
        eachChild(fn) {
            fn(this.lhs);
            fn(this.rhs);
            if (this.collator) {
                fn(this.collator);
            }
        }
        outputDefined() {
            return true;
        }
    };
}
const Equals = makeComparison('==', eq, eqCollate);
const NotEquals = makeComparison('!=', neq, neqCollate);
const LessThan = makeComparison('<', lt, ltCollate);
const GreaterThan = makeComparison('>', gt, gtCollate);
const LessThanOrEqual = makeComparison('<=', lteq, lteqCollate);
const GreaterThanOrEqual = makeComparison('>=', gteq, gteqCollate);

class CollatorExpression {
    constructor(caseSensitive, diacriticSensitive, locale) {
        this.type = CollatorType;
        this.locale = locale;
        this.caseSensitive = caseSensitive;
        this.diacriticSensitive = diacriticSensitive;
    }
    static parse(args, context) {
        if (args.length !== 2)
            return context.error('Expected one argument.');
        const options = args[1];
        if (typeof options !== 'object' || Array.isArray(options))
            return context.error('Collator options argument must be an object.');
        const caseSensitive = context.parse(options['case-sensitive'] === undefined ? false : options['case-sensitive'], 1, BooleanType);
        if (!caseSensitive)
            return null;
        const diacriticSensitive = context.parse(options['diacritic-sensitive'] === undefined ? false : options['diacritic-sensitive'], 1, BooleanType);
        if (!diacriticSensitive)
            return null;
        let locale = null;
        if (options['locale']) {
            locale = context.parse(options['locale'], 1, StringType);
            if (!locale)
                return null;
        }
        return new CollatorExpression(caseSensitive, diacriticSensitive, locale);
    }
    evaluate(ctx) {
        return new Collator(this.caseSensitive.evaluate(ctx), this.diacriticSensitive.evaluate(ctx), this.locale ? this.locale.evaluate(ctx) : null);
    }
    eachChild(fn) {
        fn(this.caseSensitive);
        fn(this.diacriticSensitive);
        if (this.locale) {
            fn(this.locale);
        }
    }
    outputDefined() {
        // Technically the set of possible outputs is the combinatoric set of Collators produced
        // by all possible outputs of locale/caseSensitive/diacriticSensitive
        // But for the primary use of Collators in comparison operators, we ignore the Collator's
        // possible outputs anyway, so we can get away with leaving this false for now.
        return false;
    }
}

class NumberFormat {
    constructor(number, locale, currency, minFractionDigits, maxFractionDigits) {
        this.type = StringType;
        this.number = number;
        this.locale = locale;
        this.currency = currency;
        this.minFractionDigits = minFractionDigits;
        this.maxFractionDigits = maxFractionDigits;
    }
    static parse(args, context) {
        if (args.length !== 3)
            return context.error('Expected two arguments.');
        const number = context.parse(args[1], 1, NumberType);
        if (!number)
            return null;
        const options = args[2];
        if (typeof options !== 'object' || Array.isArray(options))
            return context.error('NumberFormat options argument must be an object.');
        let locale = null;
        if (options['locale']) {
            locale = context.parse(options['locale'], 1, StringType);
            if (!locale)
                return null;
        }
        let currency = null;
        if (options['currency']) {
            currency = context.parse(options['currency'], 1, StringType);
            if (!currency)
                return null;
        }
        let minFractionDigits = null;
        if (options['min-fraction-digits']) {
            minFractionDigits = context.parse(options['min-fraction-digits'], 1, NumberType);
            if (!minFractionDigits)
                return null;
        }
        let maxFractionDigits = null;
        if (options['max-fraction-digits']) {
            maxFractionDigits = context.parse(options['max-fraction-digits'], 1, NumberType);
            if (!maxFractionDigits)
                return null;
        }
        return new NumberFormat(number, locale, currency, minFractionDigits, maxFractionDigits);
    }
    evaluate(ctx) {
        return new Intl.NumberFormat(this.locale ? this.locale.evaluate(ctx) : [], {
            style: this.currency ? 'currency' : 'decimal',
            currency: this.currency ? this.currency.evaluate(ctx) : undefined,
            minimumFractionDigits: this.minFractionDigits ? this.minFractionDigits.evaluate(ctx) : undefined,
            maximumFractionDigits: this.maxFractionDigits ? this.maxFractionDigits.evaluate(ctx) : undefined,
        }).format(this.number.evaluate(ctx));
    }
    eachChild(fn) {
        fn(this.number);
        if (this.locale) {
            fn(this.locale);
        }
        if (this.currency) {
            fn(this.currency);
        }
        if (this.minFractionDigits) {
            fn(this.minFractionDigits);
        }
        if (this.maxFractionDigits) {
            fn(this.maxFractionDigits);
        }
    }
    outputDefined() {
        return false;
    }
}

class FormatExpression {
    constructor(sections) {
        this.type = FormattedType;
        this.sections = sections;
    }
    static parse(args, context) {
        if (args.length < 2) {
            return context.error('Expected at least one argument.');
        }
        const firstArg = args[1];
        if (!Array.isArray(firstArg) && typeof firstArg === 'object') {
            return context.error('First argument must be an image or text section.');
        }
        const sections = [];
        let nextTokenMayBeObject = false;
        for (let i = 1; i <= args.length - 1; ++i) {
            const arg = args[i];
            if (nextTokenMayBeObject && typeof arg === 'object' && !Array.isArray(arg)) {
                nextTokenMayBeObject = false;
                let scale = null;
                if (arg['font-scale']) {
                    scale = context.parse(arg['font-scale'], 1, NumberType);
                    if (!scale)
                        return null;
                }
                let font = null;
                if (arg['text-font']) {
                    font = context.parse(arg['text-font'], 1, array$1(StringType));
                    if (!font)
                        return null;
                }
                let textColor = null;
                if (arg['text-color']) {
                    textColor = context.parse(arg['text-color'], 1, ColorType);
                    if (!textColor)
                        return null;
                }
                const lastExpression = sections[sections.length - 1];
                lastExpression.scale = scale;
                lastExpression.font = font;
                lastExpression.textColor = textColor;
            }
            else {
                const content = context.parse(args[i], 1, ValueType);
                if (!content)
                    return null;
                const kind = content.type.kind;
                if (kind !== 'string' && kind !== 'value' && kind !== 'null' && kind !== 'resolvedImage')
                    return context.error('Formatted text type must be \'string\', \'value\', \'image\' or \'null\'.');
                nextTokenMayBeObject = true;
                sections.push({ content, scale: null, font: null, textColor: null });
            }
        }
        return new FormatExpression(sections);
    }
    evaluate(ctx) {
        const evaluateSection = section => {
            const evaluatedContent = section.content.evaluate(ctx);
            if (typeOf(evaluatedContent) === ResolvedImageType) {
                return new FormattedSection('', evaluatedContent, null, null, null);
            }
            return new FormattedSection(toString(evaluatedContent), null, section.scale ? section.scale.evaluate(ctx) : null, section.font ? section.font.evaluate(ctx).join(',') : null, section.textColor ? section.textColor.evaluate(ctx) : null);
        };
        return new Formatted(this.sections.map(evaluateSection));
    }
    eachChild(fn) {
        for (const section of this.sections) {
            fn(section.content);
            if (section.scale) {
                fn(section.scale);
            }
            if (section.font) {
                fn(section.font);
            }
            if (section.textColor) {
                fn(section.textColor);
            }
        }
    }
    outputDefined() {
        // Technically the combinatoric set of all children
        // Usually, this.text will be undefined anyway
        return false;
    }
}

class ImageExpression {
    constructor(input) {
        this.type = ResolvedImageType;
        this.input = input;
    }
    static parse(args, context) {
        if (args.length !== 2) {
            return context.error('Expected two arguments.');
        }
        const name = context.parse(args[1], 1, StringType);
        if (!name)
            return context.error('No image name provided.');
        return new ImageExpression(name);
    }
    evaluate(ctx) {
        const evaluatedImageName = this.input.evaluate(ctx);
        const value = ResolvedImage.fromString(evaluatedImageName);
        if (value && ctx.availableImages)
            value.available = ctx.availableImages.indexOf(evaluatedImageName) > -1;
        return value;
    }
    eachChild(fn) {
        fn(this.input);
    }
    outputDefined() {
        // The output of image is determined by the list of available images in the evaluation context
        return false;
    }
}

class Length {
    constructor(input) {
        this.type = NumberType;
        this.input = input;
    }
    static parse(args, context) {
        if (args.length !== 2)
            return context.error(`Expected 1 argument, but found ${args.length - 1} instead.`);
        const input = context.parse(args[1], 1);
        if (!input)
            return null;
        if (input.type.kind !== 'array' && input.type.kind !== 'string' && input.type.kind !== 'value')
            return context.error(`Expected argument of type string or array, but found ${toString$1(input.type)} instead.`);
        return new Length(input);
    }
    evaluate(ctx) {
        const input = this.input.evaluate(ctx);
        if (typeof input === 'string') {
            // The length may be affected by surrogate pairs.
            return [...input].length;
        }
        else if (Array.isArray(input)) {
            return input.length;
        }
        else {
            throw new RuntimeError(`Expected value to be of type string or array, but found ${toString$1(typeOf(input))} instead.`);
        }
    }
    eachChild(fn) {
        fn(this.input);
    }
    outputDefined() {
        return false;
    }
}

const EXTENT = 8192;
function getTileCoordinates(p, canonical) {
    const x = mercatorXfromLng(p[0]);
    const y = mercatorYfromLat(p[1]);
    const tilesAtZoom = Math.pow(2, canonical.z);
    return [Math.round(x * tilesAtZoom * EXTENT), Math.round(y * tilesAtZoom * EXTENT)];
}
function getLngLatFromTileCoord(coord, canonical) {
    const tilesAtZoom = Math.pow(2, canonical.z);
    const x = (coord[0] / EXTENT + canonical.x) / tilesAtZoom;
    const y = (coord[1] / EXTENT + canonical.y) / tilesAtZoom;
    return [lngFromMercatorXfromLng(x), latFromMercatorY(y)];
}
function mercatorXfromLng(lng) {
    return (180 + lng) / 360;
}
function lngFromMercatorXfromLng(mercatorX) {
    return mercatorX * 360 - 180;
}
function mercatorYfromLat(lat) {
    return (180 - (180 / Math.PI * Math.log(Math.tan(Math.PI / 4 + lat * Math.PI / 360)))) / 360;
}
function latFromMercatorY(mercatorY) {
    return 360 / Math.PI * Math.atan(Math.exp((180 - mercatorY * 360) * Math.PI / 180)) - 90;
}
function updateBBox(bbox, coord) {
    bbox[0] = Math.min(bbox[0], coord[0]);
    bbox[1] = Math.min(bbox[1], coord[1]);
    bbox[2] = Math.max(bbox[2], coord[0]);
    bbox[3] = Math.max(bbox[3], coord[1]);
}
function boxWithinBox(bbox1, bbox2) {
    if (bbox1[0] <= bbox2[0])
        return false;
    if (bbox1[2] >= bbox2[2])
        return false;
    if (bbox1[1] <= bbox2[1])
        return false;
    if (bbox1[3] >= bbox2[3])
        return false;
    return true;
}
function rayIntersect(p, p1, p2) {
    return ((p1[1] > p[1]) !== (p2[1] > p[1])) && (p[0] < (p2[0] - p1[0]) * (p[1] - p1[1]) / (p2[1] - p1[1]) + p1[0]);
}
function pointOnBoundary(p, p1, p2) {
    const x1 = p[0] - p1[0];
    const y1 = p[1] - p1[1];
    const x2 = p[0] - p2[0];
    const y2 = p[1] - p2[1];
    return (x1 * y2 - x2 * y1 === 0) && (x1 * x2 <= 0) && (y1 * y2 <= 0);
}
// a, b are end points for line segment1, c and d are end points for line segment2
function segmentIntersectSegment(a, b, c, d) {
    // check if two segments are parallel or not
    // precondition is end point a, b is inside polygon, if line a->b is
    // parallel to polygon edge c->d, then a->b won't intersect with c->d
    const vectorP = [b[0] - a[0], b[1] - a[1]];
    const vectorQ = [d[0] - c[0], d[1] - c[1]];
    if (perp(vectorQ, vectorP) === 0)
        return false;
    // If lines are intersecting with each other, the relative location should be:
    // a and b lie in different sides of segment c->d
    // c and d lie in different sides of segment a->b
    if (twoSided(a, b, c, d) && twoSided(c, d, a, b))
        return true;
    return false;
}
function lineIntersectPolygon(p1, p2, polygon) {
    for (const ring of polygon) {
        // loop through every edge of the ring
        for (let j = 0; j < ring.length - 1; ++j) {
            if (segmentIntersectSegment(p1, p2, ring[j], ring[j + 1])) {
                return true;
            }
        }
    }
    return false;
}
// ray casting algorithm for detecting if point is in polygon
function pointWithinPolygon(point, rings, trueIfOnBoundary = false) {
    let inside = false;
    for (const ring of rings) {
        for (let j = 0; j < ring.length - 1; j++) {
            if (pointOnBoundary(point, ring[j], ring[j + 1]))
                return trueIfOnBoundary;
            if (rayIntersect(point, ring[j], ring[j + 1]))
                inside = !inside;
        }
    }
    return inside;
}
function pointWithinPolygons(point, polygons) {
    for (const polygon of polygons) {
        if (pointWithinPolygon(point, polygon))
            return true;
    }
    return false;
}
function lineStringWithinPolygon(line, polygon) {
    // First, check if geometry points of line segments are all inside polygon
    for (const point of line) {
        if (!pointWithinPolygon(point, polygon)) {
            return false;
        }
    }
    // Second, check if there is line segment intersecting polygon edge
    for (let i = 0; i < line.length - 1; ++i) {
        if (lineIntersectPolygon(line[i], line[i + 1], polygon)) {
            return false;
        }
    }
    return true;
}
function lineStringWithinPolygons(line, polygons) {
    for (const polygon of polygons) {
        if (lineStringWithinPolygon(line, polygon))
            return true;
    }
    return false;
}
function perp(v1, v2) {
    return (v1[0] * v2[1] - v1[1] * v2[0]);
}
// check if p1 and p2 are in different sides of line segment q1->q2
function twoSided(p1, p2, q1, q2) {
    // q1->p1 (x1, y1), q1->p2 (x2, y2), q1->q2 (x3, y3)
    const x1 = p1[0] - q1[0];
    const y1 = p1[1] - q1[1];
    const x2 = p2[0] - q1[0];
    const y2 = p2[1] - q1[1];
    const x3 = q2[0] - q1[0];
    const y3 = q2[1] - q1[1];
    const det1 = (x1 * y3 - x3 * y1);
    const det2 = (x2 * y3 - x3 * y2);
    if ((det1 > 0 && det2 < 0) || (det1 < 0 && det2 > 0))
        return true;
    return false;
}

function getTilePolygon(coordinates, bbox, canonical) {
    const polygon = [];
    for (let i = 0; i < coordinates.length; i++) {
        const ring = [];
        for (let j = 0; j < coordinates[i].length; j++) {
            const coord = getTileCoordinates(coordinates[i][j], canonical);
            updateBBox(bbox, coord);
            ring.push(coord);
        }
        polygon.push(ring);
    }
    return polygon;
}
function getTilePolygons(coordinates, bbox, canonical) {
    const polygons = [];
    for (let i = 0; i < coordinates.length; i++) {
        const polygon = getTilePolygon(coordinates[i], bbox, canonical);
        polygons.push(polygon);
    }
    return polygons;
}
function updatePoint(p, bbox, polyBBox, worldSize) {
    if (p[0] < polyBBox[0] || p[0] > polyBBox[2]) {
        const halfWorldSize = worldSize * 0.5;
        let shift = (p[0] - polyBBox[0] > halfWorldSize) ? -worldSize : (polyBBox[0] - p[0] > halfWorldSize) ? worldSize : 0;
        if (shift === 0) {
            shift = (p[0] - polyBBox[2] > halfWorldSize) ? -worldSize : (polyBBox[2] - p[0] > halfWorldSize) ? worldSize : 0;
        }
        p[0] += shift;
    }
    updateBBox(bbox, p);
}
function resetBBox(bbox) {
    bbox[0] = bbox[1] = Infinity;
    bbox[2] = bbox[3] = -Infinity;
}
function getTilePoints(geometry, pointBBox, polyBBox, canonical) {
    const worldSize = Math.pow(2, canonical.z) * EXTENT;
    const shifts = [canonical.x * EXTENT, canonical.y * EXTENT];
    const tilePoints = [];
    for (const points of geometry) {
        for (const point of points) {
            const p = [point.x + shifts[0], point.y + shifts[1]];
            updatePoint(p, pointBBox, polyBBox, worldSize);
            tilePoints.push(p);
        }
    }
    return tilePoints;
}
function getTileLines(geometry, lineBBox, polyBBox, canonical) {
    const worldSize = Math.pow(2, canonical.z) * EXTENT;
    const shifts = [canonical.x * EXTENT, canonical.y * EXTENT];
    const tileLines = [];
    for (const line of geometry) {
        const tileLine = [];
        for (const point of line) {
            const p = [point.x + shifts[0], point.y + shifts[1]];
            updateBBox(lineBBox, p);
            tileLine.push(p);
        }
        tileLines.push(tileLine);
    }
    if (lineBBox[2] - lineBBox[0] <= worldSize / 2) {
        resetBBox(lineBBox);
        for (const line of tileLines) {
            for (const p of line) {
                updatePoint(p, lineBBox, polyBBox, worldSize);
            }
        }
    }
    return tileLines;
}
function pointsWithinPolygons(ctx, polygonGeometry) {
    const pointBBox = [Infinity, Infinity, -Infinity, -Infinity];
    const polyBBox = [Infinity, Infinity, -Infinity, -Infinity];
    const canonical = ctx.canonicalID();
    if (polygonGeometry.type === 'Polygon') {
        const tilePolygon = getTilePolygon(polygonGeometry.coordinates, polyBBox, canonical);
        const tilePoints = getTilePoints(ctx.geometry(), pointBBox, polyBBox, canonical);
        if (!boxWithinBox(pointBBox, polyBBox))
            return false;
        for (const point of tilePoints) {
            if (!pointWithinPolygon(point, tilePolygon))
                return false;
        }
    }
    if (polygonGeometry.type === 'MultiPolygon') {
        const tilePolygons = getTilePolygons(polygonGeometry.coordinates, polyBBox, canonical);
        const tilePoints = getTilePoints(ctx.geometry(), pointBBox, polyBBox, canonical);
        if (!boxWithinBox(pointBBox, polyBBox))
            return false;
        for (const point of tilePoints) {
            if (!pointWithinPolygons(point, tilePolygons))
                return false;
        }
    }
    return true;
}
function linesWithinPolygons(ctx, polygonGeometry) {
    const lineBBox = [Infinity, Infinity, -Infinity, -Infinity];
    const polyBBox = [Infinity, Infinity, -Infinity, -Infinity];
    const canonical = ctx.canonicalID();
    if (polygonGeometry.type === 'Polygon') {
        const tilePolygon = getTilePolygon(polygonGeometry.coordinates, polyBBox, canonical);
        const tileLines = getTileLines(ctx.geometry(), lineBBox, polyBBox, canonical);
        if (!boxWithinBox(lineBBox, polyBBox))
            return false;
        for (const line of tileLines) {
            if (!lineStringWithinPolygon(line, tilePolygon))
                return false;
        }
    }
    if (polygonGeometry.type === 'MultiPolygon') {
        const tilePolygons = getTilePolygons(polygonGeometry.coordinates, polyBBox, canonical);
        const tileLines = getTileLines(ctx.geometry(), lineBBox, polyBBox, canonical);
        if (!boxWithinBox(lineBBox, polyBBox))
            return false;
        for (const line of tileLines) {
            if (!lineStringWithinPolygons(line, tilePolygons))
                return false;
        }
    }
    return true;
}
class Within {
    constructor(geojson, geometries) {
        this.type = BooleanType;
        this.geojson = geojson;
        this.geometries = geometries;
    }
    static parse(args, context) {
        if (args.length !== 2)
            return context.error(`'within' expression requires exactly one argument, but found ${args.length - 1} instead.`);
        if (isValue(args[1])) {
            const geojson = args[1];
            if (geojson.type === 'FeatureCollection') {
                const polygonsCoords = [];
                for (const polygon of geojson.features) {
                    const { type, coordinates } = polygon.geometry;
                    if (type === 'Polygon') {
                        polygonsCoords.push(coordinates);
                    }
                    if (type === 'MultiPolygon') {
                        polygonsCoords.push(...coordinates);
                    }
                }
                if (polygonsCoords.length) {
                    const multipolygonWrapper = {
                        type: 'MultiPolygon',
                        coordinates: polygonsCoords
                    };
                    return new Within(geojson, multipolygonWrapper);
                }
            }
            else if (geojson.type === 'Feature') {
                const type = geojson.geometry.type;
                if (type === 'Polygon' || type === 'MultiPolygon') {
                    return new Within(geojson, geojson.geometry);
                }
            }
            else if (geojson.type === 'Polygon' || geojson.type === 'MultiPolygon') {
                return new Within(geojson, geojson);
            }
        }
        return context.error('\'within\' expression requires valid geojson object that contains polygon geometry type.');
    }
    evaluate(ctx) {
        if (ctx.geometry() != null && ctx.canonicalID() != null) {
            if (ctx.geometryType() === 'Point') {
                return pointsWithinPolygons(ctx, this.geometries);
            }
            else if (ctx.geometryType() === 'LineString') {
                return linesWithinPolygons(ctx, this.geometries);
            }
        }
        return false;
    }
    eachChild() { }
    outputDefined() {
        return true;
    }
}

class TinyQueue {
    constructor(data = [], compare = (a, b) => (a < b ? -1 : a > b ? 1 : 0)) {
        this.data = data;
        this.length = this.data.length;
        this.compare = compare;

        if (this.length > 0) {
            for (let i = (this.length >> 1) - 1; i >= 0; i--) this._down(i);
        }
    }

    push(item) {
        this.data.push(item);
        this._up(this.length++);
    }

    pop() {
        if (this.length === 0) return undefined;

        const top = this.data[0];
        const bottom = this.data.pop();

        if (--this.length > 0) {
            this.data[0] = bottom;
            this._down(0);
        }

        return top;
    }

    peek() {
        return this.data[0];
    }

    _up(pos) {
        const {data, compare} = this;
        const item = data[pos];

        while (pos > 0) {
            const parent = (pos - 1) >> 1;
            const current = data[parent];
            if (compare(item, current) >= 0) break;
            data[pos] = current;
            pos = parent;
        }

        data[pos] = item;
    }

    _down(pos) {
        const {data, compare} = this;
        const halfLength = this.length >> 1;
        const item = data[pos];

        while (pos < halfLength) {
            let bestChild = (pos << 1) + 1; // initially it is the left child
            const right = bestChild + 1;

            if (right < this.length && compare(data[right], data[bestChild]) < 0) {
                bestChild = right;
            }
            if (compare(data[bestChild], item) >= 0) break;

            data[pos] = data[bestChild];
            pos = bestChild;
        }

        data[pos] = item;
    }
}

/**
 * Classifies an array of rings into polygons with outer rings and holes
 * @param rings - the rings to classify
 * @param maxRings - the maximum number of rings to include in a polygon, use 0 to include all rings
 * @returns an array of polygons with internal rings as holes
 */
function classifyRings(rings, maxRings) {
    const len = rings.length;
    if (len <= 1)
        return [rings];
    const polygons = [];
    let polygon;
    let ccw;
    for (const ring of rings) {
        const area = calculateSignedArea(ring);
        if (area === 0)
            continue;
        ring.area = Math.abs(area);
        if (ccw === undefined)
            ccw = area < 0;
        if (ccw === area < 0) {
            if (polygon)
                polygons.push(polygon);
            polygon = [ring];
        }
        else {
            polygon.push(ring);
        }
    }
    if (polygon)
        polygons.push(polygon);
    return polygons;
}
/**
 * Returns the signed area for the polygon ring.  Positive areas are exterior rings and
 * have a clockwise winding.  Negative areas are interior rings and have a counter clockwise
 * ordering.
 *
 * @param ring - Exterior or interior ring
 * @returns Signed area
 */
function calculateSignedArea(ring) {
    let sum = 0;
    for (let i = 0, len = ring.length, j = len - 1, p1, p2; i < len; j = i++) {
        p1 = ring[i];
        p2 = ring[j];
        sum += (p2.x - p1.x) * (p1.y + p2.y);
    }
    return sum;
}

// This is taken from https://github.com/mapbox/cheap-ruler/ in order to take only the relevant parts
// Values that define WGS84 ellipsoid model of the Earth
const RE = 6378.137; // equatorial radius
const FE = 1 / 298.257223563; // flattening
const E2 = FE * (2 - FE);
const RAD = Math.PI / 180;
class CheapRuler {
    constructor(lat) {
        // Curvature formulas from https://en.wikipedia.org/wiki/Earth_radius#Meridional
        const m = RAD * RE * 1000;
        const coslat = Math.cos(lat * RAD);
        const w2 = 1 / (1 - E2 * (1 - coslat * coslat));
        const w = Math.sqrt(w2);
        // multipliers for converting longitude and latitude degrees into distance
        this.kx = m * w * coslat; // based on normal radius of curvature
        this.ky = m * w * w2 * (1 - E2); // based on meridional radius of curvature
    }
    /**
     * Given two points of the form [longitude, latitude], returns the distance.
     *
     * @param a - point [longitude, latitude]
     * @param b - point [longitude, latitude]
     * @returns distance
     * @example
     * const distance = ruler.distance([30.5, 50.5], [30.51, 50.49]);
     * //=distance
     */
    distance(a, b) {
        const dx = this.wrap(a[0] - b[0]) * this.kx;
        const dy = (a[1] - b[1]) * this.ky;
        return Math.sqrt(dx * dx + dy * dy);
    }
    /**
     * Returns an object of the form {point, index, t}, where point is closest point on the line
     * from the given point, index is the start index of the segment with the closest point,
     * and t is a parameter from 0 to 1 that indicates where the closest point is on that segment.
     *
     * @param line - an array of points that form the line
     * @param p - point [longitude, latitude]
     * @returns the nearest point, its index in the array and the proportion along the line
     * @example
     * const point = ruler.pointOnLine(line, [-67.04, 50.5]).point;
     * //=point
     */
    pointOnLine(line, p) {
        let minDist = Infinity;
        let minX, minY, minI, minT;
        for (let i = 0; i < line.length - 1; i++) {
            let x = line[i][0];
            let y = line[i][1];
            let dx = this.wrap(line[i + 1][0] - x) * this.kx;
            let dy = (line[i + 1][1] - y) * this.ky;
            let t = 0;
            if (dx !== 0 || dy !== 0) {
                t = (this.wrap(p[0] - x) * this.kx * dx + (p[1] - y) * this.ky * dy) / (dx * dx + dy * dy);
                if (t > 1) {
                    x = line[i + 1][0];
                    y = line[i + 1][1];
                }
                else if (t > 0) {
                    x += (dx / this.kx) * t;
                    y += (dy / this.ky) * t;
                }
            }
            dx = this.wrap(p[0] - x) * this.kx;
            dy = (p[1] - y) * this.ky;
            const sqDist = dx * dx + dy * dy;
            if (sqDist < minDist) {
                minDist = sqDist;
                minX = x;
                minY = y;
                minI = i;
                minT = t;
            }
        }
        return {
            point: [minX, minY],
            index: minI,
            t: Math.max(0, Math.min(1, minT))
        };
    }
    wrap(deg) {
        while (deg < -180)
            deg += 360;
        while (deg > 180)
            deg -= 360;
        return deg;
    }
}

const MinPointsSize = 100;
const MinLinePointsSize = 50;
function compareDistPair(a, b) {
    return b[0] - a[0];
}
function getRangeSize(range) {
    return range[1] - range[0] + 1;
}
function isRangeSafe(range, threshold) {
    return range[1] >= range[0] && range[1] < threshold;
}
function splitRange(range, isLine) {
    if (range[0] > range[1]) {
        return [null, null];
    }
    const size = getRangeSize(range);
    if (isLine) {
        if (size === 2) {
            return [range, null];
        }
        const size1 = Math.floor(size / 2);
        return [[range[0], range[0] + size1],
            [range[0] + size1, range[1]]];
    }
    if (size === 1) {
        return [range, null];
    }
    const size1 = Math.floor(size / 2) - 1;
    return [[range[0], range[0] + size1],
        [range[0] + size1 + 1, range[1]]];
}
function getBBox(coords, range) {
    if (!isRangeSafe(range, coords.length)) {
        return [Infinity, Infinity, -Infinity, -Infinity];
    }
    const bbox = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = range[0]; i <= range[1]; ++i) {
        updateBBox(bbox, coords[i]);
    }
    return bbox;
}
function getPolygonBBox(polygon) {
    const bbox = [Infinity, Infinity, -Infinity, -Infinity];
    for (const ring of polygon) {
        for (const coord of ring) {
            updateBBox(bbox, coord);
        }
    }
    return bbox;
}
function isValidBBox(bbox) {
    return bbox[0] !== -Infinity && bbox[1] !== -Infinity && bbox[2] !== Infinity && bbox[3] !== Infinity;
}
// Calculate the distance between two bounding boxes.
// Calculate the delta in x and y direction, and use two fake points {0.0, 0.0}
// and {dx, dy} to calculate the distance. Distance will be 0.0 if bounding box are overlapping.
function bboxToBBoxDistance(bbox1, bbox2, ruler) {
    if (!isValidBBox(bbox1) || !isValidBBox(bbox2)) {
        return NaN;
    }
    let dx = 0.0;
    let dy = 0.0;
    // bbox1 in left side
    if (bbox1[2] < bbox2[0]) {
        dx = bbox2[0] - bbox1[2];
    }
    // bbox1 in right side
    if (bbox1[0] > bbox2[2]) {
        dx = bbox1[0] - bbox2[2];
    }
    // bbox1 in above side
    if (bbox1[1] > bbox2[3]) {
        dy = bbox1[1] - bbox2[3];
    }
    // bbox1 in down side
    if (bbox1[3] < bbox2[1]) {
        dy = bbox2[1] - bbox1[3];
    }
    return ruler.distance([0.0, 0.0], [dx, dy]);
}
function pointToLineDistance(point, line, ruler) {
    const nearestPoint = ruler.pointOnLine(line, point);
    return ruler.distance(point, nearestPoint.point);
}
function segmentToSegmentDistance(p1, p2, q1, q2, ruler) {
    const dist1 = Math.min(pointToLineDistance(p1, [q1, q2], ruler), pointToLineDistance(p2, [q1, q2], ruler));
    const dist2 = Math.min(pointToLineDistance(q1, [p1, p2], ruler), pointToLineDistance(q2, [p1, p2], ruler));
    return Math.min(dist1, dist2);
}
function lineToLineDistance(line1, range1, line2, range2, ruler) {
    const rangeSafe = isRangeSafe(range1, line1.length) && isRangeSafe(range2, line2.length);
    if (!rangeSafe) {
        return Infinity;
    }
    let dist = Infinity;
    for (let i = range1[0]; i < range1[1]; ++i) {
        const p1 = line1[i];
        const p2 = line1[i + 1];
        for (let j = range2[0]; j < range2[1]; ++j) {
            const q1 = line2[j];
            const q2 = line2[j + 1];
            if (segmentIntersectSegment(p1, p2, q1, q2)) {
                return 0.0;
            }
            dist = Math.min(dist, segmentToSegmentDistance(p1, p2, q1, q2, ruler));
        }
    }
    return dist;
}
function pointsToPointsDistance(points1, range1, points2, range2, ruler) {
    const rangeSafe = isRangeSafe(range1, points1.length) && isRangeSafe(range2, points2.length);
    if (!rangeSafe) {
        return NaN;
    }
    let dist = Infinity;
    for (let i = range1[0]; i <= range1[1]; ++i) {
        for (let j = range2[0]; j <= range2[1]; ++j) {
            dist = Math.min(dist, ruler.distance(points1[i], points2[j]));
            if (dist === 0.0) {
                return dist;
            }
        }
    }
    return dist;
}
function pointToPolygonDistance(point, polygon, ruler) {
    if (pointWithinPolygon(point, polygon, true)) {
        return 0.0;
    }
    let dist = Infinity;
    for (const ring of polygon) {
        const front = ring[0];
        const back = ring[ring.length - 1];
        if (front !== back) {
            dist = Math.min(dist, pointToLineDistance(point, [back, front], ruler));
            if (dist === 0.0) {
                return dist;
            }
        }
        const nearestPoint = ruler.pointOnLine(ring, point);
        dist = Math.min(dist, ruler.distance(point, nearestPoint.point));
        if (dist === 0.0) {
            return dist;
        }
    }
    return dist;
}
function lineToPolygonDistance(line, range, polygon, ruler) {
    if (!isRangeSafe(range, line.length)) {
        return NaN;
    }
    for (let i = range[0]; i <= range[1]; ++i) {
        if (pointWithinPolygon(line[i], polygon, true)) {
            return 0.0;
        }
    }
    let dist = Infinity;
    for (let i = range[0]; i < range[1]; ++i) {
        const p1 = line[i];
        const p2 = line[i + 1];
        for (const ring of polygon) {
            for (let j = 0, len = ring.length, k = len - 1; j < len; k = j++) {
                const q1 = ring[k];
                const q2 = ring[j];
                if (segmentIntersectSegment(p1, p2, q1, q2)) {
                    return 0.0;
                }
                dist = Math.min(dist, segmentToSegmentDistance(p1, p2, q1, q2, ruler));
            }
        }
    }
    return dist;
}
function polygonIntersect(poly1, poly2) {
    for (const ring of poly1) {
        for (const point of ring) {
            if (pointWithinPolygon(point, poly2, true)) {
                return true;
            }
        }
    }
    return false;
}
function polygonToPolygonDistance(polygon1, polygon2, ruler, currentMiniDist = Infinity) {
    const bbox1 = getPolygonBBox(polygon1);
    const bbox2 = getPolygonBBox(polygon2);
    if (currentMiniDist !== Infinity && bboxToBBoxDistance(bbox1, bbox2, ruler) >= currentMiniDist) {
        return currentMiniDist;
    }
    if (boxWithinBox(bbox1, bbox2)) {
        if (polygonIntersect(polygon1, polygon2)) {
            return 0.0;
        }
    }
    else if (polygonIntersect(polygon2, polygon1)) {
        return 0.0;
    }
    let dist = Infinity;
    for (const ring1 of polygon1) {
        for (let i = 0, len1 = ring1.length, l = len1 - 1; i < len1; l = i++) {
            const p1 = ring1[l];
            const p2 = ring1[i];
            for (const ring2 of polygon2) {
                for (let j = 0, len2 = ring2.length, k = len2 - 1; j < len2; k = j++) {
                    const q1 = ring2[k];
                    const q2 = ring2[j];
                    if (segmentIntersectSegment(p1, p2, q1, q2)) {
                        return 0.0;
                    }
                    dist = Math.min(dist, segmentToSegmentDistance(p1, p2, q1, q2, ruler));
                }
            }
        }
    }
    return dist;
}
function updateQueue(distQueue, miniDist, ruler, points, polyBBox, rangeA) {
    if (!rangeA) {
        return;
    }
    const tempDist = bboxToBBoxDistance(getBBox(points, rangeA), polyBBox, ruler);
    // Insert new pair to the queue if the bbox distance is less than
    // miniDist, The pair with biggest distance will be at the top
    if (tempDist < miniDist) {
        distQueue.push([tempDist, rangeA, [0, 0]]);
    }
}
function updateQueueTwoSets(distQueue, miniDist, ruler, pointSet1, pointSet2, range1, range2) {
    if (!range1 || !range2) {
        return;
    }
    const tempDist = bboxToBBoxDistance(getBBox(pointSet1, range1), getBBox(pointSet2, range2), ruler);
    // Insert new pair to the queue if the bbox distance is less than
    // miniDist, The pair with biggest distance will be at the top
    if (tempDist < miniDist) {
        distQueue.push([tempDist, range1, range2]);
    }
}
// Divide and conquer, the time complexity is O(n*lgn), faster than Brute force
// O(n*n) Most of the time, use index for in-place processing.
function pointsToPolygonDistance(points, isLine, polygon, ruler, currentMiniDist = Infinity) {
    let miniDist = Math.min(ruler.distance(points[0], polygon[0][0]), currentMiniDist);
    if (miniDist === 0.0) {
        return miniDist;
    }
    const distQueue = new TinyQueue([[0, [0, points.length - 1], [0, 0]]], compareDistPair);
    const polyBBox = getPolygonBBox(polygon);
    while (distQueue.length > 0) {
        const distPair = distQueue.pop();
        if (distPair[0] >= miniDist) {
            continue;
        }
        const range = distPair[1];
        // In case the set size are relatively small, we could use brute-force directly
        const threshold = isLine ? MinLinePointsSize : MinPointsSize;
        if (getRangeSize(range) <= threshold) {
            if (!isRangeSafe(range, points.length)) {
                return NaN;
            }
            if (isLine) {
                const tempDist = lineToPolygonDistance(points, range, polygon, ruler);
                if (isNaN(tempDist) || tempDist === 0.0) {
                    return tempDist;
                }
                miniDist = Math.min(miniDist, tempDist);
            }
            else {
                for (let i = range[0]; i <= range[1]; ++i) {
                    const tempDist = pointToPolygonDistance(points[i], polygon, ruler);
                    miniDist = Math.min(miniDist, tempDist);
                    if (miniDist === 0.0) {
                        return 0.0;
                    }
                }
            }
        }
        else {
            const newRangesA = splitRange(range, isLine);
            updateQueue(distQueue, miniDist, ruler, points, polyBBox, newRangesA[0]);
            updateQueue(distQueue, miniDist, ruler, points, polyBBox, newRangesA[1]);
        }
    }
    return miniDist;
}
function pointSetToPointSetDistance(pointSet1, isLine1, pointSet2, isLine2, ruler, currentMiniDist = Infinity) {
    let miniDist = Math.min(currentMiniDist, ruler.distance(pointSet1[0], pointSet2[0]));
    if (miniDist === 0.0) {
        return miniDist;
    }
    const distQueue = new TinyQueue([[0, [0, pointSet1.length - 1], [0, pointSet2.length - 1]]], compareDistPair);
    while (distQueue.length > 0) {
        const distPair = distQueue.pop();
        if (distPair[0] >= miniDist) {
            continue;
        }
        const rangeA = distPair[1];
        const rangeB = distPair[2];
        const threshold1 = isLine1 ? MinLinePointsSize : MinPointsSize;
        const threshold2 = isLine2 ? MinLinePointsSize : MinPointsSize;
        // In case the set size are relatively small, we could use brute-force directly
        if (getRangeSize(rangeA) <= threshold1 && getRangeSize(rangeB) <= threshold2) {
            if (!isRangeSafe(rangeA, pointSet1.length) && isRangeSafe(rangeB, pointSet2.length)) {
                return NaN;
            }
            let tempDist;
            if (isLine1 && isLine2) {
                tempDist = lineToLineDistance(pointSet1, rangeA, pointSet2, rangeB, ruler);
                miniDist = Math.min(miniDist, tempDist);
            }
            else if (isLine1 && !isLine2) {
                const sublibe = pointSet1.slice(rangeA[0], rangeA[1] + 1);
                for (let i = rangeB[0]; i <= rangeB[1]; ++i) {
                    tempDist = pointToLineDistance(pointSet2[i], sublibe, ruler);
                    miniDist = Math.min(miniDist, tempDist);
                    if (miniDist === 0.0) {
                        return miniDist;
                    }
                }
            }
            else if (!isLine1 && isLine2) {
                const sublibe = pointSet2.slice(rangeB[0], rangeB[1] + 1);
                for (let i = rangeA[0]; i <= rangeA[1]; ++i) {
                    tempDist = pointToLineDistance(pointSet1[i], sublibe, ruler);
                    miniDist = Math.min(miniDist, tempDist);
                    if (miniDist === 0.0) {
                        return miniDist;
                    }
                }
            }
            else {
                tempDist = pointsToPointsDistance(pointSet1, rangeA, pointSet2, rangeB, ruler);
                miniDist = Math.min(miniDist, tempDist);
            }
        }
        else {
            const newRangesA = splitRange(rangeA, isLine1);
            const newRangesB = splitRange(rangeB, isLine2);
            updateQueueTwoSets(distQueue, miniDist, ruler, pointSet1, pointSet2, newRangesA[0], newRangesB[0]);
            updateQueueTwoSets(distQueue, miniDist, ruler, pointSet1, pointSet2, newRangesA[0], newRangesB[1]);
            updateQueueTwoSets(distQueue, miniDist, ruler, pointSet1, pointSet2, newRangesA[1], newRangesB[0]);
            updateQueueTwoSets(distQueue, miniDist, ruler, pointSet1, pointSet2, newRangesA[1], newRangesB[1]);
        }
    }
    return miniDist;
}
function pointToGeometryDistance(ctx, geometries) {
    const tilePoints = ctx.geometry();
    const pointPosition = tilePoints.flat().map(p => getLngLatFromTileCoord([p.x, p.y], ctx.canonical));
    if (tilePoints.length === 0) {
        return NaN;
    }
    const ruler = new CheapRuler(pointPosition[0][1]);
    let dist = Infinity;
    for (const geometry of geometries) {
        switch (geometry.type) {
            case 'Point':
                dist = Math.min(dist, pointSetToPointSetDistance(pointPosition, false, [geometry.coordinates], false, ruler, dist));
                break;
            case 'LineString':
                dist = Math.min(dist, pointSetToPointSetDistance(pointPosition, false, geometry.coordinates, true, ruler, dist));
                break;
            case 'Polygon':
                dist = Math.min(dist, pointsToPolygonDistance(pointPosition, false, geometry.coordinates, ruler, dist));
                break;
        }
        if (dist === 0.0) {
            return dist;
        }
    }
    return dist;
}
function lineStringToGeometryDistance(ctx, geometries) {
    const tileLine = ctx.geometry();
    const linePositions = tileLine.flat().map(p => getLngLatFromTileCoord([p.x, p.y], ctx.canonical));
    if (tileLine.length === 0) {
        return NaN;
    }
    const ruler = new CheapRuler(linePositions[0][1]);
    let dist = Infinity;
    for (const geometry of geometries) {
        switch (geometry.type) {
            case 'Point':
                dist = Math.min(dist, pointSetToPointSetDistance(linePositions, true, [geometry.coordinates], false, ruler, dist));
                break;
            case 'LineString':
                dist = Math.min(dist, pointSetToPointSetDistance(linePositions, true, geometry.coordinates, true, ruler, dist));
                break;
            case 'Polygon':
                dist = Math.min(dist, pointsToPolygonDistance(linePositions, true, geometry.coordinates, ruler, dist));
                break;
        }
        if (dist === 0.0) {
            return dist;
        }
    }
    return dist;
}
function polygonToGeometryDistance(ctx, geometries) {
    const tilePolygon = ctx.geometry();
    if (tilePolygon.length === 0 || tilePolygon[0].length === 0) {
        return NaN;
    }
    const polygons = classifyRings(tilePolygon).map(polygon => {
        return polygon.map(ring => {
            return ring.map(p => getLngLatFromTileCoord([p.x, p.y], ctx.canonical));
        });
    });
    const ruler = new CheapRuler(polygons[0][0][0][1]);
    let dist = Infinity;
    for (const geometry of geometries) {
        for (const polygon of polygons) {
            switch (geometry.type) {
                case 'Point':
                    dist = Math.min(dist, pointsToPolygonDistance([geometry.coordinates], false, polygon, ruler, dist));
                    break;
                case 'LineString':
                    dist = Math.min(dist, pointsToPolygonDistance(geometry.coordinates, true, polygon, ruler, dist));
                    break;
                case 'Polygon':
                    dist = Math.min(dist, polygonToPolygonDistance(polygon, geometry.coordinates, ruler, dist));
                    break;
            }
            if (dist === 0.0) {
                return dist;
            }
        }
    }
    return dist;
}
function toSimpleGeometry(geometry) {
    if (geometry.type === 'MultiPolygon') {
        return geometry.coordinates.map(polygon => {
            return {
                type: 'Polygon',
                coordinates: polygon
            };
        });
    }
    if (geometry.type === 'MultiLineString') {
        return geometry.coordinates.map(lineString => {
            return {
                type: 'LineString',
                coordinates: lineString
            };
        });
    }
    if (geometry.type === 'MultiPoint') {
        return geometry.coordinates.map(point => {
            return {
                type: 'Point',
                coordinates: point
            };
        });
    }
    return [geometry];
}
class Distance {
    constructor(geojson, geometries) {
        this.type = NumberType;
        this.geojson = geojson;
        this.geometries = geometries;
    }
    static parse(args, context) {
        if (args.length !== 2)
            return context.error(`'distance' expression requires exactly one argument, but found ${args.length - 1} instead.`);
        if (isValue(args[1])) {
            const geojson = args[1];
            if (geojson.type === 'FeatureCollection') {
                return new Distance(geojson, geojson.features.map(feature => toSimpleGeometry(feature.geometry)).flat());
            }
            else if (geojson.type === 'Feature') {
                return new Distance(geojson, toSimpleGeometry(geojson.geometry));
            }
            else if ('type' in geojson && 'coordinates' in geojson) {
                return new Distance(geojson, toSimpleGeometry(geojson));
            }
        }
        return context.error('\'distance\' expression requires valid geojson object that contains polygon geometry type.');
    }
    evaluate(ctx) {
        if (ctx.geometry() != null && ctx.canonicalID() != null) {
            if (ctx.geometryType() === 'Point') {
                return pointToGeometryDistance(ctx, this.geometries);
            }
            else if (ctx.geometryType() === 'LineString') {
                return lineStringToGeometryDistance(ctx, this.geometries);
            }
            else if (ctx.geometryType() === 'Polygon') {
                return polygonToGeometryDistance(ctx, this.geometries);
            }
        }
        return NaN;
    }
    eachChild() { }
    outputDefined() {
        return true;
    }
}

const expressions = {
    // special forms
    '==': Equals,
    '!=': NotEquals,
    '>': GreaterThan,
    '<': LessThan,
    '>=': GreaterThanOrEqual,
    '<=': LessThanOrEqual,
    'array': Assertion,
    'at': At,
    'boolean': Assertion,
    'case': Case,
    'coalesce': Coalesce,
    'collator': CollatorExpression,
    'format': FormatExpression,
    'image': ImageExpression,
    'in': In,
    'index-of': IndexOf,
    'interpolate': Interpolate,
    'interpolate-hcl': Interpolate,
    'interpolate-lab': Interpolate,
    'length': Length,
    'let': Let,
    'literal': Literal,
    'match': Match,
    'number': Assertion,
    'number-format': NumberFormat,
    'object': Assertion,
    'slice': Slice,
    'step': Step,
    'string': Assertion,
    'to-boolean': Coercion,
    'to-color': Coercion,
    'to-number': Coercion,
    'to-string': Coercion,
    'var': Var,
    'within': Within,
    'distance': Distance
};

class CompoundExpression {
    constructor(name, type, evaluate, args) {
        this.name = name;
        this.type = type;
        this._evaluate = evaluate;
        this.args = args;
    }
    evaluate(ctx) {
        return this._evaluate(ctx, this.args);
    }
    eachChild(fn) {
        this.args.forEach(fn);
    }
    outputDefined() {
        return false;
    }
    static parse(args, context) {
        const op = args[0];
        const definition = CompoundExpression.definitions[op];
        if (!definition) {
            return context.error(`Unknown expression "${op}". If you wanted a literal array, use ["literal", [...]].`, 0);
        }
        // Now check argument types against each signature
        const type = Array.isArray(definition) ?
            definition[0] : definition.type;
        const availableOverloads = Array.isArray(definition) ?
            [[definition[1], definition[2]]] :
            definition.overloads;
        const overloads = availableOverloads.filter(([signature]) => (!Array.isArray(signature) || // varags
            signature.length === args.length - 1 // correct param count
        ));
        let signatureContext = null;
        for (const [params, evaluate] of overloads) {
            // Use a fresh context for each attempted signature so that, if
            // we eventually succeed, we haven't polluted `context.errors`.
            signatureContext = new ParsingContext(context.registry, isExpressionConstant, context.path, null, context.scope);
            // First parse all the args, potentially coercing to the
            // types expected by this overload.
            const parsedArgs = [];
            let argParseFailed = false;
            for (let i = 1; i < args.length; i++) {
                const arg = args[i];
                const expectedType = Array.isArray(params) ?
                    params[i - 1] :
                    params.type;
                const parsed = signatureContext.parse(arg, 1 + parsedArgs.length, expectedType);
                if (!parsed) {
                    argParseFailed = true;
                    break;
                }
                parsedArgs.push(parsed);
            }
            if (argParseFailed) {
                // Couldn't coerce args of this overload to expected type, move
                // on to next one.
                continue;
            }
            if (Array.isArray(params)) {
                if (params.length !== parsedArgs.length) {
                    signatureContext.error(`Expected ${params.length} arguments, but found ${parsedArgs.length} instead.`);
                    continue;
                }
            }
            for (let i = 0; i < parsedArgs.length; i++) {
                const expected = Array.isArray(params) ? params[i] : params.type;
                const arg = parsedArgs[i];
                signatureContext.concat(i + 1).checkSubtype(expected, arg.type);
            }
            if (signatureContext.errors.length === 0) {
                return new CompoundExpression(op, type, evaluate, parsedArgs);
            }
        }
        if (overloads.length === 1) {
            context.errors.push(...signatureContext.errors);
        }
        else {
            const expected = overloads.length ? overloads : availableOverloads;
            const signatures = expected
                .map(([params]) => stringifySignature(params))
                .join(' | ');
            const actualTypes = [];
            // For error message, re-parse arguments without trying to
            // apply any coercions
            for (let i = 1; i < args.length; i++) {
                const parsed = context.parse(args[i], 1 + actualTypes.length);
                if (!parsed)
                    return null;
                actualTypes.push(toString$1(parsed.type));
            }
            context.error(`Expected arguments of type ${signatures}, but found (${actualTypes.join(', ')}) instead.`);
        }
        return null;
    }
    static register(registry, definitions) {
        CompoundExpression.definitions = definitions;
        for (const name in definitions) {
            registry[name] = CompoundExpression;
        }
    }
}
function rgba(ctx, [r, g, b, a]) {
    r = r.evaluate(ctx);
    g = g.evaluate(ctx);
    b = b.evaluate(ctx);
    const alpha = a ? a.evaluate(ctx) : 1;
    const error = validateRGBA(r, g, b, alpha);
    if (error)
        throw new RuntimeError(error);
    return new Color(r / 255, g / 255, b / 255, alpha, false);
}
function has(key, obj) {
    return key in obj;
}
function get(key, obj) {
    const v = obj[key];
    return typeof v === 'undefined' ? null : v;
}
function binarySearch(v, a, i, j) {
    while (i <= j) {
        const m = (i + j) >> 1;
        if (a[m] === v)
            return true;
        if (a[m] > v)
            j = m - 1;
        else
            i = m + 1;
    }
    return false;
}
function varargs(type) {
    return { type };
}
CompoundExpression.register(expressions, {
    'error': [
        ErrorType,
        [StringType],
        (ctx, [v]) => { throw new RuntimeError(v.evaluate(ctx)); }
    ],
    'typeof': [
        StringType,
        [ValueType],
        (ctx, [v]) => toString$1(typeOf(v.evaluate(ctx)))
    ],
    'to-rgba': [
        array$1(NumberType, 4),
        [ColorType],
        (ctx, [v]) => {
            const [r, g, b, a] = v.evaluate(ctx).rgb;
            return [r * 255, g * 255, b * 255, a];
        },
    ],
    'rgb': [
        ColorType,
        [NumberType, NumberType, NumberType],
        rgba
    ],
    'rgba': [
        ColorType,
        [NumberType, NumberType, NumberType, NumberType],
        rgba
    ],
    'has': {
        type: BooleanType,
        overloads: [
            [
                [StringType],
                (ctx, [key]) => has(key.evaluate(ctx), ctx.properties())
            ], [
                [StringType, ObjectType],
                (ctx, [key, obj]) => has(key.evaluate(ctx), obj.evaluate(ctx))
            ]
        ]
    },
    'get': {
        type: ValueType,
        overloads: [
            [
                [StringType],
                (ctx, [key]) => get(key.evaluate(ctx), ctx.properties())
            ], [
                [StringType, ObjectType],
                (ctx, [key, obj]) => get(key.evaluate(ctx), obj.evaluate(ctx))
            ]
        ]
    },
    'feature-state': [
        ValueType,
        [StringType],
        (ctx, [key]) => get(key.evaluate(ctx), ctx.featureState || {})
    ],
    'properties': [
        ObjectType,
        [],
        (ctx) => ctx.properties()
    ],
    'geometry-type': [
        StringType,
        [],
        (ctx) => ctx.geometryType()
    ],
    'id': [
        ValueType,
        [],
        (ctx) => ctx.id()
    ],
    'zoom': [
        NumberType,
        [],
        (ctx) => ctx.globals.zoom
    ],
    'heatmap-density': [
        NumberType,
        [],
        (ctx) => ctx.globals.heatmapDensity || 0
    ],
    'line-progress': [
        NumberType,
        [],
        (ctx) => ctx.globals.lineProgress || 0
    ],
    'accumulated': [
        ValueType,
        [],
        (ctx) => ctx.globals.accumulated === undefined ? null : ctx.globals.accumulated
    ],
    '+': [
        NumberType,
        varargs(NumberType),
        (ctx, args) => {
            let result = 0;
            for (const arg of args) {
                result += arg.evaluate(ctx);
            }
            return result;
        }
    ],
    '*': [
        NumberType,
        varargs(NumberType),
        (ctx, args) => {
            let result = 1;
            for (const arg of args) {
                result *= arg.evaluate(ctx);
            }
            return result;
        }
    ],
    '-': {
        type: NumberType,
        overloads: [
            [
                [NumberType, NumberType],
                (ctx, [a, b]) => a.evaluate(ctx) - b.evaluate(ctx)
            ], [
                [NumberType],
                (ctx, [a]) => -a.evaluate(ctx)
            ]
        ]
    },
    '/': [
        NumberType,
        [NumberType, NumberType],
        (ctx, [a, b]) => a.evaluate(ctx) / b.evaluate(ctx)
    ],
    '%': [
        NumberType,
        [NumberType, NumberType],
        (ctx, [a, b]) => a.evaluate(ctx) % b.evaluate(ctx)
    ],
    'ln2': [
        NumberType,
        [],
        () => Math.LN2
    ],
    'pi': [
        NumberType,
        [],
        () => Math.PI
    ],
    'e': [
        NumberType,
        [],
        () => Math.E
    ],
    '^': [
        NumberType,
        [NumberType, NumberType],
        (ctx, [b, e]) => Math.pow(b.evaluate(ctx), e.evaluate(ctx))
    ],
    'sqrt': [
        NumberType,
        [NumberType],
        (ctx, [x]) => Math.sqrt(x.evaluate(ctx))
    ],
    'log10': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.log(n.evaluate(ctx)) / Math.LN10
    ],
    'ln': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.log(n.evaluate(ctx))
    ],
    'log2': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.log(n.evaluate(ctx)) / Math.LN2
    ],
    'sin': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.sin(n.evaluate(ctx))
    ],
    'cos': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.cos(n.evaluate(ctx))
    ],
    'tan': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.tan(n.evaluate(ctx))
    ],
    'asin': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.asin(n.evaluate(ctx))
    ],
    'acos': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.acos(n.evaluate(ctx))
    ],
    'atan': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.atan(n.evaluate(ctx))
    ],
    'min': [
        NumberType,
        varargs(NumberType),
        (ctx, args) => Math.min(...args.map(arg => arg.evaluate(ctx)))
    ],
    'max': [
        NumberType,
        varargs(NumberType),
        (ctx, args) => Math.max(...args.map(arg => arg.evaluate(ctx)))
    ],
    'abs': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.abs(n.evaluate(ctx))
    ],
    'round': [
        NumberType,
        [NumberType],
        (ctx, [n]) => {
            const v = n.evaluate(ctx);
            // Javascript's Math.round() rounds towards +Infinity for halfway
            // values, even when they're negative. It's more common to round
            // away from 0 (e.g., this is what python and C++ do)
            return v < 0 ? -Math.round(-v) : Math.round(v);
        }
    ],
    'floor': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.floor(n.evaluate(ctx))
    ],
    'ceil': [
        NumberType,
        [NumberType],
        (ctx, [n]) => Math.ceil(n.evaluate(ctx))
    ],
    'filter-==': [
        BooleanType,
        [StringType, ValueType],
        (ctx, [k, v]) => ctx.properties()[k.value] === v.value
    ],
    'filter-id-==': [
        BooleanType,
        [ValueType],
        (ctx, [v]) => ctx.id() === v.value
    ],
    'filter-type-==': [
        BooleanType,
        [StringType],
        (ctx, [v]) => ctx.geometryType() === v.value
    ],
    'filter-<': [
        BooleanType,
        [StringType, ValueType],
        (ctx, [k, v]) => {
            const a = ctx.properties()[k.value];
            const b = v.value;
            return typeof a === typeof b && a < b;
        }
    ],
    'filter-id-<': [
        BooleanType,
        [ValueType],
        (ctx, [v]) => {
            const a = ctx.id();
            const b = v.value;
            return typeof a === typeof b && a < b;
        }
    ],
    'filter->': [
        BooleanType,
        [StringType, ValueType],
        (ctx, [k, v]) => {
            const a = ctx.properties()[k.value];
            const b = v.value;
            return typeof a === typeof b && a > b;
        }
    ],
    'filter-id->': [
        BooleanType,
        [ValueType],
        (ctx, [v]) => {
            const a = ctx.id();
            const b = v.value;
            return typeof a === typeof b && a > b;
        }
    ],
    'filter-<=': [
        BooleanType,
        [StringType, ValueType],
        (ctx, [k, v]) => {
            const a = ctx.properties()[k.value];
            const b = v.value;
            return typeof a === typeof b && a <= b;
        }
    ],
    'filter-id-<=': [
        BooleanType,
        [ValueType],
        (ctx, [v]) => {
            const a = ctx.id();
            const b = v.value;
            return typeof a === typeof b && a <= b;
        }
    ],
    'filter->=': [
        BooleanType,
        [StringType, ValueType],
        (ctx, [k, v]) => {
            const a = ctx.properties()[k.value];
            const b = v.value;
            return typeof a === typeof b && a >= b;
        }
    ],
    'filter-id->=': [
        BooleanType,
        [ValueType],
        (ctx, [v]) => {
            const a = ctx.id();
            const b = v.value;
            return typeof a === typeof b && a >= b;
        }
    ],
    'filter-has': [
        BooleanType,
        [ValueType],
        (ctx, [k]) => k.value in ctx.properties()
    ],
    'filter-has-id': [
        BooleanType,
        [],
        (ctx) => (ctx.id() !== null && ctx.id() !== undefined)
    ],
    'filter-type-in': [
        BooleanType,
        [array$1(StringType)],
        (ctx, [v]) => v.value.indexOf(ctx.geometryType()) >= 0
    ],
    'filter-id-in': [
        BooleanType,
        [array$1(ValueType)],
        (ctx, [v]) => v.value.indexOf(ctx.id()) >= 0
    ],
    'filter-in-small': [
        BooleanType,
        [StringType, array$1(ValueType)],
        // assumes v is an array literal
        (ctx, [k, v]) => v.value.indexOf(ctx.properties()[k.value]) >= 0
    ],
    'filter-in-large': [
        BooleanType,
        [StringType, array$1(ValueType)],
        // assumes v is a array literal with values sorted in ascending order and of a single type
        (ctx, [k, v]) => binarySearch(ctx.properties()[k.value], v.value, 0, v.value.length - 1)
    ],
    'all': {
        type: BooleanType,
        overloads: [
            [
                [BooleanType, BooleanType],
                (ctx, [a, b]) => a.evaluate(ctx) && b.evaluate(ctx)
            ],
            [
                varargs(BooleanType),
                (ctx, args) => {
                    for (const arg of args) {
                        if (!arg.evaluate(ctx))
                            return false;
                    }
                    return true;
                }
            ]
        ]
    },
    'any': {
        type: BooleanType,
        overloads: [
            [
                [BooleanType, BooleanType],
                (ctx, [a, b]) => a.evaluate(ctx) || b.evaluate(ctx)
            ],
            [
                varargs(BooleanType),
                (ctx, args) => {
                    for (const arg of args) {
                        if (arg.evaluate(ctx))
                            return true;
                    }
                    return false;
                }
            ]
        ]
    },
    '!': [
        BooleanType,
        [BooleanType],
        (ctx, [b]) => !b.evaluate(ctx)
    ],
    'is-supported-script': [
        BooleanType,
        [StringType],
        // At parse time this will always return true, so we need to exclude this expression with isGlobalPropertyConstant
        (ctx, [s]) => {
            const isSupportedScript = ctx.globals && ctx.globals.isSupportedScript;
            if (isSupportedScript) {
                return isSupportedScript(s.evaluate(ctx));
            }
            return true;
        }
    ],
    'upcase': [
        StringType,
        [StringType],
        (ctx, [s]) => s.evaluate(ctx).toUpperCase()
    ],
    'downcase': [
        StringType,
        [StringType],
        (ctx, [s]) => s.evaluate(ctx).toLowerCase()
    ],
    'concat': [
        StringType,
        varargs(ValueType),
        (ctx, args) => args.map(arg => toString(arg.evaluate(ctx))).join('')
    ],
    'resolved-locale': [
        StringType,
        [CollatorType],
        (ctx, [collator]) => collator.evaluate(ctx).resolvedLocale()
    ]
});
function stringifySignature(signature) {
    if (Array.isArray(signature)) {
        return `(${signature.map(toString$1).join(', ')})`;
    }
    else {
        return `(${toString$1(signature.type)}...)`;
    }
}
function isExpressionConstant(expression) {
    if (expression instanceof Var) {
        return isExpressionConstant(expression.boundExpression);
    }
    else if (expression instanceof CompoundExpression && expression.name === 'error') {
        return false;
    }
    else if (expression instanceof CollatorExpression) {
        // Although the results of a Collator expression with fixed arguments
        // generally shouldn't change between executions, we can't serialize them
        // as constant expressions because results change based on environment.
        return false;
    }
    else if (expression instanceof Within) {
        return false;
    }
    else if (expression instanceof Distance) {
        return false;
    }
    const isTypeAnnotation = expression instanceof Coercion ||
        expression instanceof Assertion;
    let childrenConstant = true;
    expression.eachChild(child => {
        // We can _almost_ assume that if `expressions` children are constant,
        // they would already have been evaluated to Literal values when they
        // were parsed.  Type annotations are the exception, because they might
        // have been inferred and added after a child was parsed.
        // So we recurse into isConstant() for the children of type annotations,
        // but otherwise simply check whether they are Literals.
        if (isTypeAnnotation) {
            childrenConstant = childrenConstant && isExpressionConstant(child);
        }
        else {
            childrenConstant = childrenConstant && child instanceof Literal;
        }
    });
    if (!childrenConstant) {
        return false;
    }
    return isFeatureConstant(expression) &&
        isGlobalPropertyConstant(expression, ['zoom', 'heatmap-density', 'line-progress', 'accumulated', 'is-supported-script']);
}
function isFeatureConstant(e) {
    if (e instanceof CompoundExpression) {
        if (e.name === 'get' && e.args.length === 1) {
            return false;
        }
        else if (e.name === 'feature-state') {
            return false;
        }
        else if (e.name === 'has' && e.args.length === 1) {
            return false;
        }
        else if (e.name === 'properties' ||
            e.name === 'geometry-type' ||
            e.name === 'id') {
            return false;
        }
        else if (/^filter-/.test(e.name)) {
            return false;
        }
    }
    if (e instanceof Within) {
        return false;
    }
    if (e instanceof Distance) {
        return false;
    }
    let result = true;
    e.eachChild(arg => {
        if (result && !isFeatureConstant(arg)) {
            result = false;
        }
    });
    return result;
}
function isStateConstant(e) {
    if (e instanceof CompoundExpression) {
        if (e.name === 'feature-state') {
            return false;
        }
    }
    let result = true;
    e.eachChild(arg => {
        if (result && !isStateConstant(arg)) {
            result = false;
        }
    });
    return result;
}
function isGlobalPropertyConstant(e, properties) {
    if (e instanceof CompoundExpression && properties.indexOf(e.name) >= 0) {
        return false;
    }
    let result = true;
    e.eachChild((arg) => {
        if (result && !isGlobalPropertyConstant(arg, properties)) {
            result = false;
        }
    });
    return result;
}

function success(value) {
    return { result: 'success', value };
}
function error(value) {
    return { result: 'error', value };
}

function supportsPropertyExpression(spec) {
    return spec['property-type'] === 'data-driven' || spec['property-type'] === 'cross-faded-data-driven';
}
function supportsZoomExpression(spec) {
    return !!spec.expression && spec.expression.parameters.indexOf('zoom') > -1;
}
function supportsInterpolation(spec) {
    return !!spec.expression && spec.expression.interpolated;
}

function getType(val) {
    if (val instanceof Number) {
        return 'number';
    }
    else if (val instanceof String) {
        return 'string';
    }
    else if (val instanceof Boolean) {
        return 'boolean';
    }
    else if (Array.isArray(val)) {
        return 'array';
    }
    else if (val === null) {
        return 'null';
    }
    else {
        return typeof val;
    }
}

function isFunction(value) {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

class StyleExpression {
    constructor(expression, propertySpec) {
        this.expression = expression;
        this._warningHistory = {};
        this._evaluator = new EvaluationContext();
        this._defaultValue = propertySpec ? getDefaultValue(propertySpec) : null;
        this._enumValues = propertySpec && propertySpec.type === 'enum' ? propertySpec.values : null;
    }
    evaluateWithoutErrorHandling(globals, feature, featureState, canonical, availableImages, formattedSection) {
        this._evaluator.globals = globals;
        this._evaluator.feature = feature;
        this._evaluator.featureState = featureState;
        this._evaluator.canonical = canonical;
        this._evaluator.availableImages = availableImages || null;
        this._evaluator.formattedSection = formattedSection;
        return this.expression.evaluate(this._evaluator);
    }
    evaluate(globals, feature, featureState, canonical, availableImages, formattedSection) {
        this._evaluator.globals = globals;
        this._evaluator.feature = feature || null;
        this._evaluator.featureState = featureState || null;
        this._evaluator.canonical = canonical;
        this._evaluator.availableImages = availableImages || null;
        this._evaluator.formattedSection = formattedSection || null;
        try {
            const val = this.expression.evaluate(this._evaluator);
            // eslint-disable-next-line no-self-compare
            if (val === null || val === undefined || (typeof val === 'number' && val !== val)) {
                return this._defaultValue;
            }
            if (this._enumValues && !(val in this._enumValues)) {
                throw new RuntimeError(`Expected value to be one of ${Object.keys(this._enumValues).map(v => JSON.stringify(v)).join(', ')}, but found ${JSON.stringify(val)} instead.`);
            }
            return val;
        }
        catch (e) {
            if (!this._warningHistory[e.message]) {
                this._warningHistory[e.message] = true;
                if (typeof console !== 'undefined') {
                    console.warn(e.message);
                }
            }
            return this._defaultValue;
        }
    }
}
function isExpression(expression) {
    return Array.isArray(expression) && expression.length > 0 &&
        typeof expression[0] === 'string' && expression[0] in expressions;
}
/**
 * Parse and typecheck the given style spec JSON expression.  If
 * options.defaultValue is provided, then the resulting StyleExpression's
 * `evaluate()` method will handle errors by logging a warning (once per
 * message) and returning the default value.  Otherwise, it will throw
 * evaluation errors.
 *
 * @private
 */
function createExpression(expression, propertySpec) {
    const parser = new ParsingContext(expressions, isExpressionConstant, [], propertySpec ? getExpectedType(propertySpec) : undefined);
    // For string-valued properties, coerce to string at the top level rather than asserting.
    const parsed = parser.parse(expression, undefined, undefined, undefined, propertySpec && propertySpec.type === 'string' ? { typeAnnotation: 'coerce' } : undefined);
    if (!parsed) {
        return error(parser.errors);
    }
    return success(new StyleExpression(parsed, propertySpec));
}
class ZoomConstantExpression {
    constructor(kind, expression) {
        this.kind = kind;
        this._styleExpression = expression;
        this.isStateDependent = kind !== 'constant' && !isStateConstant(expression.expression);
    }
    evaluateWithoutErrorHandling(globals, feature, featureState, canonical, availableImages, formattedSection) {
        return this._styleExpression.evaluateWithoutErrorHandling(globals, feature, featureState, canonical, availableImages, formattedSection);
    }
    evaluate(globals, feature, featureState, canonical, availableImages, formattedSection) {
        return this._styleExpression.evaluate(globals, feature, featureState, canonical, availableImages, formattedSection);
    }
}
class ZoomDependentExpression {
    constructor(kind, expression, zoomStops, interpolationType) {
        this.kind = kind;
        this.zoomStops = zoomStops;
        this._styleExpression = expression;
        this.isStateDependent = kind !== 'camera' && !isStateConstant(expression.expression);
        this.interpolationType = interpolationType;
    }
    evaluateWithoutErrorHandling(globals, feature, featureState, canonical, availableImages, formattedSection) {
        return this._styleExpression.evaluateWithoutErrorHandling(globals, feature, featureState, canonical, availableImages, formattedSection);
    }
    evaluate(globals, feature, featureState, canonical, availableImages, formattedSection) {
        return this._styleExpression.evaluate(globals, feature, featureState, canonical, availableImages, formattedSection);
    }
    interpolationFactor(input, lower, upper) {
        if (this.interpolationType) {
            return Interpolate.interpolationFactor(this.interpolationType, input, lower, upper);
        }
        else {
            return 0;
        }
    }
}
function createPropertyExpression(expressionInput, propertySpec) {
    const expression = createExpression(expressionInput, propertySpec);
    if (expression.result === 'error') {
        return expression;
    }
    const parsed = expression.value.expression;
    const isFeatureConstantResult = isFeatureConstant(parsed);
    if (!isFeatureConstantResult && !supportsPropertyExpression(propertySpec)) {
        return error([new ExpressionParsingError('', 'data expressions not supported')]);
    }
    const isZoomConstant = isGlobalPropertyConstant(parsed, ['zoom']);
    if (!isZoomConstant && !supportsZoomExpression(propertySpec)) {
        return error([new ExpressionParsingError('', 'zoom expressions not supported')]);
    }
    const zoomCurve = findZoomCurve(parsed);
    if (!zoomCurve && !isZoomConstant) {
        return error([new ExpressionParsingError('', '"zoom" expression may only be used as input to a top-level "step" or "interpolate" expression.')]);
    }
    else if (zoomCurve instanceof ExpressionParsingError) {
        return error([zoomCurve]);
    }
    else if (zoomCurve instanceof Interpolate && !supportsInterpolation(propertySpec)) {
        return error([new ExpressionParsingError('', '"interpolate" expressions cannot be used with this property')]);
    }
    if (!zoomCurve) {
        return success(isFeatureConstantResult ?
            new ZoomConstantExpression('constant', expression.value) :
            new ZoomConstantExpression('source', expression.value));
    }
    const interpolationType = zoomCurve instanceof Interpolate ? zoomCurve.interpolation : undefined;
    return success(isFeatureConstantResult ?
        new ZoomDependentExpression('camera', expression.value, zoomCurve.labels, interpolationType) :
        new ZoomDependentExpression('composite', expression.value, zoomCurve.labels, interpolationType));
}
// Zoom-dependent expressions may only use ["zoom"] as the input to a top-level "step" or "interpolate"
// expression (collectively referred to as a "curve"). The curve may be wrapped in one or more "let" or
// "coalesce" expressions.
function findZoomCurve(expression) {
    let result = null;
    if (expression instanceof Let) {
        result = findZoomCurve(expression.result);
    }
    else if (expression instanceof Coalesce) {
        for (const arg of expression.args) {
            result = findZoomCurve(arg);
            if (result) {
                break;
            }
        }
    }
    else if ((expression instanceof Step || expression instanceof Interpolate) &&
        expression.input instanceof CompoundExpression &&
        expression.input.name === 'zoom') {
        result = expression;
    }
    if (result instanceof ExpressionParsingError) {
        return result;
    }
    expression.eachChild((child) => {
        const childResult = findZoomCurve(child);
        if (childResult instanceof ExpressionParsingError) {
            result = childResult;
        }
        else if (!result && childResult) {
            result = new ExpressionParsingError('', '"zoom" expression may only be used as input to a top-level "step" or "interpolate" expression.');
        }
        else if (result && childResult && result !== childResult) {
            result = new ExpressionParsingError('', 'Only one zoom-based "step" or "interpolate" subexpression may be used in an expression.');
        }
    });
    return result;
}
function getExpectedType(spec) {
    const types = {
        color: ColorType,
        string: StringType,
        number: NumberType,
        enum: StringType,
        boolean: BooleanType,
        formatted: FormattedType,
        padding: PaddingType,
        resolvedImage: ResolvedImageType,
        variableAnchorOffsetCollection: VariableAnchorOffsetCollectionType
    };
    if (spec.type === 'array') {
        return array$1(types[spec.value] || ValueType, spec.length);
    }
    return types[spec.type];
}
function getDefaultValue(spec) {
    if (spec.type === 'color' && isFunction(spec.default)) {
        // Special case for heatmap-color: it uses the 'default:' to define a
        // default color ramp, but createExpression expects a simple value to fall
        // back to in case of runtime errors
        return new Color(0, 0, 0, 0);
    }
    else if (spec.type === 'color') {
        return Color.parse(spec.default) || null;
    }
    else if (spec.type === 'padding') {
        return Padding.parse(spec.default) || null;
    }
    else if (spec.type === 'variableAnchorOffsetCollection') {
        return VariableAnchorOffsetCollection.parse(spec.default) || null;
    }
    else if (spec.default === undefined) {
        return null;
    }
    else {
        return spec.default;
    }
}

function validateObject(options) {
    const key = options.key;
    const object = options.value;
    const elementSpecs = options.valueSpec || {};
    const elementValidators = options.objectElementValidators || {};
    const style = options.style;
    const styleSpec = options.styleSpec;
    const validateSpec = options.validateSpec;
    let errors = [];
    const type = getType(object);
    if (type !== 'object') {
        return [new ValidationError(key, object, `object expected, ${type} found`)];
    }
    for (const objectKey in object) {
        const elementSpecKey = objectKey.split('.')[0]; // treat 'paint.*' as 'paint'
        const elementSpec = elementSpecs[elementSpecKey] || elementSpecs['*'];
        let validateElement;
        if (elementValidators[elementSpecKey]) {
            validateElement = elementValidators[elementSpecKey];
        }
        else if (elementSpecs[elementSpecKey]) {
            validateElement = validateSpec;
        }
        else if (elementValidators['*']) {
            validateElement = elementValidators['*'];
        }
        else if (elementSpecs['*']) {
            validateElement = validateSpec;
        }
        else {
            errors.push(new ValidationError(key, object[objectKey], `unknown property "${objectKey}"`));
            continue;
        }
        errors = errors.concat(validateElement({
            key: (key ? `${key}.` : key) + objectKey,
            value: object[objectKey],
            valueSpec: elementSpec,
            style,
            styleSpec,
            object,
            objectKey,
            validateSpec,
        }, object));
    }
    for (const elementSpecKey in elementSpecs) {
        // Don't check `required` when there's a custom validator for that property.
        if (elementValidators[elementSpecKey]) {
            continue;
        }
        if (elementSpecs[elementSpecKey].required && elementSpecs[elementSpecKey]['default'] === undefined && object[elementSpecKey] === undefined) {
            errors.push(new ValidationError(key, object, `missing required property "${elementSpecKey}"`));
        }
    }
    return errors;
}

function validateArray(options) {
    const array = options.value;
    const arraySpec = options.valueSpec;
    const validateSpec = options.validateSpec;
    const style = options.style;
    const styleSpec = options.styleSpec;
    const key = options.key;
    const validateArrayElement = options.arrayElementValidator || validateSpec;
    if (getType(array) !== 'array') {
        return [new ValidationError(key, array, `array expected, ${getType(array)} found`)];
    }
    if (arraySpec.length && array.length !== arraySpec.length) {
        return [new ValidationError(key, array, `array length ${arraySpec.length} expected, length ${array.length} found`)];
    }
    if (arraySpec['min-length'] && array.length < arraySpec['min-length']) {
        return [new ValidationError(key, array, `array length at least ${arraySpec['min-length']} expected, length ${array.length} found`)];
    }
    let arrayElementSpec = {
        'type': arraySpec.value,
        'values': arraySpec.values
    };
    if (styleSpec.$version < 7) {
        arrayElementSpec['function'] = arraySpec.function;
    }
    if (getType(arraySpec.value) === 'object') {
        arrayElementSpec = arraySpec.value;
    }
    let errors = [];
    for (let i = 0; i < array.length; i++) {
        errors = errors.concat(validateArrayElement({
            array,
            arrayIndex: i,
            value: array[i],
            valueSpec: arrayElementSpec,
            validateSpec: options.validateSpec,
            style,
            styleSpec,
            key: `${key}[${i}]`
        }));
    }
    return errors;
}

function validateNumber(options) {
    const key = options.key;
    const value = options.value;
    const valueSpec = options.valueSpec;
    let type = getType(value);
    // eslint-disable-next-line no-self-compare
    if (type === 'number' && value !== value) {
        type = 'NaN';
    }
    if (type !== 'number') {
        return [new ValidationError(key, value, `number expected, ${type} found`)];
    }
    if ('minimum' in valueSpec && value < valueSpec.minimum) {
        return [new ValidationError(key, value, `${value} is less than the minimum value ${valueSpec.minimum}`)];
    }
    if ('maximum' in valueSpec && value > valueSpec.maximum) {
        return [new ValidationError(key, value, `${value} is greater than the maximum value ${valueSpec.maximum}`)];
    }
    return [];
}

function validateFunction(options) {
    const functionValueSpec = options.valueSpec;
    const functionType = unbundle(options.value.type);
    let stopKeyType;
    let stopDomainValues = {};
    let previousStopDomainValue;
    let previousStopDomainZoom;
    const isZoomFunction = functionType !== 'categorical' && options.value.property === undefined;
    const isPropertyFunction = !isZoomFunction;
    const isZoomAndPropertyFunction = getType(options.value.stops) === 'array' &&
        getType(options.value.stops[0]) === 'array' &&
        getType(options.value.stops[0][0]) === 'object';
    const errors = validateObject({
        key: options.key,
        value: options.value,
        valueSpec: options.styleSpec.function,
        validateSpec: options.validateSpec,
        style: options.style,
        styleSpec: options.styleSpec,
        objectElementValidators: {
            stops: validateFunctionStops,
            default: validateFunctionDefault
        }
    });
    if (functionType === 'identity' && isZoomFunction) {
        errors.push(new ValidationError(options.key, options.value, 'missing required property "property"'));
    }
    if (functionType !== 'identity' && !options.value.stops) {
        errors.push(new ValidationError(options.key, options.value, 'missing required property "stops"'));
    }
    if (functionType === 'exponential' && options.valueSpec.expression && !supportsInterpolation(options.valueSpec)) {
        errors.push(new ValidationError(options.key, options.value, 'exponential functions not supported'));
    }
    if (options.styleSpec.$version >= 8) {
        if (isPropertyFunction && !supportsPropertyExpression(options.valueSpec)) {
            errors.push(new ValidationError(options.key, options.value, 'property functions not supported'));
        }
        else if (isZoomFunction && !supportsZoomExpression(options.valueSpec)) {
            errors.push(new ValidationError(options.key, options.value, 'zoom functions not supported'));
        }
    }
    if ((functionType === 'categorical' || isZoomAndPropertyFunction) && options.value.property === undefined) {
        errors.push(new ValidationError(options.key, options.value, '"property" property is required'));
    }
    return errors;
    function validateFunctionStops(options) {
        if (functionType === 'identity') {
            return [new ValidationError(options.key, options.value, 'identity function may not have a "stops" property')];
        }
        let errors = [];
        const value = options.value;
        errors = errors.concat(validateArray({
            key: options.key,
            value,
            valueSpec: options.valueSpec,
            validateSpec: options.validateSpec,
            style: options.style,
            styleSpec: options.styleSpec,
            arrayElementValidator: validateFunctionStop
        }));
        if (getType(value) === 'array' && value.length === 0) {
            errors.push(new ValidationError(options.key, value, 'array must have at least one stop'));
        }
        return errors;
    }
    function validateFunctionStop(options) {
        let errors = [];
        const value = options.value;
        const key = options.key;
        if (getType(value) !== 'array') {
            return [new ValidationError(key, value, `array expected, ${getType(value)} found`)];
        }
        if (value.length !== 2) {
            return [new ValidationError(key, value, `array length 2 expected, length ${value.length} found`)];
        }
        if (isZoomAndPropertyFunction) {
            if (getType(value[0]) !== 'object') {
                return [new ValidationError(key, value, `object expected, ${getType(value[0])} found`)];
            }
            if (value[0].zoom === undefined) {
                return [new ValidationError(key, value, 'object stop key must have zoom')];
            }
            if (value[0].value === undefined) {
                return [new ValidationError(key, value, 'object stop key must have value')];
            }
            if (previousStopDomainZoom && previousStopDomainZoom > unbundle(value[0].zoom)) {
                return [new ValidationError(key, value[0].zoom, 'stop zoom values must appear in ascending order')];
            }
            if (unbundle(value[0].zoom) !== previousStopDomainZoom) {
                previousStopDomainZoom = unbundle(value[0].zoom);
                previousStopDomainValue = undefined;
                stopDomainValues = {};
            }
            errors = errors.concat(validateObject({
                key: `${key}[0]`,
                value: value[0],
                valueSpec: { zoom: {} },
                validateSpec: options.validateSpec,
                style: options.style,
                styleSpec: options.styleSpec,
                objectElementValidators: { zoom: validateNumber, value: validateStopDomainValue }
            }));
        }
        else {
            errors = errors.concat(validateStopDomainValue({
                key: `${key}[0]`,
                value: value[0],
                valueSpec: {},
                validateSpec: options.validateSpec,
                style: options.style,
                styleSpec: options.styleSpec
            }, value));
        }
        if (isExpression(deepUnbundle(value[1]))) {
            return errors.concat([new ValidationError(`${key}[1]`, value[1], 'expressions are not allowed in function stops.')]);
        }
        return errors.concat(options.validateSpec({
            key: `${key}[1]`,
            value: value[1],
            valueSpec: functionValueSpec,
            validateSpec: options.validateSpec,
            style: options.style,
            styleSpec: options.styleSpec
        }));
    }
    function validateStopDomainValue(options, stop) {
        const type = getType(options.value);
        const value = unbundle(options.value);
        const reportValue = options.value !== null ? options.value : stop;
        if (!stopKeyType) {
            stopKeyType = type;
        }
        else if (type !== stopKeyType) {
            return [new ValidationError(options.key, reportValue, `${type} stop domain type must match previous stop domain type ${stopKeyType}`)];
        }
        if (type !== 'number' && type !== 'string' && type !== 'boolean') {
            return [new ValidationError(options.key, reportValue, 'stop domain value must be a number, string, or boolean')];
        }
        if (type !== 'number' && functionType !== 'categorical') {
            let message = `number expected, ${type} found`;
            if (supportsPropertyExpression(functionValueSpec) && functionType === undefined) {
                message += '\nIf you intended to use a categorical function, specify `"type": "categorical"`.';
            }
            return [new ValidationError(options.key, reportValue, message)];
        }
        if (functionType === 'categorical' && type === 'number' && (!isFinite(value) || Math.floor(value) !== value)) {
            return [new ValidationError(options.key, reportValue, `integer expected, found ${value}`)];
        }
        if (functionType !== 'categorical' && type === 'number' && previousStopDomainValue !== undefined && value < previousStopDomainValue) {
            return [new ValidationError(options.key, reportValue, 'stop domain values must appear in ascending order')];
        }
        else {
            previousStopDomainValue = value;
        }
        if (functionType === 'categorical' && value in stopDomainValues) {
            return [new ValidationError(options.key, reportValue, 'stop domain values must be unique')];
        }
        else {
            stopDomainValues[value] = true;
        }
        return [];
    }
    function validateFunctionDefault(options) {
        return options.validateSpec({
            key: options.key,
            value: options.value,
            valueSpec: functionValueSpec,
            validateSpec: options.validateSpec,
            style: options.style,
            styleSpec: options.styleSpec
        });
    }
}

function validateExpression(options) {
    const expression = (options.expressionContext === 'property' ? createPropertyExpression : createExpression)(deepUnbundle(options.value), options.valueSpec);
    if (expression.result === 'error') {
        return expression.value.map((error) => {
            return new ValidationError(`${options.key}${error.key}`, options.value, error.message);
        });
    }
    const expressionObj = expression.value.expression || expression.value._styleExpression.expression;
    if (options.expressionContext === 'property' && (options.propertyKey === 'text-font') &&
        !expressionObj.outputDefined()) {
        return [new ValidationError(options.key, options.value, `Invalid data expression for "${options.propertyKey}". Output values must be contained as literals within the expression.`)];
    }
    if (options.expressionContext === 'property' && options.propertyType === 'layout' &&
        (!isStateConstant(expressionObj))) {
        return [new ValidationError(options.key, options.value, '"feature-state" data expressions are not supported with layout properties.')];
    }
    if (options.expressionContext === 'filter' && !isStateConstant(expressionObj)) {
        return [new ValidationError(options.key, options.value, '"feature-state" data expressions are not supported with filters.')];
    }
    if (options.expressionContext && options.expressionContext.indexOf('cluster') === 0) {
        if (!isGlobalPropertyConstant(expressionObj, ['zoom', 'feature-state'])) {
            return [new ValidationError(options.key, options.value, '"zoom" and "feature-state" expressions are not supported with cluster properties.')];
        }
        if (options.expressionContext === 'cluster-initial' && !isFeatureConstant(expressionObj)) {
            return [new ValidationError(options.key, options.value, 'Feature data expressions are not supported with initial expression part of cluster properties.')];
        }
    }
    return [];
}

function validateBoolean(options) {
    const value = options.value;
    const key = options.key;
    const type = getType(value);
    if (type !== 'boolean') {
        return [new ValidationError(key, value, `boolean expected, ${type} found`)];
    }
    return [];
}

function validateColor(options) {
    const key = options.key;
    const value = options.value;
    const type = getType(value);
    if (type !== 'string') {
        return [new ValidationError(key, value, `color expected, ${type} found`)];
    }
    if (!Color.parse(String(value))) { // cast String object to string primitive
        return [new ValidationError(key, value, `color expected, "${value}" found`)];
    }
    return [];
}

function validateEnum(options) {
    const key = options.key;
    const value = options.value;
    const valueSpec = options.valueSpec;
    const errors = [];
    if (Array.isArray(valueSpec.values)) { // <=v7
        if (valueSpec.values.indexOf(unbundle(value)) === -1) {
            errors.push(new ValidationError(key, value, `expected one of [${valueSpec.values.join(', ')}], ${JSON.stringify(value)} found`));
        }
    }
    else { // >=v8
        if (Object.keys(valueSpec.values).indexOf(unbundle(value)) === -1) {
            errors.push(new ValidationError(key, value, `expected one of [${Object.keys(valueSpec.values).join(', ')}], ${JSON.stringify(value)} found`));
        }
    }
    return errors;
}

function isExpressionFilter(filter) {
    if (filter === true || filter === false) {
        return true;
    }
    if (!Array.isArray(filter) || filter.length === 0) {
        return false;
    }
    switch (filter[0]) {
        case 'has':
            return filter.length >= 2 && filter[1] !== '$id' && filter[1] !== '$type';
        case 'in':
            return filter.length >= 3 && (typeof filter[1] !== 'string' || Array.isArray(filter[2]));
        case '!in':
        case '!has':
        case 'none':
            return false;
        case '==':
        case '!=':
        case '>':
        case '>=':
        case '<':
        case '<=':
            return filter.length !== 3 || (Array.isArray(filter[1]) || Array.isArray(filter[2]));
        case 'any':
        case 'all':
            for (const f of filter.slice(1)) {
                if (!isExpressionFilter(f) && typeof f !== 'boolean') {
                    return false;
                }
            }
            return true;
        default:
            return true;
    }
}

function validateFilter(options) {
    if (isExpressionFilter(deepUnbundle(options.value))) {
        return validateExpression(extendBy({}, options, {
            expressionContext: 'filter',
            valueSpec: { value: 'boolean' }
        }));
    }
    else {
        return validateNonExpressionFilter(options);
    }
}
function validateNonExpressionFilter(options) {
    const value = options.value;
    const key = options.key;
    if (getType(value) !== 'array') {
        return [new ValidationError(key, value, `array expected, ${getType(value)} found`)];
    }
    const styleSpec = options.styleSpec;
    let type;
    let errors = [];
    if (value.length < 1) {
        return [new ValidationError(key, value, 'filter array must have at least 1 element')];
    }
    errors = errors.concat(validateEnum({
        key: `${key}[0]`,
        value: value[0],
        valueSpec: styleSpec.filter_operator,
        style: options.style,
        styleSpec: options.styleSpec
    }));
    switch (unbundle(value[0])) {
        case '<':
        case '<=':
        case '>':
        case '>=':
            if (value.length >= 2 && unbundle(value[1]) === '$type') {
                errors.push(new ValidationError(key, value, `"$type" cannot be use with operator "${value[0]}"`));
            }
        /* falls through */
        case '==':
        case '!=':
            if (value.length !== 3) {
                errors.push(new ValidationError(key, value, `filter array for operator "${value[0]}" must have 3 elements`));
            }
        /* falls through */
        case 'in':
        case '!in':
            if (value.length >= 2) {
                type = getType(value[1]);
                if (type !== 'string') {
                    errors.push(new ValidationError(`${key}[1]`, value[1], `string expected, ${type} found`));
                }
            }
            for (let i = 2; i < value.length; i++) {
                type = getType(value[i]);
                if (unbundle(value[1]) === '$type') {
                    errors = errors.concat(validateEnum({
                        key: `${key}[${i}]`,
                        value: value[i],
                        valueSpec: styleSpec.geometry_type,
                        style: options.style,
                        styleSpec: options.styleSpec
                    }));
                }
                else if (type !== 'string' && type !== 'number' && type !== 'boolean') {
                    errors.push(new ValidationError(`${key}[${i}]`, value[i], `string, number, or boolean expected, ${type} found`));
                }
            }
            break;
        case 'any':
        case 'all':
        case 'none':
            for (let i = 1; i < value.length; i++) {
                errors = errors.concat(validateNonExpressionFilter({
                    key: `${key}[${i}]`,
                    value: value[i],
                    style: options.style,
                    styleSpec: options.styleSpec
                }));
            }
            break;
        case 'has':
        case '!has':
            type = getType(value[1]);
            if (value.length !== 2) {
                errors.push(new ValidationError(key, value, `filter array for "${value[0]}" operator must have 2 elements`));
            }
            else if (type !== 'string') {
                errors.push(new ValidationError(`${key}[1]`, value[1], `string expected, ${type} found`));
            }
            break;
    }
    return errors;
}

function validateProperty(options, propertyType) {
    const key = options.key;
    const validateSpec = options.validateSpec;
    const style = options.style;
    const styleSpec = options.styleSpec;
    const value = options.value;
    const propertyKey = options.objectKey;
    const layerSpec = styleSpec[`${propertyType}_${options.layerType}`];
    if (!layerSpec)
        return [];
    const transitionMatch = propertyKey.match(/^(.*)-transition$/);
    if (propertyType === 'paint' && transitionMatch && layerSpec[transitionMatch[1]] && layerSpec[transitionMatch[1]].transition) {
        return validateSpec({
            key,
            value,
            valueSpec: styleSpec.transition,
            style,
            styleSpec
        });
    }
    const valueSpec = options.valueSpec || layerSpec[propertyKey];
    if (!valueSpec) {
        return [new ValidationError(key, value, `unknown property "${propertyKey}"`)];
    }
    let tokenMatch;
    if (getType(value) === 'string' && supportsPropertyExpression(valueSpec) && !valueSpec.tokens && (tokenMatch = /^{([^}]+)}$/.exec(value))) {
        return [new ValidationError(key, value, `"${propertyKey}" does not support interpolation syntax\n` +
                `Use an identity property function instead: \`{ "type": "identity", "property": ${JSON.stringify(tokenMatch[1])} }\`.`)];
    }
    const errors = [];
    if (options.layerType === 'symbol') {
        if (propertyKey === 'text-field' && style && !style.glyphs) {
            errors.push(new ValidationError(key, value, 'use of "text-field" requires a style "glyphs" property'));
        }
        if (propertyKey === 'text-font' && isFunction(deepUnbundle(value)) && unbundle(value.type) === 'identity') {
            errors.push(new ValidationError(key, value, '"text-font" does not support identity functions'));
        }
    }
    return errors.concat(validateSpec({
        key: options.key,
        value,
        valueSpec,
        style,
        styleSpec,
        expressionContext: 'property',
        propertyType,
        propertyKey
    }));
}

function validatePaintProperty(options) {
    return validateProperty(options, 'paint');
}

function validateLayoutProperty(options) {
    return validateProperty(options, 'layout');
}

function validateLayer(options) {
    let errors = [];
    const layer = options.value;
    const key = options.key;
    const style = options.style;
    const styleSpec = options.styleSpec;
    if (!layer.type && !layer.ref) {
        errors.push(new ValidationError(key, layer, 'either "type" or "ref" is required'));
    }
    let type = unbundle(layer.type);
    const ref = unbundle(layer.ref);
    if (layer.id) {
        const layerId = unbundle(layer.id);
        for (let i = 0; i < options.arrayIndex; i++) {
            const otherLayer = style.layers[i];
            if (unbundle(otherLayer.id) === layerId) {
                errors.push(new ValidationError(key, layer.id, `duplicate layer id "${layer.id}", previously used at line ${otherLayer.id.__line__}`));
            }
        }
    }
    if ('ref' in layer) {
        ['type', 'source', 'source-layer', 'filter', 'layout'].forEach((p) => {
            if (p in layer) {
                errors.push(new ValidationError(key, layer[p], `"${p}" is prohibited for ref layers`));
            }
        });
        let parent;
        style.layers.forEach((layer) => {
            if (unbundle(layer.id) === ref)
                parent = layer;
        });
        if (!parent) {
            errors.push(new ValidationError(key, layer.ref, `ref layer "${ref}" not found`));
        }
        else if (parent.ref) {
            errors.push(new ValidationError(key, layer.ref, 'ref cannot reference another ref layer'));
        }
        else {
            type = unbundle(parent.type);
        }
    }
    else if (type !== 'background') {
        if (!layer.source) {
            errors.push(new ValidationError(key, layer, 'missing required property "source"'));
        }
        else {
            const source = style.sources && style.sources[layer.source];
            const sourceType = source && unbundle(source.type);
            if (!source) {
                errors.push(new ValidationError(key, layer.source, `source "${layer.source}" not found`));
            }
            else if (sourceType === 'vector' && type === 'raster') {
                errors.push(new ValidationError(key, layer.source, `layer "${layer.id}" requires a raster source`));
            }
            else if (sourceType !== 'raster-dem' && type === 'hillshade') {
                errors.push(new ValidationError(key, layer.source, `layer "${layer.id}" requires a raster-dem source`));
            }
            else if (sourceType === 'raster' && type !== 'raster') {
                errors.push(new ValidationError(key, layer.source, `layer "${layer.id}" requires a vector source`));
            }
            else if (sourceType === 'vector' && !layer['source-layer']) {
                errors.push(new ValidationError(key, layer, `layer "${layer.id}" must specify a "source-layer"`));
            }
            else if (sourceType === 'raster-dem' && type !== 'hillshade') {
                errors.push(new ValidationError(key, layer.source, 'raster-dem source can only be used with layer type \'hillshade\'.'));
            }
            else if (type === 'line' && layer.paint && layer.paint['line-gradient'] &&
                (sourceType !== 'geojson' || !source.lineMetrics)) {
                errors.push(new ValidationError(key, layer, `layer "${layer.id}" specifies a line-gradient, which requires a GeoJSON source with \`lineMetrics\` enabled.`));
            }
        }
    }
    errors = errors.concat(validateObject({
        key,
        value: layer,
        valueSpec: styleSpec.layer,
        style: options.style,
        styleSpec: options.styleSpec,
        validateSpec: options.validateSpec,
        objectElementValidators: {
            '*'() {
                return [];
            },
            // We don't want to enforce the spec's `"requires": true` for backward compatibility with refs;
            // the actual requirement is validated above. See https://github.com/mapbox/mapbox-gl-js/issues/5772.
            type() {
                return options.validateSpec({
                    key: `${key}.type`,
                    value: layer.type,
                    valueSpec: styleSpec.layer.type,
                    style: options.style,
                    styleSpec: options.styleSpec,
                    validateSpec: options.validateSpec,
                    object: layer,
                    objectKey: 'type'
                });
            },
            filter: validateFilter,
            layout(options) {
                return validateObject({
                    layer,
                    key: options.key,
                    value: options.value,
                    style: options.style,
                    styleSpec: options.styleSpec,
                    validateSpec: options.validateSpec,
                    objectElementValidators: {
                        '*'(options) {
                            return validateLayoutProperty(extendBy({ layerType: type }, options));
                        }
                    }
                });
            },
            paint(options) {
                return validateObject({
                    layer,
                    key: options.key,
                    value: options.value,
                    style: options.style,
                    styleSpec: options.styleSpec,
                    validateSpec: options.validateSpec,
                    objectElementValidators: {
                        '*'(options) {
                            return validatePaintProperty(extendBy({ layerType: type }, options));
                        }
                    }
                });
            }
        }
    }));
    return errors;
}

function validateString(options) {
    const value = options.value;
    const key = options.key;
    const type = getType(value);
    if (type !== 'string') {
        return [new ValidationError(key, value, `string expected, ${type} found`)];
    }
    return [];
}

function validateRasterDEMSource(options) {
    var _a;
    const sourceName = (_a = options.sourceName) !== null && _a !== void 0 ? _a : '';
    const rasterDEM = options.value;
    const styleSpec = options.styleSpec;
    const rasterDEMSpec = styleSpec.source_raster_dem;
    const style = options.style;
    let errors = [];
    const rootType = getType(rasterDEM);
    if (rasterDEM === undefined) {
        return errors;
    }
    else if (rootType !== 'object') {
        errors.push(new ValidationError('source_raster_dem', rasterDEM, `object expected, ${rootType} found`));
        return errors;
    }
    const encoding = unbundle(rasterDEM.encoding);
    const isCustomEncoding = encoding === 'custom';
    const customEncodingKeys = ['redFactor', 'greenFactor', 'blueFactor', 'baseShift'];
    const encodingName = options.value.encoding ? `"${options.value.encoding}"` : 'Default';
    for (const key in rasterDEM) {
        if (!isCustomEncoding && customEncodingKeys.includes(key)) {
            errors.push(new ValidationError(key, rasterDEM[key], `In "${sourceName}": "${key}" is only valid when "encoding" is set to "custom". ${encodingName} encoding found`));
        }
        else if (rasterDEMSpec[key]) {
            errors = errors.concat(options.validateSpec({
                key,
                value: rasterDEM[key],
                valueSpec: rasterDEMSpec[key],
                validateSpec: options.validateSpec,
                style,
                styleSpec
            }));
        }
        else {
            errors.push(new ValidationError(key, rasterDEM[key], `unknown property "${key}"`));
        }
    }
    return errors;
}

const objectElementValidators = {
    promoteId: validatePromoteId
};
function validateSource(options) {
    const value = options.value;
    const key = options.key;
    const styleSpec = options.styleSpec;
    const style = options.style;
    const validateSpec = options.validateSpec;
    if (!value.type) {
        return [new ValidationError(key, value, '"type" is required')];
    }
    const type = unbundle(value.type);
    let errors;
    switch (type) {
        case 'vector':
        case 'raster':
            errors = validateObject({
                key,
                value,
                valueSpec: styleSpec[`source_${type.replace('-', '_')}`],
                style: options.style,
                styleSpec,
                objectElementValidators,
                validateSpec,
            });
            return errors;
        case 'raster-dem':
            errors = validateRasterDEMSource({
                sourceName: key,
                value,
                style: options.style,
                styleSpec,
                validateSpec,
            });
            return errors;
        case 'geojson':
            errors = validateObject({
                key,
                value,
                valueSpec: styleSpec.source_geojson,
                style,
                styleSpec,
                validateSpec,
                objectElementValidators
            });
            if (value.cluster) {
                for (const prop in value.clusterProperties) {
                    const [operator, mapExpr] = value.clusterProperties[prop];
                    const reduceExpr = typeof operator === 'string' ? [operator, ['accumulated'], ['get', prop]] : operator;
                    errors.push(...validateExpression({
                        key: `${key}.${prop}.map`,
                        value: mapExpr,
                        validateSpec,
                        expressionContext: 'cluster-map'
                    }));
                    errors.push(...validateExpression({
                        key: `${key}.${prop}.reduce`,
                        value: reduceExpr,
                        validateSpec,
                        expressionContext: 'cluster-reduce'
                    }));
                }
            }
            return errors;
        case 'video':
            return validateObject({
                key,
                value,
                valueSpec: styleSpec.source_video,
                style,
                validateSpec,
                styleSpec
            });
        case 'image':
            return validateObject({
                key,
                value,
                valueSpec: styleSpec.source_image,
                style,
                validateSpec,
                styleSpec
            });
        case 'canvas':
            return [new ValidationError(key, null, 'Please use runtime APIs to add canvas sources, rather than including them in stylesheets.', 'source.canvas')];
        default:
            return validateEnum({
                key: `${key}.type`,
                value: value.type,
                valueSpec: { values: ['vector', 'raster', 'raster-dem', 'geojson', 'video', 'image'] },
                style,
                validateSpec,
                styleSpec
            });
    }
}
function validatePromoteId({ key, value }) {
    if (getType(value) === 'string') {
        return validateString({ key, value });
    }
    else {
        const errors = [];
        for (const prop in value) {
            errors.push(...validateString({ key: `${key}.${prop}`, value: value[prop] }));
        }
        return errors;
    }
}

function validateLight(options) {
    const light = options.value;
    const styleSpec = options.styleSpec;
    const lightSpec = styleSpec.light;
    const style = options.style;
    let errors = [];
    const rootType = getType(light);
    if (light === undefined) {
        return errors;
    }
    else if (rootType !== 'object') {
        errors = errors.concat([new ValidationError('light', light, `object expected, ${rootType} found`)]);
        return errors;
    }
    for (const key in light) {
        const transitionMatch = key.match(/^(.*)-transition$/);
        if (transitionMatch && lightSpec[transitionMatch[1]] && lightSpec[transitionMatch[1]].transition) {
            errors = errors.concat(options.validateSpec({
                key,
                value: light[key],
                valueSpec: styleSpec.transition,
                validateSpec: options.validateSpec,
                style,
                styleSpec
            }));
        }
        else if (lightSpec[key]) {
            errors = errors.concat(options.validateSpec({
                key,
                value: light[key],
                valueSpec: lightSpec[key],
                validateSpec: options.validateSpec,
                style,
                styleSpec
            }));
        }
        else {
            errors = errors.concat([new ValidationError(key, light[key], `unknown property "${key}"`)]);
        }
    }
    return errors;
}

function validateSky(options) {
    const sky = options.value;
    const styleSpec = options.styleSpec;
    const skySpec = styleSpec.sky;
    const style = options.style;
    const rootType = getType(sky);
    if (sky === undefined) {
        return [];
    }
    else if (rootType !== 'object') {
        return [new ValidationError('sky', sky, `object expected, ${rootType} found`)];
    }
    let errors = [];
    for (const key in sky) {
        if (skySpec[key]) {
            errors = errors.concat(options.validateSpec({
                key,
                value: sky[key],
                valueSpec: skySpec[key],
                style,
                styleSpec
            }));
        }
        else {
            errors = errors.concat([new ValidationError(key, sky[key], `unknown property "${key}"`)]);
        }
    }
    return errors;
}

function validateTerrain(options) {
    const terrain = options.value;
    const styleSpec = options.styleSpec;
    const terrainSpec = styleSpec.terrain;
    const style = options.style;
    let errors = [];
    const rootType = getType(terrain);
    if (terrain === undefined) {
        return errors;
    }
    else if (rootType !== 'object') {
        errors = errors.concat([new ValidationError('terrain', terrain, `object expected, ${rootType} found`)]);
        return errors;
    }
    for (const key in terrain) {
        if (terrainSpec[key]) {
            errors = errors.concat(options.validateSpec({
                key,
                value: terrain[key],
                valueSpec: terrainSpec[key],
                validateSpec: options.validateSpec,
                style,
                styleSpec
            }));
        }
        else {
            errors = errors.concat([new ValidationError(key, terrain[key], `unknown property "${key}"`)]);
        }
    }
    return errors;
}

function validateFormatted(options) {
    if (validateString(options).length === 0) {
        return [];
    }
    return validateExpression(options);
}

function validateImage(options) {
    if (validateString(options).length === 0) {
        return [];
    }
    return validateExpression(options);
}

function validatePadding(options) {
    const key = options.key;
    const value = options.value;
    const type = getType(value);
    if (type === 'array') {
        if (value.length < 1 || value.length > 4) {
            return [new ValidationError(key, value, `padding requires 1 to 4 values; ${value.length} values found`)];
        }
        const arrayElementSpec = {
            type: 'number'
        };
        let errors = [];
        for (let i = 0; i < value.length; i++) {
            errors = errors.concat(options.validateSpec({
                key: `${key}[${i}]`,
                value: value[i],
                validateSpec: options.validateSpec,
                valueSpec: arrayElementSpec
            }));
        }
        return errors;
    }
    else {
        return validateNumber({
            key,
            value,
            valueSpec: {}
        });
    }
}

function validateVariableAnchorOffsetCollection(options) {
    const key = options.key;
    const value = options.value;
    const type = getType(value);
    const styleSpec = options.styleSpec;
    if (type !== 'array' || value.length < 1 || value.length % 2 !== 0) {
        return [new ValidationError(key, value, 'variableAnchorOffsetCollection requires a non-empty array of even length')];
    }
    let errors = [];
    for (let i = 0; i < value.length; i += 2) {
        // Elements in even positions should be values from text-anchor enum
        errors = errors.concat(validateEnum({
            key: `${key}[${i}]`,
            value: value[i],
            valueSpec: styleSpec['layout_symbol']['text-anchor']
        }));
        // Elements in odd positions should be points (2-element numeric arrays)
        errors = errors.concat(validateArray({
            key: `${key}[${i + 1}]`,
            value: value[i + 1],
            valueSpec: {
                length: 2,
                value: 'number'
            },
            validateSpec: options.validateSpec,
            style: options.style,
            styleSpec
        }));
    }
    return errors;
}

function validateSprite(options) {
    let errors = [];
    const sprite = options.value;
    const key = options.key;
    if (!Array.isArray(sprite)) {
        return validateString({
            key,
            value: sprite
        });
    }
    else {
        const allSpriteIds = [];
        const allSpriteURLs = [];
        for (const i in sprite) {
            if (sprite[i].id && allSpriteIds.includes(sprite[i].id))
                errors.push(new ValidationError(key, sprite, `all the sprites' ids must be unique, but ${sprite[i].id} is duplicated`));
            allSpriteIds.push(sprite[i].id);
            if (sprite[i].url && allSpriteURLs.includes(sprite[i].url))
                errors.push(new ValidationError(key, sprite, `all the sprites' URLs must be unique, but ${sprite[i].url} is duplicated`));
            allSpriteURLs.push(sprite[i].url);
            const pairSpec = {
                id: {
                    type: 'string',
                    required: true,
                },
                url: {
                    type: 'string',
                    required: true,
                }
            };
            errors = errors.concat(validateObject({
                key: `${key}[${i}]`,
                value: sprite[i],
                valueSpec: pairSpec,
                validateSpec: options.validateSpec,
            }));
        }
        return errors;
    }
}

function validateProjection(options) {
    const projection = options.value;
    const styleSpec = options.styleSpec;
    const projectionSpec = styleSpec.projection;
    const style = options.style;
    const rootType = getType(projection);
    if (projection === undefined) {
        return [];
    }
    else if (rootType !== 'object') {
        return [new ValidationError('projection', projection, `object expected, ${rootType} found`)];
    }
    let errors = [];
    for (const key in projection) {
        if (projectionSpec[key]) {
            errors = errors.concat(options.validateSpec({
                key,
                value: projection[key],
                valueSpec: projectionSpec[key],
                style,
                styleSpec
            }));
        }
        else {
            errors = errors.concat([new ValidationError(key, projection[key], `unknown property "${key}"`)]);
        }
    }
    return errors;
}

const VALIDATORS = {
    '*'() {
        return [];
    },
    'array': validateArray,
    'boolean': validateBoolean,
    'number': validateNumber,
    'color': validateColor,
    'constants': validateConstants,
    'enum': validateEnum,
    'filter': validateFilter,
    'function': validateFunction,
    'layer': validateLayer,
    'object': validateObject,
    'source': validateSource,
    'light': validateLight,
    'sky': validateSky,
    'terrain': validateTerrain,
    'projection': validateProjection,
    'string': validateString,
    'formatted': validateFormatted,
    'resolvedImage': validateImage,
    'padding': validatePadding,
    'variableAnchorOffsetCollection': validateVariableAnchorOffsetCollection,
    'sprite': validateSprite,
};
/**
 * Main recursive validation function used internally.
 * You should use `validateStyleMin` in the browser or `validateStyle` in node env.
 * @param options - the options object
 * @param options.key - string representing location of validation in style tree. Used only
 * for more informative error reporting.
 * @param options.value - current value from style being evaluated. May be anything from a
 * high level object that needs to be descended into deeper or a simple
 * scalar value.
 * @param options.valueSpec - current spec being evaluated. Tracks value.
 * @param options.styleSpec - current full spec being evaluated.
 * @param options.validateSpec - the validate function itself
 * @param options.style - the style object
 * @param options.objectElementValidators - optional object of functions that will be called
 * @returns an array of errors, or an empty array if no errors are found.
 */
function validate(options) {
    const value = options.value;
    const valueSpec = options.valueSpec;
    const styleSpec = options.styleSpec;
    options.validateSpec = validate;
    if (valueSpec.expression && isFunction(unbundle(value))) {
        return validateFunction(options);
    }
    else if (valueSpec.expression && isExpression(deepUnbundle(value))) {
        return validateExpression(options);
    }
    else if (valueSpec.type && VALIDATORS[valueSpec.type]) {
        return VALIDATORS[valueSpec.type](options);
    }
    else {
        const valid = validateObject(extendBy({}, options, {
            valueSpec: valueSpec.type ? styleSpec[valueSpec.type] : valueSpec
        }));
        return valid;
    }
}

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
var v8Spec = {
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

function validateGlyphsUrl(options) {
    const value = options.value;
    const key = options.key;
    const errors = validateString(options);
    if (errors.length)
        return errors;
    if (value.indexOf('{fontstack}') === -1) {
        errors.push(new ValidationError(key, value, '"glyphs" url must include a "{fontstack}" token'));
    }
    if (value.indexOf('{range}') === -1) {
        errors.push(new ValidationError(key, value, '"glyphs" url must include a "{range}" token'));
    }
    return errors;
}

/**
 * Validate a MapLibre style against the style specification.
 * Use this when running in the browser.
 *
 * @param style - The style to be validated.
 * @param styleSpec - The style specification to validate against.
 * If omitted, the latest style spec is used.
 * @returns an array of errors, or an empty array if no errors are found.
 * @example
 *   const validate = require('@maplibre/maplibre-gl-style-spec/').validateStyleMin;
 *   const errors = validate(style);
 */
function validateStyleMin(style, styleSpec = v8Spec) {
    let errors = [];
    errors = errors.concat(validate({
        key: '',
        value: style,
        valueSpec: styleSpec.$root,
        styleSpec,
        style,
        validateSpec: validate,
        objectElementValidators: {
            glyphs: validateGlyphsUrl,
            '*'() {
                return [];
            }
        }
    }));
    if (style['constants']) {
        errors = errors.concat(validateConstants({
            key: 'constants',
            value: style['constants'],
            style,
            styleSpec,
            validateSpec: validate,
        }));
    }
    return sortErrors(errors);
}
validateStyleMin.source = wrapCleanErrors(injectValidateSpec(validateSource));
validateStyleMin.sprite = wrapCleanErrors(injectValidateSpec(validateSprite));
validateStyleMin.glyphs = wrapCleanErrors(injectValidateSpec(validateGlyphsUrl));
validateStyleMin.light = wrapCleanErrors(injectValidateSpec(validateLight));
validateStyleMin.sky = wrapCleanErrors(injectValidateSpec(validateSky));
validateStyleMin.terrain = wrapCleanErrors(injectValidateSpec(validateTerrain));
validateStyleMin.layer = wrapCleanErrors(injectValidateSpec(validateLayer));
validateStyleMin.filter = wrapCleanErrors(injectValidateSpec(validateFilter));
validateStyleMin.paintProperty = wrapCleanErrors(injectValidateSpec(validatePaintProperty));
validateStyleMin.layoutProperty = wrapCleanErrors(injectValidateSpec(validateLayoutProperty));
function injectValidateSpec(validator) {
    return function (options) {
        return validator({
            ...options,
            validateSpec: validate,
        });
    };
}
function sortErrors(errors) {
    return [].concat(errors).sort((a, b) => {
        return a.line - b.line;
    });
}
function wrapCleanErrors(inner) {
    return function (...args) {
        return sortErrors(inner.apply(this, args));
    };
}

// Note: Do not inherit from Error. It breaks when transpiling to ES5.
class ParsingError {
    constructor(error) {
        this.error = error;
        this.message = error.message;
        const match = error.message.match(/line (\d+)/);
        this.line = match ? parseInt(match[1], 10) : 0;
    }
}

const v8 = v8Spec;

function commonjsRequire(path) {
	throw new Error('Could not dynamically require "' + path + '". Please configure the dynamicRequireTargets or/and ignoreDynamicRequires option of @rollup/plugin-commonjs appropriately for this require call to work.');
}

var jsonlint$1 = {};

/* parser generated by jison 0.4.15 */

var hasRequiredJsonlint;

function requireJsonlint () {
	if (hasRequiredJsonlint) return jsonlint$1;
	hasRequiredJsonlint = 1;
	(function (exports) {
		/*
		  Returns a Parser object of the following structure:

		  Parser: {
		    yy: {}
		  }

		  Parser.prototype: {
		    yy: {},
		    trace: function(),
		    symbols_: {associative list: name ==> number},
		    terminals_: {associative list: number ==> name},
		    productions_: [...],
		    performAction: function anonymous(yytext, yyleng, yylineno, yy, yystate, $$, _$),
		    table: [...],
		    defaultActions: {...},
		    parseError: function(str, hash),
		    parse: function(input),

		    lexer: {
		        EOF: 1,
		        parseError: function(str, hash),
		        setInput: function(input),
		        input: function(),
		        unput: function(str),
		        more: function(),
		        less: function(n),
		        pastInput: function(),
		        upcomingInput: function(),
		        showPosition: function(),
		        test_match: function(regex_match_array, rule_index),
		        next: function(),
		        lex: function(),
		        begin: function(condition),
		        popState: function(),
		        _currentRules: function(),
		        topState: function(),
		        pushState: function(condition),

		        options: {
		            ranges: boolean           (optional: true ==> token location info will include a .range[] member)
		            flex: boolean             (optional: true ==> flex-like lexing behaviour where the rules are tested exhaustively to find the longest match)
		            backtrack_lexer: boolean  (optional: true ==> lexer regexes are tested in order and for each matching regex the action code is invoked; the lexer terminates the scan when a token is returned by the action code)
		        },

		        performAction: function(yy, yy_, $avoiding_name_collisions, YY_START),
		        rules: [...],
		        conditions: {associative list: name ==> set},
		    }
		  }


		  token location info (@$, _$, etc.): {
		    first_line: n,
		    last_line: n,
		    first_column: n,
		    last_column: n,
		    range: [start_number, end_number]       (where the numbers are indexes into the input string, regular zero-based)
		  }


		  the parseError function receives a 'hash' object with these members for lexer and parser errors: {
		    text:        (matched text)
		    token:       (the produced terminal token, if any)
		    line:        (yylineno)
		  }
		  while parser (grammar) errors will also provide these members, i.e. parser errors deliver a superset of attributes: {
		    loc:         (yylloc)
		    expected:    (string describing the set of expected tokens)
		    recoverable: (boolean: TRUE when the parser has a error recovery rule available for this particular error)
		  }
		*/
		var parser = (function(){
		var o=function(k,v,o,l){for(o=o||{},l=k.length;l--;o[k[l]]=v);return o},$V0=[1,12],$V1=[1,13],$V2=[1,9],$V3=[1,10],$V4=[1,11],$V5=[1,14],$V6=[1,15],$V7=[14,18,22,24],$V8=[18,22],$V9=[22,24];
		var parser = {trace: function trace() { },
		yy: {},
		symbols_: {"error":2,"JSONString":3,"STRING":4,"JSONNumber":5,"NUMBER":6,"JSONNullLiteral":7,"NULL":8,"JSONBooleanLiteral":9,"TRUE":10,"FALSE":11,"JSONText":12,"JSONValue":13,"EOF":14,"JSONObject":15,"JSONArray":16,"{":17,"}":18,"JSONMemberList":19,"JSONMember":20,":":21,",":22,"[":23,"]":24,"JSONElementList":25,"$accept":0,"$end":1},
		terminals_: {2:"error",4:"STRING",6:"NUMBER",8:"NULL",10:"TRUE",11:"FALSE",14:"EOF",17:"{",18:"}",21:":",22:",",23:"[",24:"]"},
		productions_: [0,[3,1],[5,1],[7,1],[9,1],[9,1],[12,2],[13,1],[13,1],[13,1],[13,1],[13,1],[13,1],[15,2],[15,3],[20,3],[19,1],[19,3],[16,2],[16,3],[25,1],[25,3]],
		performAction: function anonymous(yytext, yyleng, yylineno, yy, yystate /* action[1] */, $$ /* vstack */, _$ /* lstack */) {
		/* this == yyval */

		var $0 = $$.length - 1;
		switch (yystate) {
		case 1:
		 // replace escaped characters with actual character
		          this.$ = new String(yytext.replace(/\\(\\|")/g, "$"+"1")
		                     .replace(/\\n/g,'\n')
		                     .replace(/\\r/g,'\r')
		                     .replace(/\\t/g,'\t')
		                     .replace(/\\v/g,'\v')
		                     .replace(/\\f/g,'\f')
		                     .replace(/\\b/g,'\b'));
		          this.$.__line__ =  this._$.first_line;
		        
		break;
		case 2:

		            this.$ = new Number(yytext);
		            this.$.__line__ =  this._$.first_line;
		        
		break;
		case 3:

		            this.$ = null;
		        
		break;
		case 4:

		            this.$ = new Boolean(true);
		            this.$.__line__ = this._$.first_line;
		        
		break;
		case 5:

		            this.$ = new Boolean(false);
		            this.$.__line__ = this._$.first_line;
		        
		break;
		case 6:
		return this.$ = $$[$0-1];
		case 13:
		this.$ = {}; Object.defineProperty(this.$, '__line__', {
		            value: this._$.first_line,
		            enumerable: false
		        });
		break;
		case 14: case 19:
		this.$ = $$[$0-1]; Object.defineProperty(this.$, '__line__', {
		            value: this._$.first_line,
		            enumerable: false
		        });
		break;
		case 15:
		this.$ = [$$[$0-2], $$[$0]];
		break;
		case 16:
		this.$ = {}; this.$[$$[$0][0]] = $$[$0][1];
		break;
		case 17:
		this.$ = $$[$0-2]; $$[$0-2][$$[$0][0]] = $$[$0][1];
		break;
		case 18:
		this.$ = []; Object.defineProperty(this.$, '__line__', {
		            value: this._$.first_line,
		            enumerable: false
		        });
		break;
		case 20:
		this.$ = [$$[$0]];
		break;
		case 21:
		this.$ = $$[$0-2]; $$[$0-2].push($$[$0]);
		break;
		}
		},
		table: [{3:5,4:$V0,5:6,6:$V1,7:3,8:$V2,9:4,10:$V3,11:$V4,12:1,13:2,15:7,16:8,17:$V5,23:$V6},{1:[3]},{14:[1,16]},o($V7,[2,7]),o($V7,[2,8]),o($V7,[2,9]),o($V7,[2,10]),o($V7,[2,11]),o($V7,[2,12]),o($V7,[2,3]),o($V7,[2,4]),o($V7,[2,5]),o([14,18,21,22,24],[2,1]),o($V7,[2,2]),{3:20,4:$V0,18:[1,17],19:18,20:19},{3:5,4:$V0,5:6,6:$V1,7:3,8:$V2,9:4,10:$V3,11:$V4,13:23,15:7,16:8,17:$V5,23:$V6,24:[1,21],25:22},{1:[2,6]},o($V7,[2,13]),{18:[1,24],22:[1,25]},o($V8,[2,16]),{21:[1,26]},o($V7,[2,18]),{22:[1,28],24:[1,27]},o($V9,[2,20]),o($V7,[2,14]),{3:20,4:$V0,20:29},{3:5,4:$V0,5:6,6:$V1,7:3,8:$V2,9:4,10:$V3,11:$V4,13:30,15:7,16:8,17:$V5,23:$V6},o($V7,[2,19]),{3:5,4:$V0,5:6,6:$V1,7:3,8:$V2,9:4,10:$V3,11:$V4,13:31,15:7,16:8,17:$V5,23:$V6},o($V8,[2,17]),o($V8,[2,15]),o($V9,[2,21])],
		defaultActions: {16:[2,6]},
		parseError: function parseError(str, hash) {
		    if (hash.recoverable) {
		        this.trace(str);
		    } else {
		        throw new Error(str);
		    }
		},
		parse: function parse(input) {
		    var self = this, stack = [0], vstack = [null], lstack = [], table = this.table, yytext = '', yylineno = 0, yyleng = 0, TERROR = 2, EOF = 1;
		    var args = lstack.slice.call(arguments, 1);
		    var lexer = Object.create(this.lexer);
		    var sharedState = { yy: {} };
		    for (var k in this.yy) {
		        if (Object.prototype.hasOwnProperty.call(this.yy, k)) {
		            sharedState.yy[k] = this.yy[k];
		        }
		    }
		    lexer.setInput(input, sharedState.yy);
		    sharedState.yy.lexer = lexer;
		    sharedState.yy.parser = this;
		    if (typeof lexer.yylloc == 'undefined') {
		        lexer.yylloc = {};
		    }
		    var yyloc = lexer.yylloc;
		    lstack.push(yyloc);
		    var ranges = lexer.options && lexer.options.ranges;
		    if (typeof sharedState.yy.parseError === 'function') {
		        this.parseError = sharedState.yy.parseError;
		    } else {
		        this.parseError = Object.getPrototypeOf(this).parseError;
		    }
		    
		        function lex() {
		            var token;
		            token = lexer.lex() || EOF;
		            if (typeof token !== 'number') {
		                token = self.symbols_[token] || token;
		            }
		            return token;
		        }
		    var symbol, state, action, r, yyval = {}, p, len, newState, expected;
		    while (true) {
		        state = stack[stack.length - 1];
		        if (this.defaultActions[state]) {
		            action = this.defaultActions[state];
		        } else {
		            if (symbol === null || typeof symbol == 'undefined') {
		                symbol = lex();
		            }
		            action = table[state] && table[state][symbol];
		        }
		                    if (typeof action === 'undefined' || !action.length || !action[0]) {
		                var errStr = '';
		                expected = [];
		                for (p in table[state]) {
		                    if (this.terminals_[p] && p > TERROR) {
		                        expected.push('\'' + this.terminals_[p] + '\'');
		                    }
		                }
		                if (lexer.showPosition) {
		                    errStr = 'Parse error on line ' + (yylineno + 1) + ':\n' + lexer.showPosition() + '\nExpecting ' + expected.join(', ') + ', got \'' + (this.terminals_[symbol] || symbol) + '\'';
		                } else {
		                    errStr = 'Parse error on line ' + (yylineno + 1) + ': Unexpected ' + (symbol == EOF ? 'end of input' : '\'' + (this.terminals_[symbol] || symbol) + '\'');
		                }
		                this.parseError(errStr, {
		                    text: lexer.match,
		                    token: this.terminals_[symbol] || symbol,
		                    line: lexer.yylineno,
		                    loc: yyloc,
		                    expected: expected
		                });
		            }
		        if (action[0] instanceof Array && action.length > 1) {
		            throw new Error('Parse Error: multiple actions possible at state: ' + state + ', token: ' + symbol);
		        }
		        switch (action[0]) {
		        case 1:
		            stack.push(symbol);
		            vstack.push(lexer.yytext);
		            lstack.push(lexer.yylloc);
		            stack.push(action[1]);
		            symbol = null;
		            {
		                yyleng = lexer.yyleng;
		                yytext = lexer.yytext;
		                yylineno = lexer.yylineno;
		                yyloc = lexer.yylloc;
		            }
		            break;
		        case 2:
		            len = this.productions_[action[1]][1];
		            yyval.$ = vstack[vstack.length - len];
		            yyval._$ = {
		                first_line: lstack[lstack.length - (len || 1)].first_line,
		                last_line: lstack[lstack.length - 1].last_line,
		                first_column: lstack[lstack.length - (len || 1)].first_column,
		                last_column: lstack[lstack.length - 1].last_column
		            };
		            if (ranges) {
		                yyval._$.range = [
		                    lstack[lstack.length - (len || 1)].range[0],
		                    lstack[lstack.length - 1].range[1]
		                ];
		            }
		            r = this.performAction.apply(yyval, [
		                yytext,
		                yyleng,
		                yylineno,
		                sharedState.yy,
		                action[1],
		                vstack,
		                lstack
		            ].concat(args));
		            if (typeof r !== 'undefined') {
		                return r;
		            }
		            if (len) {
		                stack = stack.slice(0, -1 * len * 2);
		                vstack = vstack.slice(0, -1 * len);
		                lstack = lstack.slice(0, -1 * len);
		            }
		            stack.push(this.productions_[action[1]][0]);
		            vstack.push(yyval.$);
		            lstack.push(yyval._$);
		            newState = table[stack[stack.length - 2]][stack[stack.length - 1]];
		            stack.push(newState);
		            break;
		        case 3:
		            return true;
		        }
		    }
		    return true;
		}};
		/* generated by jison-lex 0.3.4 */
		var lexer = (function(){
		var lexer = ({

		EOF:1,

		parseError:function parseError(str, hash) {
		        if (this.yy.parser) {
		            this.yy.parser.parseError(str, hash);
		        } else {
		            throw new Error(str);
		        }
		    },

		// resets the lexer, sets new input
		setInput:function (input, yy) {
		        this.yy = yy || this.yy || {};
		        this._input = input;
		        this._more = this._backtrack = this.done = false;
		        this.yylineno = this.yyleng = 0;
		        this.yytext = this.matched = this.match = '';
		        this.conditionStack = ['INITIAL'];
		        this.yylloc = {
		            first_line: 1,
		            first_column: 0,
		            last_line: 1,
		            last_column: 0
		        };
		        if (this.options.ranges) {
		            this.yylloc.range = [0,0];
		        }
		        this.offset = 0;
		        return this;
		    },

		// consumes and returns one char from the input
		input:function () {
		        var ch = this._input[0];
		        this.yytext += ch;
		        this.yyleng++;
		        this.offset++;
		        this.match += ch;
		        this.matched += ch;
		        var lines = ch.match(/(?:\r\n?|\n).*/g);
		        if (lines) {
		            this.yylineno++;
		            this.yylloc.last_line++;
		        } else {
		            this.yylloc.last_column++;
		        }
		        if (this.options.ranges) {
		            this.yylloc.range[1]++;
		        }

		        this._input = this._input.slice(1);
		        return ch;
		    },

		// unshifts one char (or a string) into the input
		unput:function (ch) {
		        var len = ch.length;
		        var lines = ch.split(/(?:\r\n?|\n)/g);

		        this._input = ch + this._input;
		        this.yytext = this.yytext.substr(0, this.yytext.length - len);
		        //this.yyleng -= len;
		        this.offset -= len;
		        var oldLines = this.match.split(/(?:\r\n?|\n)/g);
		        this.match = this.match.substr(0, this.match.length - 1);
		        this.matched = this.matched.substr(0, this.matched.length - 1);

		        if (lines.length - 1) {
		            this.yylineno -= lines.length - 1;
		        }
		        var r = this.yylloc.range;

		        this.yylloc = {
		            first_line: this.yylloc.first_line,
		            last_line: this.yylineno + 1,
		            first_column: this.yylloc.first_column,
		            last_column: lines ?
		                (lines.length === oldLines.length ? this.yylloc.first_column : 0)
		                 + oldLines[oldLines.length - lines.length].length - lines[0].length :
		              this.yylloc.first_column - len
		        };

		        if (this.options.ranges) {
		            this.yylloc.range = [r[0], r[0] + this.yyleng - len];
		        }
		        this.yyleng = this.yytext.length;
		        return this;
		    },

		// When called from action, caches matched text and appends it on next action
		more:function () {
		        this._more = true;
		        return this;
		    },

		// When called from action, signals the lexer that this rule fails to match the input, so the next matching rule (regex) should be tested instead.
		reject:function () {
		        if (this.options.backtrack_lexer) {
		            this._backtrack = true;
		        } else {
		            return this.parseError('Lexical error on line ' + (this.yylineno + 1) + '. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).\n' + this.showPosition(), {
		                text: "",
		                token: null,
		                line: this.yylineno
		            });

		        }
		        return this;
		    },

		// retain first n characters of the match
		less:function (n) {
		        this.unput(this.match.slice(n));
		    },

		// displays already matched input, i.e. for error messages
		pastInput:function () {
		        var past = this.matched.substr(0, this.matched.length - this.match.length);
		        return (past.length > 20 ? '...':'') + past.substr(-20).replace(/\n/g, "");
		    },

		// displays upcoming input, i.e. for error messages
		upcomingInput:function () {
		        var next = this.match;
		        if (next.length < 20) {
		            next += this._input.substr(0, 20-next.length);
		        }
		        return (next.substr(0,20) + (next.length > 20 ? '...' : '')).replace(/\n/g, "");
		    },

		// displays the character position where the lexing error occurred, i.e. for error messages
		showPosition:function () {
		        var pre = this.pastInput();
		        var c = new Array(pre.length + 1).join("-");
		        return pre + this.upcomingInput() + "\n" + c + "^";
		    },

		// test the lexed token: return FALSE when not a match, otherwise return token
		test_match:function (match, indexed_rule) {
		        var token,
		            lines,
		            backup;

		        if (this.options.backtrack_lexer) {
		            // save context
		            backup = {
		                yylineno: this.yylineno,
		                yylloc: {
		                    first_line: this.yylloc.first_line,
		                    last_line: this.last_line,
		                    first_column: this.yylloc.first_column,
		                    last_column: this.yylloc.last_column
		                },
		                yytext: this.yytext,
		                match: this.match,
		                matches: this.matches,
		                matched: this.matched,
		                yyleng: this.yyleng,
		                offset: this.offset,
		                _more: this._more,
		                _input: this._input,
		                yy: this.yy,
		                conditionStack: this.conditionStack.slice(0),
		                done: this.done
		            };
		            if (this.options.ranges) {
		                backup.yylloc.range = this.yylloc.range.slice(0);
		            }
		        }

		        lines = match[0].match(/(?:\r\n?|\n).*/g);
		        if (lines) {
		            this.yylineno += lines.length;
		        }
		        this.yylloc = {
		            first_line: this.yylloc.last_line,
		            last_line: this.yylineno + 1,
		            first_column: this.yylloc.last_column,
		            last_column: lines ?
		                         lines[lines.length - 1].length - lines[lines.length - 1].match(/\r?\n?/)[0].length :
		                         this.yylloc.last_column + match[0].length
		        };
		        this.yytext += match[0];
		        this.match += match[0];
		        this.matches = match;
		        this.yyleng = this.yytext.length;
		        if (this.options.ranges) {
		            this.yylloc.range = [this.offset, this.offset += this.yyleng];
		        }
		        this._more = false;
		        this._backtrack = false;
		        this._input = this._input.slice(match[0].length);
		        this.matched += match[0];
		        token = this.performAction.call(this, this.yy, this, indexed_rule, this.conditionStack[this.conditionStack.length - 1]);
		        if (this.done && this._input) {
		            this.done = false;
		        }
		        if (token) {
		            return token;
		        } else if (this._backtrack) {
		            // recover context
		            for (var k in backup) {
		                this[k] = backup[k];
		            }
		            return false; // rule action called reject() implying the next rule should be tested instead.
		        }
		        return false;
		    },

		// return next match in input
		next:function () {
		        if (this.done) {
		            return this.EOF;
		        }
		        if (!this._input) {
		            this.done = true;
		        }

		        var token,
		            match,
		            tempMatch,
		            index;
		        if (!this._more) {
		            this.yytext = '';
		            this.match = '';
		        }
		        var rules = this._currentRules();
		        for (var i = 0; i < rules.length; i++) {
		            tempMatch = this._input.match(this.rules[rules[i]]);
		            if (tempMatch && (!match || tempMatch[0].length > match[0].length)) {
		                match = tempMatch;
		                index = i;
		                if (this.options.backtrack_lexer) {
		                    token = this.test_match(tempMatch, rules[i]);
		                    if (token !== false) {
		                        return token;
		                    } else if (this._backtrack) {
		                        match = false;
		                        continue; // rule action called reject() implying a rule MISmatch.
		                    } else {
		                        // else: this is a lexer rule which consumes input without producing a token (e.g. whitespace)
		                        return false;
		                    }
		                } else if (!this.options.flex) {
		                    break;
		                }
		            }
		        }
		        if (match) {
		            token = this.test_match(match, rules[index]);
		            if (token !== false) {
		                return token;
		            }
		            // else: this is a lexer rule which consumes input without producing a token (e.g. whitespace)
		            return false;
		        }
		        if (this._input === "") {
		            return this.EOF;
		        } else {
		            return this.parseError('Lexical error on line ' + (this.yylineno + 1) + '. Unrecognized text.\n' + this.showPosition(), {
		                text: "",
		                token: null,
		                line: this.yylineno
		            });
		        }
		    },

		// return next match that has a token
		lex:function lex() {
		        var r = this.next();
		        if (r) {
		            return r;
		        } else {
		            return this.lex();
		        }
		    },

		// activates a new lexer condition state (pushes the new lexer condition state onto the condition stack)
		begin:function begin(condition) {
		        this.conditionStack.push(condition);
		    },

		// pop the previously active lexer condition state off the condition stack
		popState:function popState() {
		        var n = this.conditionStack.length - 1;
		        if (n > 0) {
		            return this.conditionStack.pop();
		        } else {
		            return this.conditionStack[0];
		        }
		    },

		// produce the lexer rule set which is active for the currently active lexer condition state
		_currentRules:function _currentRules() {
		        if (this.conditionStack.length && this.conditionStack[this.conditionStack.length - 1]) {
		            return this.conditions[this.conditionStack[this.conditionStack.length - 1]].rules;
		        } else {
		            return this.conditions["INITIAL"].rules;
		        }
		    },

		// return the currently active lexer condition state; when an index argument is provided it produces the N-th previous condition state, if available
		topState:function topState(n) {
		        n = this.conditionStack.length - 1 - Math.abs(n || 0);
		        if (n >= 0) {
		            return this.conditionStack[n];
		        } else {
		            return "INITIAL";
		        }
		    },

		// alias for begin(condition)
		pushState:function pushState(condition) {
		        this.begin(condition);
		    },

		// return the number of states currently on the stack
		stateStackSize:function stateStackSize() {
		        return this.conditionStack.length;
		    },
		options: {},
		performAction: function anonymous(yy,yy_,$avoiding_name_collisions,YY_START) {
		switch($avoiding_name_collisions) {
		case 0:/* skip whitespace */
		break;
		case 1:return 6
		case 2:yy_.yytext = yy_.yytext.substr(1,yy_.yyleng-2); return 4
		case 3:return 17
		case 4:return 18
		case 5:return 23
		case 6:return 24
		case 7:return 22
		case 8:return 21
		case 9:return 10
		case 10:return 11
		case 11:return 8
		case 12:return 14
		case 13:return 'INVALID'
		}
		},
		rules: [/^(?:\s+)/,/^(?:(-?([0-9]|[1-9][0-9]+))(\.[0-9]+)?([eE][-+]?[0-9]+)?\b)/,/^(?:"(?:\\[\\"bfnrt/]|\\u[a-fA-F0-9]{4}|[^\\\0-\x09\x0a-\x1f"])*")/,/^(?:\{)/,/^(?:\})/,/^(?:\[)/,/^(?:\])/,/^(?:,)/,/^(?::)/,/^(?:true\b)/,/^(?:false\b)/,/^(?:null\b)/,/^(?:$)/,/^(?:.)/],
		conditions: {"INITIAL":{"rules":[0,1,2,3,4,5,6,7,8,9,10,11,12,13],"inclusive":true}}
		});
		return lexer;
		})();
		parser.lexer = lexer;
		function Parser () {
		  this.yy = {};
		}
		Parser.prototype = parser;parser.Parser = Parser;
		return new Parser;
		})();


		if (typeof commonjsRequire !== 'undefined' && 'object' !== 'undefined') {
		exports.parser = parser;
		exports.Parser = parser.Parser;
		exports.parse = function () { return parser.parse.apply(parser, arguments); };
		} 
	} (jsonlint$1));
	return jsonlint$1;
}

var jsonlintExports = requireJsonlint();
var jsonlint = /*@__PURE__*/getDefaultExportFromCjs(jsonlintExports);

function readStyle(style) {
    if (style instanceof String || typeof style === 'string' || style instanceof Buffer) {
        try {
            return jsonlint.parse(style.toString());
        }
        catch (e) {
            throw new ParsingError(e);
        }
    }
    return style;
}

/**
 * Validate a MapLibre GL style against the style specification.
 *
 * @param style - The style to be validated. If a `String` or `Buffer` is provided,
 * the returned errors will contain line numbers.
 * @param styleSpec - The style specification to validate against.
 * If omitted, the spec version is inferred from the stylesheet.
 * @returns an array of errors, or an empty array if no errors are found.
 * @example
 *   const validate = require('maplibre-gl-style-spec').validate;
 *   const style = fs.readFileSync('./style.json', 'utf8');
 *   const errors = validate(style);
 */
function validateStyle(style, styleSpec = v8) {
    let s = style;
    try {
        s = readStyle(s);
    }
    catch (e) {
        return [e];
    }
    return validateStyleMin(s, styleSpec);
}

const argv = minimist(process.argv.slice(2), {
    boolean: 'json',
});

if (argv.help || argv.h || (!argv._.length && process.stdin.isTTY)) {
    help();
} else {
    let status = 0;

    if (!argv._.length) {
        argv._.push('/dev/stdin');
    }

    argv._.forEach((file) => {
        const errors = validateStyle(rw.readFileSync(file, 'utf8'));
        if (errors.length) {
            if (argv.json) {
                process.stdout.write(JSON.stringify(errors, null, 2));
            } else {
                errors.forEach((e) => {
                    console.log('%s:%d: %s', file, e.line, e.message);
                });
            }
            status = 1;
        }
    });

    process.exit(status);
}

function help() {
    console.log('usage:');
    console.log('  gl-style-validate file.json');
    console.log('  gl-style-validate < file.json');
    console.log('');
    console.log('options:');
    console.log('--json  output errors as json');
}
//# sourceMappingURL=gl-style-validate.mjs.map
