import { assert, getStrides, jsonDecodeObject } from "../util.js";
// Reference: https://stackoverflow.com/a/21897413
function throwOnNanReplacer(_key, value) {
    assert(!Number.isNaN(value), "JsonCodec allow_nan is false but NaN was encountered during encoding.");
    assert(value !== Number.POSITIVE_INFINITY, "JsonCodec allow_nan is false but Infinity was encountered during encoding.");
    assert(value !== Number.NEGATIVE_INFINITY, "JsonCodec allow_nan is false but -Infinity was encountered during encoding.");
    return value;
}
// Reference: https://gist.github.com/davidfurlong/463a83a33b70a3b6618e97ec9679e490
function sortKeysReplacer(_key, value) {
    return value instanceof Object && !Array.isArray(value)
        ? Object.keys(value)
            .sort()
            .reduce((sorted, key) => {
            sorted[key] = value[key];
            return sorted;
        }, {})
        : value;
}
export class JsonCodec {
    configuration;
    kind = "array_to_bytes";
    #encoderConfig;
    #decoderConfig;
    constructor(configuration = {}) {
        this.configuration = configuration;
        // Reference: https://github.com/zarr-developers/numcodecs/blob/0878717a3613d91a453fe3d3716aa9c67c023a8b/numcodecs/json.py#L36
        const { encoding = "utf-8", skipkeys = false, ensure_ascii = true, check_circular = true, allow_nan = true, sort_keys = true, indent, strict = true, } = configuration;
        let separators = configuration.separators;
        if (!separators) {
            // ensure separators are explicitly specified, and consistent behaviour across
            // Python versions, and most compact representation if indent is None
            if (!indent) {
                separators = [",", ":"];
            }
            else {
                separators = [", ", ": "];
            }
        }
        this.#encoderConfig = {
            encoding,
            skipkeys,
            ensure_ascii,
            check_circular,
            allow_nan,
            indent,
            separators,
            sort_keys,
        };
        this.#decoderConfig = { strict };
    }
    static fromConfig(configuration) {
        return new JsonCodec(configuration);
    }
    encode(buf) {
        const { indent, encoding, ensure_ascii, check_circular, allow_nan, sort_keys, } = this.#encoderConfig;
        assert(encoding === "utf-8", "JsonCodec does not yet support non-utf-8 encoding.");
        const replacerFunctions = [];
        // By default, for JSON.stringify,
        // a TypeError will be thrown if one attempts to encode an object with circular references
        assert(check_circular, "JsonCodec does not yet support skipping the check for circular references during encoding.");
        if (!allow_nan) {
            // Throw if NaN/Infinity/-Infinity are encountered during encoding.
            replacerFunctions.push(throwOnNanReplacer);
        }
        if (sort_keys) {
            // We can ensure keys are sorted but not really the opposite since
            // there is no guarantee of key ordering in JS.
            replacerFunctions.push(sortKeysReplacer);
        }
        const items = Array.from(buf.data);
        items.push("|O");
        items.push(buf.shape);
        let replacer;
        if (replacerFunctions.length) {
            replacer = (key, value) => {
                let newValue = value;
                for (let subReplacer of replacerFunctions) {
                    newValue = subReplacer(key, newValue);
                }
                return newValue;
            };
        }
        let jsonStr = JSON.stringify(items, replacer, indent);
        if (ensure_ascii) {
            // If ensure_ascii is true (the default), the output is guaranteed
            // to have all incoming non-ASCII characters escaped.
            // If ensure_ascii is false, these characters will be output as-is.
            // Reference: https://stackoverflow.com/a/31652607
            jsonStr = jsonStr.replace(/[\u007F-\uFFFF]/g, (chr) => {
                const fullStr = `0000${chr.charCodeAt(0).toString(16)}`;
                const subStr = fullStr.substring(fullStr.length - 4);
                return `\\u${subStr}`;
            });
        }
        return new TextEncoder().encode(jsonStr);
    }
    decode(bytes) {
        const { strict } = this.#decoderConfig;
        // (i.e., allowing control characters inside strings)
        assert(strict, "JsonCodec does not yet support non-strict decoding.");
        const items = jsonDecodeObject(bytes);
        const shape = items.pop();
        items.pop(); // Pop off dtype (unused)
        // O-d case
        assert(shape, "0D not implemented for JsonCodec.");
        const stride = getStrides(shape, "C");
        const data = items;
        return { data, shape, stride };
    }
}
//# sourceMappingURL=json2.js.map