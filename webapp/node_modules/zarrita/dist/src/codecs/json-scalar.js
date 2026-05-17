/*
Utilities for parsing Zarr V3 JSON-encoded scalars.

The Zarr V3 spec encodes numeric scalars in JSON as either:
  - a plain JSON number
  - a special float string: "NaN", "Infinity", "-Infinity"
  - a hex-encoded float string: "0x..." (raw IEEE 754 bytes as an unsigned integer)

See https://github.com/zarr-developers/zarr-specs/blob/main/docs/v3/data-types/index.rst#permitted-fill-values
*/
import { InvalidMetadataError, UnsupportedError } from "../errors.js";
const SPECIAL_FLOATS = {
    NaN: NaN,
    Infinity: Infinity,
    "-Infinity": -Infinity,
};
const FLOAT_BYTES = {
    float16: 2,
    float32: 4,
    float64: 8,
};
export function isFloatType(dataType) {
    return dataType in FLOAT_BYTES;
}
export function isBigintType(dataType) {
    return dataType === "int64" || dataType === "uint64";
}
/*
Reinterpret a hex-encoded integer as a float of the given byte width.
This is necessary because the Zarr V3 spec allows floats to declare their
fill value as a hex string representing the raw bytes of the float.
*/
function hexToFloat(hex, byteWidth) {
    const int = BigInt(hex);
    const buf = new ArrayBuffer(byteWidth);
    const view = new DataView(buf);
    if (byteWidth === 2) {
        if (typeof view.getFloat16 !== "function") {
            throw new UnsupportedError("float16 hex-encoded scalar decoding (requires DataView.prototype.getFloat16)");
        }
        view.setUint16(0, Number(int));
        return view.getFloat16(0);
    }
    if (byteWidth === 4) {
        view.setUint32(0, Number(int));
        return view.getFloat32(0);
    }
    view.setBigUint64(0, int);
    return view.getFloat64(0);
}
/**
 * Parse a JSON-encoded scalar value into its native JS representation
 * for the given Zarr data type.
 *
 * - Integer types: value must be a JSON number that is an integer.
 * - Bigint types (int64/uint64): value must be a JSON number that is an integer, converted to BigInt.
 * - Float types: value may be a number, a special float string, or a hex-encoded float.
 */
export function parseJsonScalar(dataType, value) {
    if (isBigintType(dataType)) {
        if (typeof value !== "number" || !Number.isInteger(value)) {
            throw new InvalidMetadataError(`Expected an integer value for data type "${dataType}", got ${JSON.stringify(value)}`);
        }
        return BigInt(value);
    }
    if (typeof value === "number") {
        if (!isFloatType(dataType) && !Number.isInteger(value)) {
            throw new InvalidMetadataError(`Expected an integer value for data type "${dataType}", got ${value}`);
        }
        return value;
    }
    // value is SpecialFloat | HexString — only valid for float types
    if (!isFloatType(dataType)) {
        throw new InvalidMetadataError(`String-encoded scalar "${value}" is not valid for non-float data type "${dataType}"`);
    }
    if (value in SPECIAL_FLOATS) {
        return SPECIAL_FLOATS[value];
    }
    return hexToFloat(value, FLOAT_BYTES[dataType]);
}
//# sourceMappingURL=json-scalar.js.map