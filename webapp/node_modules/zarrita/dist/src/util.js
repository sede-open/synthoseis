import { InvalidMetadataError } from "./errors.js";
import { BoolArray, ByteStringArray, UnicodeStringArray, } from "./typedarray.js";
export function jsonEncodeObject(o) {
    const str = JSON.stringify(o, (_key, value) => {
        // JSON.stringify converts NaN/Infinity/-Infinity to null.
        // Zarr v3 spec requires these as string representations.
        if (typeof value === "number") {
            if (Number.isNaN(value))
                return "NaN";
            if (value === Infinity)
                return "Infinity";
            if (value === -Infinity)
                return "-Infinity";
        }
        return value;
    }, 2);
    return new TextEncoder().encode(str);
}
export function assertSharedArrayBufferAvailable() {
    if (typeof SharedArrayBuffer === "undefined") {
        throw new Error("SharedArrayBuffer is not available. " +
            "In browsers, this requires Cross-Origin-Opener-Policy and " +
            "Cross-Origin-Embedder-Policy headers to be set.");
    }
}
export function createBuffer(byteLength, useShared) {
    if (useShared) {
        return new SharedArrayBuffer(byteLength);
    }
    return new ArrayBuffer(byteLength);
}
export function jsonDecodeObject(bytes) {
    const str = new TextDecoder().decode(bytes);
    try {
        return JSON.parse(str);
    }
    catch (cause) {
        throw new InvalidMetadataError("Failed to decode JSON", { cause });
    }
}
export function byteswapInplace(view, bytesPerElement) {
    const numFlips = bytesPerElement / 2;
    const endByteIndex = bytesPerElement - 1;
    let t = 0;
    for (let i = 0; i < view.length; i += bytesPerElement) {
        for (let j = 0; j < numFlips; j += 1) {
            t = view[i + j];
            view[i + j] = view[i + endByteIndex - j];
            view[i + endByteIndex - j] = t;
        }
    }
}
export function getCtr(dataType) {
    if (dataType === "v2:object") {
        return globalThis.Array;
    }
    let match = dataType.match(/v2:([US])(\d+)/);
    if (match) {
        let [, kind, chars] = match;
        // @ts-expect-error
        return (kind === "U" ? UnicodeStringArray : ByteStringArray).bind(null, Number(chars));
    }
    // Handle v3 variable-length string type
    if (dataType === "string") {
        return globalThis.Array;
    }
    // @ts-expect-error - We've checked that the key exists
    let ctr = {
        int8: Int8Array,
        int16: Int16Array,
        int32: Int32Array,
        int64: globalThis.BigInt64Array,
        uint8: Uint8Array,
        uint16: Uint16Array,
        uint32: Uint32Array,
        uint64: globalThis.BigUint64Array,
        float16: globalThis.Float16Array,
        float32: Float32Array,
        float64: Float64Array,
        bool: BoolArray,
    }[dataType];
    if (!ctr) {
        throw new InvalidMetadataError(`Unknown or unsupported dataType: ${dataType}`);
    }
    return ctr;
}
/** Compute strides for 'C' or 'F' ordered array from shape */
export function getStrides(shape, order) {
    const rank = shape.length;
    if (typeof order === "string") {
        order =
            order === "C"
                ? Array.from({ length: rank }, (_, i) => i) // Row-major (identity order)
                : Array.from({ length: rank }, (_, i) => rank - 1 - i); // Column-major (reverse order)
    }
    assert(rank === order.length, "Order length must match the number of dimensions.");
    let step = 1;
    let stride = new Array(rank);
    for (let i = order.length - 1; i >= 0; i--) {
        stride[order[i]] = step;
        step *= shape[order[i]];
    }
    return stride;
}
// https://zarr-specs.readthedocs.io/en/latest/v3/core/v3.0.html#chunk-key-encoding
export function createChunkKeyEncoder({ name, configuration, }) {
    if (name === "default") {
        const separator = configuration?.separator ?? "/";
        return (chunkCoords) => ["c", ...chunkCoords].join(separator);
    }
    if (name === "v2") {
        const separator = configuration?.separator ?? ".";
        return (chunkCoords) => chunkCoords.join(separator) || "0";
    }
    throw new InvalidMetadataError(`Unknown chunk key encoding: ${name}`);
}
function coerceDtype(dtype) {
    if (dtype === "|O") {
        return { dataType: "v2:object" };
    }
    let match = dtype.match(/^([<|>])(.*)$/);
    if (!match) {
        throw new InvalidMetadataError(`Invalid dtype: ${dtype}`);
    }
    let [, endian, rest] = match;
    let dataType = {
        b1: "bool",
        i1: "int8",
        u1: "uint8",
        i2: "int16",
        u2: "uint16",
        i4: "int32",
        u4: "uint32",
        i8: "int64",
        u8: "uint64",
        f2: "float16",
        f4: "float32",
        f8: "float64",
    }[rest] ??
        (rest.startsWith("S") || rest.startsWith("U") ? `v2:${rest}` : undefined);
    if (!dataType) {
        throw new InvalidMetadataError(`Unsupported or unknown dtype: ${dtype}`);
    }
    if (endian === "|") {
        return { dataType };
    }
    return { dataType, endian: endian === "<" ? "little" : "big" };
}
function isFixedScaleOffsetConfig(filter) {
    return ((filter.id === "fixedscaleoffset" ||
        filter.id === "numcodecs.fixedscaleoffset") &&
        typeof filter.scale === "number" &&
        typeof filter.offset === "number" &&
        (filter.astype === undefined || typeof filter.astype === "string") &&
        (filter.dtype === undefined || typeof filter.dtype === "string"));
}
export function v2ToV3ArrayMetadata(meta, attributes = {}) {
    let codecs = [];
    let dtype = coerceDtype(meta.dtype);
    if (meta.order === "F") {
        codecs.push({ name: "transpose", configuration: { order: "F" } });
    }
    for (let filter of meta.filters ?? []) {
        // Translate the numcodecs `fixedscaleoffset` filter into the native v3
        // `scale_offset` + `cast_value` pair from #395. The v2 filter is not
        // part of the zarr v3 spec (see discussion in
        // https://github.com/manzt/zarrita.js/pull/312), but together these
        // two codecs implement the same decode semantics:
        //   (enc / scale + offset).astype(dtype)
        // where `dtype` is the logical (decoded) data type the user sees and
        // `astype` is the quantized on-disk data type.
        if (filter.id === "fixedscaleoffset" ||
            filter.id === "numcodecs.fixedscaleoffset") {
            if (!isFixedScaleOffsetConfig(filter)) {
                throw new InvalidMetadataError(`Invalid fixedscaleoffset filter: ${JSON.stringify(filter)}`);
            }
            codecs.push({
                name: "scale_offset",
                configuration: {
                    scale: filter.scale,
                    offset: filter.offset,
                },
            });
            // `astype` defaults to `dtype` in numcodecs, meaning an identity
            // cast. Skip `cast_value` entirely in that case — and also when
            // `astype` equals the logical v2 dtype, since there's nothing to
            // convert.
            let astype = filter.astype ?? filter.dtype;
            if (astype !== undefined && astype !== meta.dtype) {
                let castTarget = coerceDtype(astype).dataType;
                if (!isDataType(castTarget, "number") &&
                    !isDataType(castTarget, "bigint")) {
                    throw new InvalidMetadataError(`fixedscaleoffset astype must be a numeric data type, got ${astype}`);
                }
                codecs.push({
                    name: "cast_value",
                    configuration: {
                        data_type: castTarget,
                        // `np.around` uses banker's rounding (round-half-to-even).
                        rounding: "nearest-even",
                        // Matches de-facto numpy integer-overflow behavior.
                        out_of_range: "wrap",
                    },
                });
            }
            continue;
        }
        let { id, ...configuration } = filter;
        codecs.push({ name: `numcodecs.${id}`, configuration });
    }
    // The `bytes` codec must come *after* any array-to-array codecs that
    // change the data type (e.g. `cast_value`) so that the pipeline's
    // currentMeta has been threaded through `getEncodedMeta` by the time
    // `BytesCodec.fromConfig` sees it. Relative order does not matter for
    // type-preserving codecs like `delta` or `transpose`.
    if ("endian" in dtype && dtype.endian === "big") {
        codecs.push({ name: "bytes", configuration: { endian: "big" } });
    }
    if (meta.compressor) {
        let { id, ...configuration } = meta.compressor;
        codecs.push({ name: `numcodecs.${id}`, configuration });
    }
    let dimensionNames;
    if (globalThis.Array.isArray(attributes._ARRAY_DIMENSIONS)) {
        dimensionNames = attributes._ARRAY_DIMENSIONS;
    }
    return {
        zarr_format: 3,
        node_type: "array",
        shape: meta.shape,
        data_type: dtype.dataType,
        chunk_grid: {
            name: "regular",
            configuration: {
                chunk_shape: meta.chunks,
            },
        },
        chunk_key_encoding: {
            name: "v2",
            configuration: {
                separator: meta.dimension_separator ?? ".",
            },
        },
        codecs,
        fill_value: meta.fill_value,
        dimension_names: dimensionNames,
        attributes,
    };
}
export function v2ToV3GroupMetadata(_meta, attributes = {}) {
    return {
        zarr_format: 3,
        node_type: "group",
        attributes,
    };
}
export function isDataType(dtype, query) {
    if (query !== "number" &&
        query !== "bigint" &&
        query !== "boolean" &&
        query !== "object" &&
        query !== "string") {
        return dtype === query;
    }
    let isBoolean = dtype === "bool";
    if (query === "boolean")
        return isBoolean;
    let isString = dtype.startsWith("v2:U") || dtype.startsWith("v2:S") || dtype === "string";
    if (query === "string")
        return isString;
    let isBigint = dtype === "int64" || dtype === "uint64";
    if (query === "bigint")
        return isBigint;
    let isObject = dtype === "v2:object";
    if (query === "object")
        return isObject;
    return !isString && !isBigint && !isBoolean && !isObject;
}
export function isShardingCodec(codec) {
    return codec?.name === "sharding_indexed";
}
export function ensureCorrectScalar(metadata) {
    if ((metadata.data_type === "uint64" || metadata.data_type === "int64") &&
        metadata.fill_value != null) {
        // @ts-expect-error - We've narrowed the type of fill_value correctly
        return BigInt(metadata.fill_value);
    }
    // Zarr v3 represents IEEE 754 special float values as strings in JSON.
    // Only applies to floating-point types.
    let isFloat = metadata.data_type === "float16" ||
        metadata.data_type === "float32" ||
        metadata.data_type === "float64";
    if (typeof metadata.fill_value === "string" && isFloat) {
        let mapping = {
            NaN: NaN,
            Infinity: Infinity,
            "-Infinity": -Infinity,
        };
        if (metadata.fill_value in mapping) {
            return mapping[metadata.fill_value];
        }
    }
    return metadata.fill_value;
}
/**
 * Ensures an error matches expected type(s), otherwise rethrows.
 *
 * Unmatched errors bubble up, like Python's `except`. Narrows error types for
 * type-safe property access.
 *
 * @see {@link https://gist.github.com/manzt/3702f19abb714e21c22ce48851c75abf}
 *
 * @example
 * ```ts
 * class DatabaseError extends Error { }
 * class NetworkError extends Error { }
 *
 * try {
 *   await db.query();
 * } catch (err) {
 *   rethrowUnless(err, DatabaseError, NetworkError);
 *   err // DatabaseError | NetworkError
 * }
 * ```
 *
 * @param error - The error to check
 * @param errors - Expected error type(s)
 * @throws The original error if it doesn't match expected type(s)
 */
/**
 * Merge a first-class `signal` with the deprecated `opts.signal` shim.
 * If both are set, the signals are combined via `AbortSignal.any` so that
 * aborting either cancels the request.
 */
export function resolveSignal(opts) {
    let a = opts.signal;
    let b = opts.opts?.signal;
    if (a && b)
        return AbortSignal.any([a, b]);
    return a ?? b;
}
export function rethrowUnless(error, ...errors) {
    if (!errors.some((ErrorClass) => error instanceof ErrorClass)) {
        throw error;
    }
}
/**
 * Make an assertion.
 *
 * Usage
 * @example
 * ```ts
 * const value: boolean = Math.random() <= 0.5;
 * assert(value, "value is greater than than 0.5!");
 * value // true
 * ```
 *
 * @param expression - The expression to test.
 * @param msg - The optional message to display if the assertion fails.
 * @throws an {@link Error} if `expression` is not truthy.
 */
export function assert(expression, msg = "") {
    if (!expression) {
        throw new Error(msg);
    }
}
/**
 * Decompress data using the given format via the Web Streams API.
 * Views backed by a SharedArrayBuffer are copied into a regular ArrayBuffer
 * since the Response constructor does not accept shared memory.
 */
export async function decompress(data, { format, signal }) {
    let response;
    if (data instanceof ArrayBuffer) {
        response = new Response(data);
    }
    else {
        let bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        response = new Response(bytes.slice().buffer);
    }
    assert(response.body, "Response does not contain body.");
    try {
        const decompressedResponse = new Response(response.body.pipeThrough(new DecompressionStream(format), { signal }));
        const buffer = await decompressedResponse.arrayBuffer();
        return buffer;
    }
    catch {
        signal?.throwIfAborted();
        throw new Error(`Failed to decode ${format}`);
    }
}
//# sourceMappingURL=util.js.map