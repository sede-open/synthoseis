import { InvalidMetadataError } from "../errors.js";
import { getCtr } from "../util.js";
const SUPPORTED = new Set([
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
]);
// The python codecs flatten with arr.reshape(-1, order='A').
// The zarrita codec uses the byte order of the data.
// These two are only equivalent if the array strides are C style or Fortran style.
function assertCOrFContiguous(shape, stride) {
    const n = shape.length;
    let c = n === 0 || stride[n - 1] === 1;
    for (let i = n - 2; i >= 0 && c; i--) {
        c = stride[i] === stride[i + 1] * shape[i + 1];
    }
    if (c)
        return;
    let f = n === 0 || stride[0] === 1;
    for (let i = 1; i < n && f; i++) {
        f = stride[i] === stride[i - 1] * shape[i - 1];
    }
    if (f)
        return;
    throw new Error(`DeltaCodec requires C- or Fortran-contiguous strides, got shape=${JSON.stringify(shape)} stride=${JSON.stringify(stride)}`);
}
export class DeltaCodec {
    kind = "array_to_array";
    #ctr;
    constructor(ctr) {
        this.#ctr = ctr;
    }
    static fromConfig(_config, meta) {
        if (!SUPPORTED.has(meta.dataType)) {
            throw new InvalidMetadataError(`Delta codec does not support data type: ${meta.dataType}`);
        }
        return new DeltaCodec(getCtr(meta.dataType));
    }
    encode(chunk) {
        assertCOrFContiguous(chunk.shape, chunk.stride);
        const src = chunk.data;
        const out = new this.#ctr(src.length);
        out[0] = src[0];
        for (let i = 1; i < src.length; i++) {
            // @ts-expect-error - mix of bigint and number always safe to subtract
            out[i] = src[i] - src[i - 1];
        }
        return { data: out, shape: chunk.shape, stride: chunk.stride };
    }
    decode(chunk) {
        assertCOrFContiguous(chunk.shape, chunk.stride);
        const src = chunk.data;
        const out = new this.#ctr(src.length);
        out[0] = src[0];
        for (let i = 1; i < src.length; i++) {
            // @ts-expect-error - mix of bigint and number always safe to add
            out[i] = out[i - 1] + src[i];
        }
        return { data: out, shape: chunk.shape, stride: chunk.stride };
    }
}
//# sourceMappingURL=delta.js.map