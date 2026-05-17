import type { Chunk } from "../metadata.js";
import { type JsonScalar, type NumericDataType } from "./json-scalar.js";
type RoundingMode = "nearest-even" | "towards-zero" | "towards-positive" | "towards-negative" | "nearest-away";
type OutOfRangeMode = "clamp" | "wrap";
interface ScalarMapConfig {
    encode?: [JsonScalar, JsonScalar][];
    decode?: [JsonScalar, JsonScalar][];
}
export interface CastValueConfig {
    data_type: NumericDataType;
    rounding?: RoundingMode;
    out_of_range?: OutOfRangeMode;
    scalar_map?: ScalarMapConfig;
}
type ScalarMapEntry = {
    src: number | bigint;
    tgt: number | bigint;
};
/**
 * Array-to-array codec that casts element values between two numeric types.
 *
 * Two data types are in play:
 *   - `ArrayDtype`:   the array's declared data type (`meta.dataType`).
 *                     This is what the user reads/writes.
 *   - `EncodedDtype`: the on-disk type (`config.data_type`).
 *                     This is what the bytes codec sees.
 *
 * Encode: ArrayDtype -> EncodedDtype  (not implemented, throws)
 * Decode: EncodedDtype -> ArrayDtype
 */
export declare class CastValueCodec<ArrayDtype extends NumericDataType, EncodedDtype extends NumericDataType> {
    #private;
    kind: "array_to_array";
    constructor(arrayType: ArrayDtype, encodedType: EncodedDtype, rounding: RoundingMode, outOfRange: OutOfRangeMode | undefined, decodeMapEntries: ScalarMapEntry[], encodeMapEntries: ScalarMapEntry[]);
    /** Return updated metadata reflecting the type and fill value after encoding. */
    getEncodedMeta(meta: {
        dataType: string;
        shape: number[];
        codecs: unknown[];
        fillValue: unknown;
    }): {
        dataType: string;
        shape: number[];
        codecs: unknown[];
        fillValue: unknown;
    };
    static fromConfig<A extends NumericDataType>(config: CastValueConfig, meta: {
        dataType: A;
    }): CastValueCodec<A, NumericDataType>;
    encode: (_: never) => never;
    decode(chunk: Chunk<EncodedDtype>): Chunk<ArrayDtype>;
}
export {};
//# sourceMappingURL=cast_value.d.ts.map