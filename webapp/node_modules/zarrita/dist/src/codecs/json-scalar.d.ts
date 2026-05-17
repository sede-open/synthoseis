import type { BigintDataType, NumberDataType, Scalar } from "../metadata.js";
export type NumericDataType = NumberDataType | BigintDataType;
export type SpecialFloat = "NaN" | "Infinity" | "-Infinity";
export type HexString = `0x${string}`;
export type JsonScalar = number | SpecialFloat | HexString;
export declare function isFloatType(dataType: string): boolean;
export declare function isBigintType(dataType: string): boolean;
/**
 * Parse a JSON-encoded scalar value into its native JS representation
 * for the given Zarr data type.
 *
 * - Integer types: value must be a JSON number that is an integer.
 * - Bigint types (int64/uint64): value must be a JSON number that is an integer, converted to BigInt.
 * - Float types: value may be a number, a special float string, or a hex-encoded float.
 */
export declare function parseJsonScalar<D extends NumericDataType>(dataType: D, value: JsonScalar): Scalar<D>;
//# sourceMappingURL=json-scalar.d.ts.map