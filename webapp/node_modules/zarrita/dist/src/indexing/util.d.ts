import type { Readable } from "@zarrita/storage";
import type { Array as ZarrArray } from "../hierarchy.js";
import type { DataType } from "../metadata.js";
import type { ChunkQueue, Indices, Slice } from "./types.js";
/** Similar to python's `range` function. Supports positive ranges only. */
export declare function range(start: number, stop?: number, step?: number): Iterable<number>;
/**
 * python-like itertools.product generator
 * https://gist.github.com/cybercase/db7dde901d7070c98c48
 */
export declare function product<T extends Array<Iterable<unknown>>>(...iterables: T): IterableIterator<{
    [K in keyof T]: T[K] extends Iterable<infer U> ? U : never;
}>;
export declare function sliceIndices({ start, stop, step }: Slice, length: number): Indices;
/** @category Utilty */
export declare function slice(stop: bigint | number | null): Slice;
export declare function slice(start: bigint | number | null, stop?: bigint | number | null, step?: bigint | number | null): Slice;
/** @category Utility */
export declare function select<D extends DataType, Store extends Readable>(arr: ZarrArray<D, Store>, selection: Record<string, Slice | number | null>): (Slice | number | null)[];
/** Built-in "queue" for awaiting promises. */
export declare function createQueue(): ChunkQueue;
//# sourceMappingURL=util.d.ts.map