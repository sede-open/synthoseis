import type { AbsolutePath, AsyncReadable } from "@zarrita/storage";
/**
 * Immutable report emitted once per microtask flush, per path, via the
 * optional `onFlush` callback. A fresh object is allocated per emission.
 */
export interface FlushReport {
    /** The store path this flush covered. */
    path: AbsolutePath;
    /** How many HTTP fetches the coalescer issued for this path. */
    groupCount: number;
    /** How many caller-level `getRange` requests were folded into those fetches. */
    requestCount: number;
    /** Total bytes requested across all groups (the sum of group lengths). */
    bytesFetched: number;
}
export interface RangeCoalescingOptions {
    /**
     * Byte gap threshold: two pending requests separated by less than this
     * many bytes are merged into a single fetch. Fetching across a small gap
     * is cheaper than an extra round trip.
     *
     * Default: 32768 (matches geotiff.js's `BlockedSource` heuristic and
     * Rust `object_store`'s `OBJECT_STORE_COALESCE_DEFAULT`).
     */
    coalesceSize?: number;
    /**
     * Optional observability hook. Called once per microtask flush, per path,
     * with a fresh `FlushReport`. Errors thrown from the callback are
     * swallowed via `console.warn`; async return values are ignored.
     *
     * Suffix-length range queries (`{ suffixLength }`) bypass batching and
     * do not emit `onFlush`.
     */
    onFlush?: (report: FlushReport) => void;
}
/**
 * Wraps a store with microtask-tick range batching: concurrent `getRange`
 * calls within a single microtask are grouped by path, coalesced across
 * small byte gaps, and issued as a single fetch per group. The coalesced
 * blob is sliced on return and each caller receives exactly the bytes they
 * asked for.
 *
 * `withRangeCoalescing` carries no cache state. Pair with `withByteCaching`
 * if you want cross-call caching.
 *
 * ```ts
 * import * as zarr from "zarrita";
 *
 * let store = zarr.withRangeCoalescing(
 *   new zarr.FetchStore("https://example.com/data.zarr"),
 *   { coalesceSize: 32768 },
 * );
 * ```
 */
export declare const withRangeCoalescing: <S extends AsyncReadable>(store: S, opts?: RangeCoalescingOptions | undefined) => S & {} extends infer T ? T extends S & {} ? T extends AsyncReadable ? (T extends {
    getRange: (...args: never[]) => unknown;
} ? Required<AsyncReadable> : AsyncReadable) & (Omit<T, keyof AsyncReadable> extends infer T_1 ? { [K in keyof T_1]: T_1[K]; } : never) : T : never : never;
//# sourceMappingURL=range-coalescing.d.ts.map