import type { AbsolutePath, AsyncReadable, Readable } from "@zarrita/storage";
/** The format of consolidated metadata to use. */
export type ConsolidatedFormat = "v2" | "v3";
/** Options for {@linkcode withConsolidatedMetadata} and {@linkcode withMaybeConsolidatedMetadata}. */
export interface ConsolidatedMetadataOptions {
    /**
     * The format(s) of consolidated metadata to try.
     *
     * - `"v2"` — Zarr v2 consolidated metadata (`.zmetadata`).
     * - `"v3"` — Zarr v3 consolidated metadata (`zarr.json`). This targets the
     *   experimental consolidated metadata implemented in zarr-python, which is
     *   not yet part of the official Zarr v3 specification.
     * - An array of formats to try in order (e.g., `["v3", "v2"]`).
     *
     * When omitted, the format is auto-detected using the store's version history.
     */
    readonly format?: ConsolidatedFormat | ConsolidatedFormat[];
    /**
     * Key to read consolidated metadata from. Only applies to `"v2"` format.
     *
     * @default {".zmetadata"}
     */
    readonly metadataKey?: string;
}
/** A store augmented with a `contents()` method from consolidated metadata. */
export type Listable<Store extends Readable> = Store & {
    contents(): {
        path: AbsolutePath;
        kind: "array" | "group";
    }[];
};
/**
 * Wraps a store with consolidated metadata, enabling efficient listing and
 * metadata access without extra network requests.
 *
 * Supports Zarr v2 (`.zmetadata`) and v3 (`zarr.json` with
 * `consolidated_metadata`). Throws if no consolidated metadata is found.
 *
 * @example
 * ```ts
 * // Direct
 * let store = await zarr.withConsolidatedMetadata(new zarr.FetchStore("https://..."));
 *
 * // With options
 * let store = await zarr.withConsolidatedMetadata(rawStore, { format: "v3" });
 *
 * // In a pipeline
 * let store = await zarr.extendStore(
 *   new zarr.FetchStore("https://..."),
 *   (s) => zarr.withConsolidatedMetadata(s, { format: "v3" }),
 * );
 *
 * store.contents(); // [{ path: "/", kind: "group" }, ...]
 * ```
 */
export declare const withConsolidatedMetadata: <S extends AsyncReadable>(store: S, opts?: ConsolidatedMetadataOptions | undefined) => Promise<S & {
    contents: () => {
        path: AbsolutePath;
        kind: "array" | "group";
    }[];
} extends infer T ? T extends S & {
    contents: () => {
        path: AbsolutePath;
        kind: "array" | "group";
    }[];
} ? T extends AsyncReadable ? (T extends {
    getRange: (...args: never[]) => unknown;
} ? Required<AsyncReadable> : AsyncReadable) & (Omit<T, keyof AsyncReadable> extends infer T_1 ? { [K in keyof T_1]: T_1[K]; } : never) : T : never : never>;
/**
 * Like {@linkcode withConsolidatedMetadata}, but falls back to the original store if
 * no consolidated metadata is found (instead of throwing).
 */
export declare function withMaybeConsolidatedMetadata<Store extends AsyncReadable>(store: Store, opts?: ConsolidatedMetadataOptions): Promise<Listable<Store> | Store>;
//# sourceMappingURL=consolidation.d.ts.map