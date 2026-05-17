import type { FetchStoreOptions } from "./fetch.js";
import type { AbsolutePath, AsyncReadable, RangeQuery } from "./types.js";
import { resolveUri } from "./util.js";
/** A resolved reference entry — base64 strings are decoded upfront. */
type ResolvedEntry = Uint8Array<ArrayBuffer> | string | [url: string | null] | [url: string | null, offset: number, length: number];
interface ReferenceStoreOptions {
    target?: string | URL;
    /**
     * A custom fetch handler, same as {@link FetchStoreOptions.fetch}.
     * Covers both the spec load (via {@link ReferenceStore.fromUrl}) and
     * all chunk fetches.
     */
    fetch?: FetchStoreOptions["fetch"];
    /**
     * @deprecated Prefer providing a custom {@link ReferenceStoreOptions.fetch}.
     */
    overrides?: RequestInit;
}
/**
 * A store backed by a
 * [kerchunk reference spec](https://fsspec.github.io/kerchunk/spec.html),
 * enabling random access to data in monolithic files (HDF5, TIFF, etc.)
 * that have been mapped to Zarr.
 *
 * Uses {@link FetchStore} internally. The default fetch handler translates
 * cloud-storage URIs (`s3://`, `gs://`, `gcs://`) to HTTPS via
 * {@link ReferenceStore.resolveUri}. Inline entries (plain strings and
 * base64) are served directly without making any network requests.
 *
 * @example Basic usage
 * ```ts
 * const store = await ReferenceStore.fromUrl("https://example.com/refs.json");
 * const arr = await zarr.open(store);
 * ```
 *
 * @example Custom fetch with auth
 * ```ts
 * const store = await ReferenceStore.fromUrl("https://example.com/refs.json", {
 *   async fetch(request) {
 *     const url = ReferenceStore.resolveUri(request.url);
 *     const req = new Request(url, request);
 *     req.headers.set("Authorization", `Bearer ${await getToken()}`);
 *     return fetch(req);
 *   },
 * });
 * ```
 *
 * @example Opt out of default URI translation
 * ```ts
 * const store = await ReferenceStore.fromUrl("https://example.com/refs.json", {
 *   fetch: (request) => fetch(request),
 * });
 * ```
 *
 * @experimental
 */
declare class ReferenceStore implements AsyncReadable {
    #private;
    constructor(refs: Map<string, ResolvedEntry>, opts?: ReferenceStoreOptions);
    get(key: AbsolutePath, opts?: RequestInit): Promise<Uint8Array | undefined>;
    getRange(key: AbsolutePath, range: RangeQuery, opts?: RequestInit): Promise<Uint8Array | undefined>;
    /**
     * Translate `s3://` and `gc://` URIs to HTTPS URLs.
     * Useful when writing a custom `fetch` handler for reference stores
     * whose entries contain cloud-specific URIs.
     */
    static resolveUri: typeof resolveUri;
    static fromSpec(spec: Record<string, unknown>, opts?: ReferenceStoreOptions): ReferenceStore;
    static fromSpec(spec: Promise<Record<string, unknown>>, opts?: ReferenceStoreOptions): Promise<ReferenceStore>;
    static fromUrl(url: string | URL, opts?: ReferenceStoreOptions): Promise<ReferenceStore>;
}
export default ReferenceStore;
//# sourceMappingURL=ref.d.ts.map