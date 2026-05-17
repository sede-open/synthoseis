import type { AbsolutePath, AsyncReadable, RangeQuery } from "./types.js";
/** Options for configuring a {@link FetchStore}. */
interface FetchStoreOptions {
    /**
     * A custom fetch handler to intercept requests.
     *
     * Receives a standard {@link https://github.com/nicolo-ribaudo/tc55-proposal-functions-api | WinterTC fetch handler},
     * similar to Cloudflare Workers, Deno.serve, and Bun.serve.
     *
     * Receives a {@link Request} object and must return a {@link Response}.
     * The response is handled by the store as follows:
     *
     * - **404** — treated as a missing key (`undefined`)
     * - **200 / 206** — treated as success, body is read as bytes
     * - **Any other status** — throws an error
     *
     * Use this to add authentication, presign URLs, transform requests,
     * or remap error codes for backends that don't follow the conventions
     * above.
     *
     * **Important:** Sharding and partial reads rely on `Range` headers.
     * When transforming a request (e.g., changing the URL), use
     * `new Request(newUrl, originalRequest)` to preserve headers, abort
     * signals, and other options passed via `store.get(key, init)`.
     *
     * @example Presign URLs
     * ```ts
     * const store = new FetchStore("https://my-bucket.s3.amazonaws.com/data.zarr", {
     *   async fetch(request) {
     *     const newUrl = await presign(request.url);
     *     // Preserves headers, abort signal, and other options from store.get(key, init)
     *     return fetch(new Request(newUrl, request));
     *   },
     * });
     * ```
     *
     * @example Handle S3 403 as missing key
     *
     * S3 returns 403 (not 404) for missing keys on private buckets,
     * which causes the store to throw. If you know that 403 means
     * "not found" in your setup, remap it:
     *
     * ```ts
     * const store = new FetchStore("https://my-bucket.s3.amazonaws.com/data.zarr", {
     *   async fetch(request) {
     *     const response = await fetch(request);
     *     if (response.status === 403) {
     *       return new Response(null, { status: 404 });
     *     }
     *     return response;
     *   },
     * });
     * ```
     */
    fetch?: (request: Request) => Promise<Response>;
    /**
     * Default {@link RequestInit} options applied to every request.
     *
     * @deprecated Prefer implementing a custom {@link FetchStoreOptions.fetch}
     * to intercept and modify requests.
     */
    overrides?: RequestInit;
    /** Whether to use suffix-length range requests (e.g., `Range: bytes=-N`). */
    useSuffixRequest?: boolean;
}
/**
 * Readonly store backed by the
 * [Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API).
 *
 * Works anywhere `fetch` is available: browsers, Node.js 18+, Deno, and Bun.
 *
 * ## Response handling
 *
 * The store interprets HTTP responses as follows:
 *
 * | Status | Meaning |
 * | ------ | ------- |
 * | **404** | Missing key — returns `undefined` |
 * | **200 / 206** | Success — body is read as `Uint8Array` |
 * | **Any other** | Throws an error |
 *
 * If you need to remap status codes (e.g., S3 returns 403 for missing keys
 * on private buckets), provide a custom `fetch` implementation.
 *
 * @example Basic usage
 * ```ts
 * import { FetchStore } from "@zarrita/storage";
 *
 * const store = new FetchStore("http://localhost:8080/data.zarr");
 * ```
 *
 * @example Custom fetch with presigned URLs
 * ```ts
 * import { FetchStore } from "@zarrita/storage";
 *
 * const store = new FetchStore("https://my-bucket.s3.amazonaws.com/data.zarr", {
 *   async fetch(request) {
 *     const newUrl = await presign(request.url);
 *     // Preserves Range headers, abort signal, etc.
 *     return fetch(new Request(newUrl, request));
 *   },
 * });
 * ```
 */
declare class FetchStore implements AsyncReadable {
    #private;
    url: string | URL;
    constructor(url: string | URL, options?: FetchStoreOptions);
    get(key: AbsolutePath, options?: RequestInit): Promise<Uint8Array | undefined>;
    getRange(key: AbsolutePath, range: RangeQuery, options?: RequestInit): Promise<Uint8Array | undefined>;
}
export type { FetchStoreOptions };
export default FetchStore;
//# sourceMappingURL=fetch.d.ts.map