import { mergeInit } from "./util.js";
function resolve(root, path) {
    const base = typeof root === "string" ? new URL(root) : root;
    if (!base.pathname.endsWith("/")) {
        // ensure trailing slash so that base is resolved as _directory_
        base.pathname += "/";
    }
    const resolved = new URL(path.slice(1), base);
    // copy search params to new URL
    resolved.search = base.search;
    return resolved;
}
async function handleResponse(response) {
    if (response.status === 404) {
        return undefined;
    }
    if (response.status === 200 || response.status === 206) {
        return new Uint8Array(await response.arrayBuffer());
    }
    throw new Error(`Unexpected response status ${response.status} ${response.statusText}`);
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
class FetchStore {
    url;
    #fetch;
    #overrides;
    #useSuffixRequest;
    constructor(url, options = {}) {
        this.url = url;
        this.#fetch = options.fetch ?? ((request) => fetch(request));
        this.#overrides = options.overrides ?? {};
        this.#useSuffixRequest = options.useSuffixRequest ?? false;
    }
    #buildRequest(url, init) {
        return new Request(url, mergeInit(this.#overrides, init));
    }
    async get(key, options = {}) {
        let href = resolve(this.url, key).href;
        let request = this.#buildRequest(href, options);
        let response = await this.#fetch(request);
        return handleResponse(response);
    }
    async getRange(key, range, options = {}) {
        let url = resolve(this.url, key);
        let response;
        if ("suffixLength" in range) {
            response = await this.#fetchSuffix(url, range.suffixLength, options);
        }
        else {
            let rangeInit = {
                ...options,
                headers: {
                    ...options.headers,
                    Range: `bytes=${range.offset}-${range.offset + range.length - 1}`,
                },
            };
            let request = this.#buildRequest(url, rangeInit);
            response = await this.#fetch(request);
        }
        return handleResponse(response);
    }
    async #fetchSuffix(url, suffixLength, options) {
        if (this.#useSuffixRequest) {
            let init = {
                ...options,
                headers: { ...options.headers, Range: `bytes=-${suffixLength}` },
            };
            return this.#fetch(this.#buildRequest(url, init));
        }
        let headRequest = this.#buildRequest(url, {
            ...options,
            method: "HEAD",
        });
        let response = await this.#fetch(headRequest);
        if (!response.ok) {
            return response;
        }
        let contentLength = response.headers.get("Content-Length");
        let length = Number(contentLength);
        let offset = length - suffixLength;
        let rangeInit = {
            ...options,
            headers: {
                ...options.headers,
                Range: `bytes=${offset}-${length - 1}`,
            },
        };
        return this.#fetch(this.#buildRequest(url, rangeInit));
    }
}
export default FetchStore;
//# sourceMappingURL=fetch.js.map