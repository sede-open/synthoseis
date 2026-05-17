import { InvalidMetadataError, NotFoundError } from "../errors.js";
import { VERSION_COUNTER } from "../open.js";
import { jsonDecodeObject, jsonEncodeObject, rethrowUnless } from "../util.js";
import { defineStoreExtension } from "./define.js";
function isConsolidatedV2(meta) {
    return (typeof meta === "object" &&
        meta !== null &&
        "zarr_consolidated_format" in meta &&
        meta.zarr_consolidated_format === 1 &&
        "metadata" in meta &&
        typeof meta.metadata === "object" &&
        meta.metadata !== null);
}
function isConsolidatedV3(meta) {
    return (typeof meta === "object" &&
        meta !== null &&
        "zarr_format" in meta &&
        meta.zarr_format === 3 &&
        "node_type" in meta &&
        meta.node_type === "group" &&
        "consolidated_metadata" in meta &&
        typeof meta.consolidated_metadata === "object" &&
        meta.consolidated_metadata !== null &&
        "metadata" in meta.consolidated_metadata &&
        typeof meta.consolidated_metadata.metadata === "object" &&
        meta.consolidated_metadata.metadata !== null);
}
function isMetaKey(key) {
    return (key.endsWith(".zarray") ||
        key.endsWith(".zgroup") ||
        key.endsWith(".zattrs") ||
        key.endsWith("zarr.json"));
}
function isV3(meta) {
    return "zarr_format" in meta && meta.zarr_format === 3;
}
async function loadConsolidatedV2(store, metadataKey) {
    let key = metadataKey ?? ".zmetadata";
    let bytes = await store.get(`/${key}`);
    if (!bytes) {
        throw new NotFoundError("v2 consolidated metadata", {
            path: `/${key}`,
        });
    }
    let meta = jsonDecodeObject(bytes);
    if (!isConsolidatedV2(meta)) {
        throw new InvalidMetadataError("Invalid or unsupported v2 consolidated format", { path: `/${key}` });
    }
    let knownMeta = {};
    for (let [k, value] of Object.entries(meta.metadata)) {
        knownMeta[`/${k}`] = value;
    }
    return knownMeta;
}
async function loadConsolidatedV3(store) {
    let bytes = await store.get("/zarr.json");
    if (!bytes) {
        throw new NotFoundError("v3 consolidated metadata", {
            path: "/zarr.json",
        });
    }
    let rootMeta = jsonDecodeObject(bytes);
    if (!isConsolidatedV3(rootMeta)) {
        throw new InvalidMetadataError("Root zarr.json does not contain consolidated_metadata", { path: "/zarr.json" });
    }
    let knownMeta = {};
    knownMeta["/zarr.json"] = {
        zarr_format: 3,
        node_type: "group",
        attributes: rootMeta.attributes ?? {},
    };
    for (let [path, meta] of Object.entries(rootMeta.consolidated_metadata.metadata)) {
        let normalized = path.startsWith("/") ? path : `/${path}`;
        let key = `${normalized}/zarr.json`;
        knownMeta[key] = meta;
    }
    return knownMeta;
}
function resolveFormats(store, format) {
    if (format !== undefined) {
        return globalThis.Array.isArray(format) ? format : [format];
    }
    let versionMax = VERSION_COUNTER.versionMax(store);
    return versionMax === "v3" ? ["v3", "v2"] : ["v2", "v3"];
}
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
export const withConsolidatedMetadata = defineStoreExtension(async (store, opts = {}) => {
    let formats = resolveFormats(store, opts.format);
    let lastError;
    for (let format of formats) {
        try {
            let knownMeta = format === "v2"
                ? await loadConsolidatedV2(store, opts.metadataKey)
                : await loadConsolidatedV3(store);
            return {
                async get(key, options) {
                    if (knownMeta[key]) {
                        return jsonEncodeObject(knownMeta[key]);
                    }
                    let maybeBytes = await store.get(key, options);
                    if (isMetaKey(key) && maybeBytes) {
                        knownMeta[key] = jsonDecodeObject(maybeBytes);
                    }
                    return maybeBytes;
                },
                contents() {
                    let contents = [];
                    for (let [key, value] of Object.entries(knownMeta)) {
                        let parts = key.split("/");
                        let filename = parts.pop();
                        let path = (parts.join("/") || "/");
                        if (filename === ".zarray")
                            contents.push({ path, kind: "array" });
                        if (filename === ".zgroup")
                            contents.push({ path, kind: "group" });
                        if (isV3(value)) {
                            contents.push({ path, kind: value.node_type });
                        }
                    }
                    return contents;
                },
            };
        }
        catch (err) {
            rethrowUnless(err, NotFoundError, InvalidMetadataError);
            lastError = err;
        }
    }
    throw lastError;
});
/**
 * Like {@linkcode withConsolidatedMetadata}, but falls back to the original store if
 * no consolidated metadata is found (instead of throwing).
 */
export async function withMaybeConsolidatedMetadata(store, opts = {}) {
    return withConsolidatedMetadata(store, opts).catch((error) => {
        rethrowUnless(error, NotFoundError, InvalidMetadataError);
        return store;
    });
}
//# sourceMappingURL=consolidation.js.map