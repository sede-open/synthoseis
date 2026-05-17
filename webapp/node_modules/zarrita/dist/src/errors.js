/**
 * @module
 *
 * Structured error types thrown by zarrita. Use {@link isZarritaError} to
 * detect and narrow errors at runtime — error classes are intentionally not
 * exported from the package root, so `instanceof` checks on the concrete
 * classes are not part of the public contract.
 *
 * @example
 * ```ts
 * import * as zarr from "zarrita";
 *
 * try {
 *   const arr = await zarr.open(store, { kind: "array" });
 *   await zarr.get(arr, [null, null]);
 * } catch (err) {
 *   if (zarr.isZarritaError(err, "NotFoundError")) {
 *     // err.path / err.found are available
 *   } else if (zarr.isZarritaError(err, "UnknownCodecError")) {
 *     // err.codec is the unregistered codec name
 *   } else if (zarr.isZarritaError(err)) {
 *     // any zarrita-thrown error
 *   } else {
 *     throw err;
 *   }
 * }
 * ```
 */
class ZarritaError extends Error {
}
/**
 * The store returned nothing for a required key, or `open({ kind })` was
 * asked for an array/group and the path holds the other kind.
 */
export class NotFoundError extends ZarritaError {
    _tag = "NotFoundError";
    name = "NotFoundError";
    path;
    found;
    constructor(context, options = {}) {
        super(`Not found: ${context}`, { cause: options.cause });
        this.path = options.path;
        this.found = options.found;
    }
}
/**
 * Metadata exists but isn't something zarrita can read: malformed JSON,
 * unknown dtype or chunk-key encoding, or a codec rejected its config at
 * load time.
 */
export class InvalidMetadataError extends ZarritaError {
    _tag = "InvalidMetadataError";
    name = "InvalidMetadataError";
    path;
    constructor(message, options = {}) {
        super(message, { cause: options.cause });
        this.path = options.path;
    }
}
/**
 * The metadata references a codec that isn't in the registry. Register the
 * codec via `zarr.registry` and retry to recover.
 */
export class UnknownCodecError extends ZarritaError {
    _tag = "UnknownCodecError";
    name = "UnknownCodecError";
    codec;
    constructor(codec) {
        super(`Unknown codec: ${codec}`);
        this.codec = codec;
    }
}
/**
 * A codec threw while encoding or decoding chunk bytes. The originating
 * codec error is preserved on `cause`; per-chunk failures can be retried
 * or skipped without abandoning the surrounding read.
 */
export class CodecPipelineError extends ZarritaError {
    _tag = "CodecPipelineError";
    name = "CodecPipelineError";
    direction;
    codec;
    chunkPath;
    constructor(options) {
        const parts = [
            `Failed to ${options.direction} chunk`,
            options.codec && `via codec "${options.codec}"`,
            options.chunkPath && `at ${options.chunkPath}`,
        ].filter(Boolean);
        super(parts.join(" "), { cause: options.cause });
        this.direction = options.direction;
        this.codec = options.codec;
        this.chunkPath = options.chunkPath;
    }
}
/**
 * The selection passed to `get` or `set` is invalid: wrong rank,
 * out-of-bounds, zero step, unknown dimension name, or scalar-shape
 * mismatch.
 */
export class InvalidSelectionError extends ZarritaError {
    _tag = "InvalidSelectionError";
    name = "InvalidSelectionError";
}
/**
 * The requested operation hits a capability limit: write to a sharded
 * array, encode with a codec that only implements decode, or rely on a
 * runtime feature the host doesn't provide.
 */
export class UnsupportedError extends ZarritaError {
    _tag = "UnsupportedError";
    name = "UnsupportedError";
    feature;
    constructor(feature) {
        super(`Unsupported: ${feature}`);
        this.feature = feature;
    }
}
/**
 * Runtime guard for zarrita-thrown errors. Without tags, narrows to any
 * zarrita error; with one or more tags, narrows to the matching subset
 * (including their typed fields).
 *
 * @example
 * ```ts
 * try { ... } catch (err) {
 *   if (zarr.isZarritaError(err, "NotFoundError")) {
 *     console.log(err.path);
 *   }
 * }
 * ```
 */
export function isZarritaError(err, ...tags) {
    if (!(err instanceof ZarritaError))
        return false;
    return tags.length === 0 || tags.includes(err._tag);
}
//# sourceMappingURL=errors.js.map