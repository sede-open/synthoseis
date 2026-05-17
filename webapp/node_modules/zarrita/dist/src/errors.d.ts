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
declare abstract class ZarritaError<T extends string> extends Error {
    abstract readonly _tag: T;
}
/**
 * The store returned nothing for a required key, or `open({ kind })` was
 * asked for an array/group and the path holds the other kind.
 */
export declare class NotFoundError extends ZarritaError<"NotFoundError"> {
    readonly _tag: "NotFoundError";
    readonly name = "NotFoundError";
    readonly path: string | undefined;
    readonly found: "array" | "group" | undefined;
    constructor(context: string, options?: {
        path?: string;
        found?: "array" | "group";
        cause?: unknown;
    });
}
/**
 * Metadata exists but isn't something zarrita can read: malformed JSON,
 * unknown dtype or chunk-key encoding, or a codec rejected its config at
 * load time.
 */
export declare class InvalidMetadataError extends ZarritaError<"InvalidMetadataError"> {
    readonly _tag: "InvalidMetadataError";
    readonly name = "InvalidMetadataError";
    readonly path: string | undefined;
    constructor(message: string, options?: {
        path?: string;
        cause?: unknown;
    });
}
/**
 * The metadata references a codec that isn't in the registry. Register the
 * codec via `zarr.registry` and retry to recover.
 */
export declare class UnknownCodecError extends ZarritaError<"UnknownCodecError"> {
    readonly _tag: "UnknownCodecError";
    readonly name = "UnknownCodecError";
    readonly codec: string;
    constructor(codec: string);
}
/**
 * A codec threw while encoding or decoding chunk bytes. The originating
 * codec error is preserved on `cause`; per-chunk failures can be retried
 * or skipped without abandoning the surrounding read.
 */
export declare class CodecPipelineError extends ZarritaError<"CodecPipelineError"> {
    readonly _tag: "CodecPipelineError";
    readonly name = "CodecPipelineError";
    readonly direction: "encode" | "decode";
    readonly codec: string | undefined;
    readonly chunkPath: string | undefined;
    constructor(options: {
        direction: "encode" | "decode";
        codec?: string;
        chunkPath?: string;
        cause: unknown;
    });
}
/**
 * The selection passed to `get` or `set` is invalid: wrong rank,
 * out-of-bounds, zero step, unknown dimension name, or scalar-shape
 * mismatch.
 */
export declare class InvalidSelectionError extends ZarritaError<"InvalidSelectionError"> {
    readonly _tag: "InvalidSelectionError";
    readonly name = "InvalidSelectionError";
}
/**
 * The requested operation hits a capability limit: write to a sharded
 * array, encode with a codec that only implements decode, or rely on a
 * runtime feature the host doesn't provide.
 */
export declare class UnsupportedError extends ZarritaError<"UnsupportedError"> {
    readonly _tag: "UnsupportedError";
    readonly name = "UnsupportedError";
    readonly feature: string;
    constructor(feature: string);
}
type AnyZarritaError = NotFoundError | InvalidMetadataError | UnknownCodecError | CodecPipelineError | InvalidSelectionError | UnsupportedError;
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
export declare function isZarritaError<T extends AnyZarritaError["_tag"]>(err: unknown, ...tags: T[]): err is Extract<AnyZarritaError, {
    _tag: T;
}>;
export {};
//# sourceMappingURL=errors.d.ts.map