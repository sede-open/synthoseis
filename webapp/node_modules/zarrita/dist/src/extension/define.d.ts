import type { AsyncReadable } from "@zarrita/storage";
import type { ArrayExtension } from "./define-array.js";
/** Store keys stripped from Extensions to avoid poisoning Options. */
type StoreKeys = "get" | "getRange";
/** Strip store keys from the extension type so Options aren't poisoned. */
type Extensions<T> = {
    [K in Extract<Exclude<keyof T, StoreKeys>, string>]: T[K];
} & {};
type Prettify<T> = {
    [K in keyof T]: T[K];
} & {};
/**
 * Collapse store-typed keys onto `AsyncReadable`, hide them from the combined
 * type, and show all extensions (from S and Ext) as a single merged object.
 */
type CollapseStore<T> = T extends AsyncReadable ? (T extends {
    getRange: (...args: never[]) => unknown;
} ? Required<AsyncReadable> : AsyncReadable) & Prettify<Omit<T, keyof AsyncReadable>> : T;
type WrapperResult<R, S> = R extends Promise<infer Inner> ? Promise<CollapseStore<S & Extensions<Inner>>> : CollapseStore<S & Extensions<R>>;
/**
 * Build a Proxy that serves `overrides` for listed keys and delegates the
 * rest to `target`. Getters and methods run with `this = target` so that
 * class instances with private fields (e.g. `FetchStore.#fetch`,
 * `Array.#metadata`) continue to work when accessed through the wrapper.
 */
export declare function createProxy<T extends object>(target: T, overrides: Record<string | symbol, unknown>): T;
type FactoryResult = Partial<AsyncReadable> & {
    /**
     * Array extensions carried on the store, auto-applied by `zarr.open` to
     * every `zarr.Array` it returns. When stores are composed via
     * `extendStore`, each layer's `arrayExtensions` are merged inner-first
     * so that later layers wrap earlier ones — symmetric with how store
     * extensions themselves compose.
     */
    arrayExtensions?: ReadonlyArray<ArrayExtension>;
} & Record<string, unknown>;
export declare function assertFactoryResult(value: unknown): asserts value is Record<string | symbol, unknown>;
/**
 * Define a composable store extension.
 *
 * The factory function receives the inner store and options, and returns an
 * object of overrides and extensions. Methods not returned are automatically
 * delegated to the inner store via Proxy.
 *
 * Supports both sync and async factories — if the factory returns a Promise,
 * the extension returns a Promise too.
 *
 * ```ts
 * import * as zarr from "zarrita";
 *
 * const withCaching = zarr.defineStoreExtension(
 *   (store, opts: { maxSize: number }) => {
 *     return {
 *       async get(key, options) { ... },
 *       clear() { ... },
 *     };
 *   },
 * );
 * ```
 */
export declare function defineStoreExtension<R extends FactoryResult | Promise<FactoryResult>, Opts = void>(factory: (store: AsyncReadable, opts: Opts) => R): <S extends AsyncReadable>(store: S, opts?: Opts) => WrapperResult<R, S>;
export {};
//# sourceMappingURL=define.d.ts.map