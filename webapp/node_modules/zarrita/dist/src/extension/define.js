/**
 * Build a Proxy that serves `overrides` for listed keys and delegates the
 * rest to `target`. Getters and methods run with `this = target` so that
 * class instances with private fields (e.g. `FetchStore.#fetch`,
 * `Array.#metadata`) continue to work when accessed through the wrapper.
 */
export function createProxy(target, overrides) {
    let boundCache = new Map();
    return new Proxy(target, {
        get(t, prop) {
            if (prop in overrides)
                return overrides[prop];
            let cached = boundCache.get(prop);
            if (cached !== undefined)
                return cached;
            let value = Reflect.get(t, prop, t);
            if (typeof value === "function") {
                let bound = value.bind(t);
                boundCache.set(prop, bound);
                return bound;
            }
            return value;
        },
        has(t, prop) {
            return prop in overrides || Reflect.has(t, prop);
        },
        ownKeys(t) {
            let keys = new Set([...Reflect.ownKeys(t), ...Object.keys(overrides)]);
            return [...keys];
        },
        getOwnPropertyDescriptor(t, prop) {
            if (prop in overrides) {
                return {
                    configurable: true,
                    enumerable: true,
                    value: overrides[prop],
                };
            }
            return Reflect.getOwnPropertyDescriptor(t, prop);
        },
    });
}
export function assertFactoryResult(value) {
    if (value == null || typeof value !== "object") {
        throw new Error("Extension factory must return an object of overrides");
    }
}
/**
 * Merge the inner store's `arrayExtensions` (if any) with the list returned
 * by the factory. Inner-first, outer-last: if the inner store contributed
 * `[A]` and the factory adds `[B]`, the merged list is `[A, B]` — so when
 * `zarr.open` applies them via `extendArray(arr, A, B)`, the outer (B)
 * wraps the inner (A), mirroring how store extensions themselves compose.
 */
function mergeArrayExtensions(inner, overrides) {
    let innerExts = inner.arrayExtensions;
    let freshExts = overrides.arrayExtensions;
    if (!innerExts?.length)
        return overrides;
    if (!freshExts?.length)
        return { ...overrides, arrayExtensions: innerExts };
    return { ...overrides, arrayExtensions: [...innerExts, ...freshExts] };
}
export function defineStoreExtension(factory) {
    return (store, opts) => {
        // @ts-expect-error - factory's opts parameter is wider than `never` at runtime.
        let result = factory(store, opts);
        if (result instanceof Promise) {
            return result.then((overrides) => {
                assertFactoryResult(overrides);
                return createProxy(store, mergeArrayExtensions(store, overrides));
            });
        }
        assertFactoryResult(result);
        return createProxy(store, mergeArrayExtensions(store, result));
    };
}
//# sourceMappingURL=define.js.map