import { assertFactoryResult, createProxy } from "./define.js";
export function defineArrayExtension(factory) {
    return (array, opts) => {
        // @ts-expect-error - factory's opts parameter is wider than `never` at runtime.
        let result = factory(array, opts);
        if (result instanceof Promise) {
            return result.then((overrides) => {
                assertFactoryResult(overrides);
                return createProxy(array, overrides);
            });
        }
        assertFactoryResult(result);
        return createProxy(array, result);
    };
}
//# sourceMappingURL=define-array.js.map