/**
 * Walk a list of extensions, calling each synchronously until one returns a
 * `Promise`. From that point on, chain the remaining extensions with `.then`
 * so the caller only pays the cost of a Promise if any step actually needs
 * one. This is the shared runtime used by `extendStore` and `extendArray`.
 */
export function applyExtensions(value, extensions) {
    let result = value;
    for (let ext of extensions) {
        if (result instanceof Promise) {
            result = result.then((v) => ext(v));
        }
        else {
            result = ext(result);
        }
    }
    return result;
}
//# sourceMappingURL=extend.js.map