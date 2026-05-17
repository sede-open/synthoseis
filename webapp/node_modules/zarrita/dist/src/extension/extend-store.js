import { applyExtensions } from "./extend.js";
export function extendStore(store, ...extensions) {
    return applyExtensions(store, extensions);
}
//# sourceMappingURL=extend-store.js.map