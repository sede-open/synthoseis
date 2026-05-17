import { UnsupportedError } from "../errors.js";
export function unimplementedEncode(codecName) {
    return () => {
        throw new UnsupportedError(`${codecName} encode`);
    };
}
//# sourceMappingURL=_shared.js.map