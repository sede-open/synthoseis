import { decompress } from "../util.js";
import { unimplementedEncode } from "./_shared.js";
export class GzipCodec {
    kind = "bytes_to_bytes";
    static fromConfig(_) {
        return new GzipCodec();
    }
    encode = unimplementedEncode("gzip");
    async decode(bytes) {
        const buffer = await decompress(bytes, { format: "gzip" });
        return new Uint8Array(buffer);
    }
}
//# sourceMappingURL=gzip.js.map