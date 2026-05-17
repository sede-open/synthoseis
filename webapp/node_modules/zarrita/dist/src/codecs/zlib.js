import { decompress } from "../util.js";
import { unimplementedEncode } from "./_shared.js";
export class ZlibCodec {
    kind = "bytes_to_bytes";
    static fromConfig(_) {
        return new ZlibCodec();
    }
    encode = unimplementedEncode("zlib");
    async decode(bytes) {
        const buffer = await decompress(bytes, { format: "deflate" });
        return new Uint8Array(buffer);
    }
}
//# sourceMappingURL=zlib.js.map