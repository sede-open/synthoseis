import { unimplementedEncode } from "./_shared.js";
export class Crc32cCodec {
    kind = "bytes_to_bytes";
    static fromConfig() {
        return new Crc32cCodec();
    }
    encode = unimplementedEncode("crc32c");
    decode(arr) {
        return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength - 4);
    }
    computeEncodedSize(decodedSize) {
        return decodedSize + 4;
    }
}
//# sourceMappingURL=crc32c.js.map