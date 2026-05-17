export declare class Crc32cCodec {
    readonly kind = "bytes_to_bytes";
    static fromConfig(): Crc32cCodec;
    encode: (_: never) => never;
    decode(arr: Uint8Array): Uint8Array;
    computeEncodedSize(decodedSize: number): number;
}
//# sourceMappingURL=crc32c.d.ts.map