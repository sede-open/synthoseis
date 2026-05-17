interface ZlibCodecConfig {
    level: number;
}
export declare class ZlibCodec {
    kind: string;
    static fromConfig(_: ZlibCodecConfig): ZlibCodec;
    encode: (_: never) => never;
    decode(bytes: Uint8Array): Promise<Uint8Array>;
}
export {};
//# sourceMappingURL=zlib.d.ts.map