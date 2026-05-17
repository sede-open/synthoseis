interface GzipCodecConfig {
    level: number;
}
export declare class GzipCodec {
    kind: string;
    static fromConfig(_: GzipCodecConfig): GzipCodec;
    encode: (_: never) => never;
    decode(bytes: Uint8Array): Promise<Uint8Array>;
}
export {};
//# sourceMappingURL=gzip.d.ts.map