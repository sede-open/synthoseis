import type { Chunk, CodecMetadata, DataType } from "../metadata.js";
export declare class BytesCodec<D extends Exclude<DataType, "v2:object" | "string">> {
    #private;
    kind: string;
    constructor(configuration: {
        endian?: "little" | "big";
    } | undefined, meta: {
        dataType: D;
        shape: number[];
        codecs: CodecMetadata[];
    });
    static fromConfig<D extends Exclude<DataType, "v2:object" | "string">>(configuration: {
        endian: "little" | "big";
    }, meta: {
        dataType: D;
        shape: number[];
        codecs: CodecMetadata[];
    }): BytesCodec<D>;
    encode(arr: Chunk<D>): Uint8Array;
    computeEncodedSize(decodedSize: number): number;
    decode(bytes: Uint8Array): Chunk<D>;
}
//# sourceMappingURL=bytes.d.ts.map