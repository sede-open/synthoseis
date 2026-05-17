import type { Codec as _Codec } from "numcodecs";
import type { Chunk, CodecMetadata, DataType, Scalar } from "./metadata.js";
type ChunkMetadata<D extends DataType> = {
    dataType: D;
    shape: number[];
    codecs: CodecMetadata[];
    fillValue: Scalar<D> | null;
};
type CodecEntry = {
    fromConfig: (config: unknown, meta: ChunkMetadata<DataType>) => Codec;
    kind?: "array_to_array" | "array_to_bytes" | "bytes_to_bytes";
};
type Codec = _Codec & {
    kind: CodecEntry["kind"];
    getEncodedMeta?: (meta: ChunkMetadata<DataType>) => ChunkMetadata<DataType>;
    computeEncodedSize?: (decodedSize: number) => number;
};
export declare const registry: Map<string, () => Promise<CodecEntry>>;
export declare function createCodecPipeline<Dtype extends DataType>(chunkMetadata: ChunkMetadata<Dtype>): {
    encode(chunk: Chunk<Dtype>): Promise<Uint8Array>;
    decode(bytes: Uint8Array): Promise<Chunk<Dtype>>;
    computeEncodedSize(decodedSize: number): Promise<number>;
};
export {};
//# sourceMappingURL=codecs.d.ts.map