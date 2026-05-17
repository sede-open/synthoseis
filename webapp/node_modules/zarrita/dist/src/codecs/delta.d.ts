import type { BigintDataType, Chunk, NumberDataType } from "../metadata.js";
import { getCtr } from "../util.js";
type DeltaCompatibleType = NumberDataType | BigintDataType;
export declare class DeltaCodec<D extends DeltaCompatibleType> {
    #private;
    kind: "array_to_array";
    constructor(ctr: ReturnType<typeof getCtr<D>>);
    static fromConfig<D extends DeltaCompatibleType>(_config: unknown, meta: {
        dataType: D;
    }): DeltaCodec<D>;
    encode(chunk: Chunk<D>): Chunk<D>;
    decode(chunk: Chunk<D>): Chunk<D>;
}
export {};
//# sourceMappingURL=delta.d.ts.map