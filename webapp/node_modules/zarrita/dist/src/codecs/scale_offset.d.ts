import type { Chunk, Scalar } from "../metadata.js";
import { getCtr } from "../util.js";
import { type JsonScalar, type NumericDataType } from "./json-scalar.js";
export interface ScaleOffsetConfig {
    scale?: JsonScalar;
    offset?: JsonScalar;
}
export declare class ScaleOffsetCodec<D extends NumericDataType> {
    #private;
    kind: "array_to_array";
    constructor(scale: Scalar<D>, offset: Scalar<D>, ctr: ReturnType<typeof getCtr<D>>);
    static fromConfig<D extends NumericDataType>(config: ScaleOffsetConfig, meta: {
        dataType: D;
    }): ScaleOffsetCodec<D>;
    encode: (_: never) => never;
    decode(chunk: Chunk<D>): Chunk<D>;
}
//# sourceMappingURL=scale_offset.d.ts.map