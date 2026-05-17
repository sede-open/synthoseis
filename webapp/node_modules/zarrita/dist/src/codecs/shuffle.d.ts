import type { DataType } from "../metadata.js";
/**
 * Shuffle filter codec (numcodecs compat).
 *
 * Reorders bytes so that corresponding bytes of each element are grouped
 * together, improving compression of typed data.
 */
export declare class ShuffleCodec<D extends DataType> {
    #private;
    kind: string;
    constructor(configuration: {
        elementsize?: number;
    }, meta?: {
        dataType: D;
    });
    static fromConfig<D extends DataType>(configuration: {
        elementsize?: number;
    }, meta?: {
        dataType: D;
    }): ShuffleCodec<D>;
    encode(data: Uint8Array): Uint8Array;
    decode(data: Uint8Array): Uint8Array;
}
//# sourceMappingURL=shuffle.d.ts.map