import type { Chunk, String } from "../metadata.js";
export declare class VLenUTF8 {
    #private;
    readonly kind = "array_to_bytes";
    constructor(shape: number[]);
    static fromConfig(_: unknown, meta: {
        shape: number[];
    }): VLenUTF8;
    encode: (_: never) => never;
    decode(bytes: Uint8Array): Chunk<String>;
}
//# sourceMappingURL=vlen-utf8.d.ts.map