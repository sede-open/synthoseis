import type { Mutable } from "@zarrita/storage";
import { Array, Group, Location } from "./hierarchy.js";
import type { Attributes, CodecMetadata, DataType, Scalar } from "./metadata.js";
interface CreateGroupOptions {
    attributes?: Record<string, unknown>;
}
interface CreateArrayOptions<Dtype extends DataType> {
    shape: number[];
    chunkShape: number[];
    dtype: Dtype;
    codecs?: CodecMetadata[];
    fillValue?: Scalar<Dtype>;
    chunkSeparator?: "." | "/";
    dimensionNames?: string[];
    attributes?: Attributes;
}
export declare function create<Store extends Mutable, _Dtype extends DataType = DataType>(location: Location<Store> | Store): Promise<Group<Store>>;
export declare function create<Store extends Mutable, _Dtype extends DataType = DataType>(location: Location<Store> | Store, options: CreateGroupOptions): Promise<Group<Store>>;
export declare function create<Store extends Mutable, Dtype extends DataType>(location: Location<Store> | Store, options: CreateArrayOptions<Dtype>): Promise<Array<Dtype, Store>>;
export {};
//# sourceMappingURL=create.d.ts.map