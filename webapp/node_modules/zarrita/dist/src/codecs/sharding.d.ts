import type { GetOptions, Readable } from "@zarrita/storage";
import type { Location } from "../hierarchy.js";
import type { ShardingCodecMetadata } from "../util.js";
export declare function createShardedChunkGetter(location: Location<Readable>, shardShape: number[], encodeShardKey: (coord: number[]) => string, shardingConfig: ShardingCodecMetadata["configuration"]): (chunkCoord: number[], options?: GetOptions) => Promise<Uint8Array<ArrayBufferLike> | undefined>;
//# sourceMappingURL=sharding.d.ts.map