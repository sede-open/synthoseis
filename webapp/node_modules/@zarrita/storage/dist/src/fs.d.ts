import type { AbsolutePath, AsyncMutable, GetOptions, RangeQuery } from "./types.js";
declare class FileSystemStore implements AsyncMutable {
    root: string;
    constructor(root: string);
    get(key: AbsolutePath, opts?: GetOptions): Promise<Uint8Array | undefined>;
    getRange(key: AbsolutePath, range: RangeQuery, opts?: GetOptions): Promise<Uint8Array | undefined>;
    has(key: AbsolutePath): Promise<boolean>;
    set(key: AbsolutePath, value: Uint8Array): Promise<void>;
    delete(key: AbsolutePath): Promise<boolean>;
}
export default FileSystemStore;
//# sourceMappingURL=fs.d.ts.map