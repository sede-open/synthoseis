import type { Reader, ZipInfo } from "unzipit";
import type { AbsolutePath, AsyncReadable, RangeQuery } from "./types.js";
export declare class BlobReader implements Reader {
    blob: Blob;
    constructor(blob: Blob);
    getLength(): Promise<number>;
    read(offset: number, length: number): Promise<Uint8Array<ArrayBuffer>>;
}
/** Options for {@linkcode ZipFileStore}. */
interface ZipFileStoreOptions {
    /**
     * Optional function to transform entries after unzipping.
     *
     * Useful for modifying or restructuring the paths of extracted zip entries.
     */
    transformEntries?: (entries: ZipInfo["entries"]) => ZipInfo["entries"];
}
export declare class HTTPRangeReader implements Reader {
    #private;
    url: string | URL;
    private length?;
    constructor(url: string | URL, opts?: {
        overrides?: RequestInit;
    });
    getLength(): Promise<number>;
    read(offset: number, size: number): Promise<Uint8Array<ArrayBuffer>>;
}
/** @experimental */
declare class ZipFileStore<R extends Reader = Reader> implements AsyncReadable {
    private info;
    private reader;
    constructor(reader: R, opts?: ZipFileStoreOptions);
    private getEntryDataOffset;
    get(key: AbsolutePath): Promise<Uint8Array | undefined>;
    getRange(key: AbsolutePath, range: RangeQuery): Promise<Uint8Array | undefined>;
    has(key: AbsolutePath): Promise<boolean>;
    static fromUrl(href: string | URL, opts?: {
        overrides?: RequestInit;
    } & ZipFileStoreOptions): ZipFileStore<HTTPRangeReader>;
    static fromBlob(blob: Blob, opts?: ZipFileStoreOptions): ZipFileStore<BlobReader>;
}
export default ZipFileStore;
//# sourceMappingURL=zip.d.ts.map