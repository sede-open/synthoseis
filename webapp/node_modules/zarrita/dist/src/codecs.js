import { BitroundCodec } from "./codecs/bitround.js";
import { BytesCodec } from "./codecs/bytes.js";
import { CastValueCodec } from "./codecs/cast_value.js";
import { Crc32cCodec } from "./codecs/crc32c.js";
import { DeltaCodec } from "./codecs/delta.js";
import { GzipCodec } from "./codecs/gzip.js";
import { JsonCodec } from "./codecs/json2.js";
import { ScaleOffsetCodec } from "./codecs/scale_offset.js";
import { ShuffleCodec } from "./codecs/shuffle.js";
import { TransposeCodec } from "./codecs/transpose.js";
import { VLenUTF8 } from "./codecs/vlen-utf8.js";
import { ZlibCodec } from "./codecs/zlib.js";
import { CodecPipelineError, InvalidMetadataError, UnknownCodecError, } from "./errors.js";
function createDefaultRegistry() {
    let blosc = () => import("numcodecs/blosc").then((m) => m.default);
    let lz4 = () => import("numcodecs/lz4").then((m) => m.default);
    let zstd = () => import("numcodecs/zstd").then((m) => m.default);
    let gzip = () => GzipCodec;
    let zlib = () => ZlibCodec;
    return (new Map()
        // v3 codecs
        .set("blosc", blosc)
        .set("lz4", lz4)
        .set("zstd", zstd)
        .set("gzip", gzip)
        .set("zlib", zlib)
        .set("transpose", () => TransposeCodec)
        .set("bytes", () => BytesCodec)
        .set("crc32c", () => Crc32cCodec)
        .set("vlen-utf8", () => VLenUTF8)
        .set("json2", () => JsonCodec)
        .set("bitround", () => BitroundCodec)
        .set("cast_value", () => CastValueCodec)
        .set("scale_offset", () => ScaleOffsetCodec)
        // numcodecs (v2 compat)
        .set("numcodecs.blosc", blosc)
        .set("numcodecs.lz4", lz4)
        .set("numcodecs.zstd", zstd)
        .set("numcodecs.gzip", gzip)
        .set("numcodecs.zlib", zlib)
        .set("numcodecs.vlen-utf8", () => VLenUTF8)
        .set("numcodecs.shuffle", () => ShuffleCodec)
        .set("numcodecs.delta", () => DeltaCodec)
        .set("numcodecs.bitround", () => BitroundCodec)
        .set("numcodecs.json2", () => JsonCodec));
}
export const registry = createDefaultRegistry();
export function createCodecPipeline(chunkMetadata) {
    // Lazily load codecs on first use. The promise is shared by all methods.
    let codecsPromise;
    function getCodecs() {
        if (!codecsPromise)
            codecsPromise = loadCodecs(chunkMetadata);
        return codecsPromise;
    }
    async function runStep(direction, codec, fn) {
        try {
            return await fn();
        }
        catch (cause) {
            throw new CodecPipelineError({ direction, codec, cause });
        }
    }
    return {
        async encode(chunk) {
            let codecs = await getCodecs();
            for (const { name, codec } of codecs.arrayToArray) {
                chunk = await runStep("encode", name, () => codec.encode(chunk));
            }
            let bytes = await runStep("encode", codecs.arrayToBytes.name, () => codecs.arrayToBytes.codec.encode(chunk));
            for (const { name, codec } of codecs.bytesToBytes) {
                bytes = await runStep("encode", name, () => codec.encode(bytes));
            }
            return bytes;
        },
        async decode(bytes) {
            let codecs = await getCodecs();
            for (let i = codecs.bytesToBytes.length - 1; i >= 0; i--) {
                const { name, codec } = codecs.bytesToBytes[i];
                bytes = await runStep("decode", name, () => codec.decode(bytes));
            }
            let chunk = await runStep("decode", codecs.arrayToBytes.name, () => codecs.arrayToBytes.codec.decode(bytes));
            for (let i = codecs.arrayToArray.length - 1; i >= 0; i--) {
                const { name, codec } = codecs.arrayToArray[i];
                chunk = await runStep("decode", name, () => codec.decode(chunk));
            }
            return chunk;
        },
        async computeEncodedSize(decodedSize) {
            let codecs = await getCodecs();
            let size = applyEncodedSize(codecs.arrayToBytes.name, codecs.arrayToBytes.codec, decodedSize);
            for (const { name, codec } of codecs.bytesToBytes) {
                size = applyEncodedSize(name, codec, size);
            }
            return size;
        },
    };
}
function applyEncodedSize(name, codec, size) {
    if (!codec.computeEncodedSize) {
        throw new InvalidMetadataError(`Codec "${name}" cannot compute its encoded size; it is not a fixed-size codec and cannot be used in a sharding index pipeline`);
    }
    return codec.computeEncodedSize(size);
}
async function loadCodecs(chunkMeta) {
    let promises = chunkMeta.codecs.map(async (meta) => {
        let Codec = await registry.get(meta.name)?.();
        if (!Codec) {
            throw new UnknownCodecError(meta.name);
        }
        return { Codec, meta };
    });
    let arrayToArray = [];
    let arrayToBytes;
    let bytesToBytes = [];
    // Track the "current" data type through the codec chain. Array-to-array
    // codecs like cast_value may change the type, and subsequent codecs
    // (especially bytes) need to see the updated type.
    let currentMeta = { ...chunkMeta };
    for await (let { Codec, meta } of promises) {
        let codec = Codec.fromConfig(meta.configuration, currentMeta);
        switch (codec.kind) {
            case "array_to_array":
                arrayToArray.push({
                    name: meta.name,
                    codec: codec,
                });
                // Array-to-array codecs like cast_value may change the data type
                // (and derived metadata like fill_value) between the array's
                // declared type and what's stored on disk. We call getEncodedMeta
                // so that subsequent codecs in the chain — especially the bytes
                // codec — see the correct on-disk type and fill value.
                if (codec.getEncodedMeta) {
                    currentMeta = codec.getEncodedMeta(currentMeta);
                }
                break;
            case "array_to_bytes":
                arrayToBytes = {
                    name: meta.name,
                    codec: codec,
                };
                break;
            default:
                bytesToBytes.push({
                    name: meta.name,
                    codec: codec,
                });
        }
    }
    if (!arrayToBytes) {
        if (!isTypedArrayLikeMeta(currentMeta)) {
            throw new InvalidMetadataError(`Cannot encode ${currentMeta.dataType} to bytes without a codec`);
        }
        arrayToBytes = {
            name: "bytes",
            codec: BytesCodec.fromConfig({ endian: "little" }, currentMeta),
        };
    }
    return {
        arrayToArray,
        arrayToBytes,
        bytesToBytes,
    };
}
function isTypedArrayLikeMeta(meta) {
    return meta.dataType !== "v2:object" && meta.dataType !== "string";
}
//# sourceMappingURL=codecs.js.map