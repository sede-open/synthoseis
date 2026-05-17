import { UnsupportedError } from "../errors.js";
import { defineStoreExtension } from "./define.js";
function assertRangeCapable(store) {
    if (!store.getRange) {
        throw new UnsupportedError("`zarr.withRangeCoalescing` requires a store with getRange");
    }
}
function mergeSignals(signals) {
    let present = [];
    for (let s of signals)
        if (s)
            present.push(s);
    if (present.length === 0)
        return undefined;
    if (present.length === 1)
        return present[0];
    return AbortSignal.any(present);
}
const DEFAULT_COALESCE_SIZE = 32768;
function groupRequests(sorted, coalesceSize) {
    if (sorted.length === 0)
        return [];
    let groups = [];
    let current = [sorted[0]];
    let groupStart = sorted[0].offset;
    let groupEnd = sorted[0].offset + sorted[0].length;
    for (let i = 1; i < sorted.length; i++) {
        let req = sorted[i];
        let reqEnd = req.offset + req.length;
        if (req.offset <= groupEnd + coalesceSize) {
            current.push(req);
            groupEnd = Math.max(groupEnd, reqEnd);
        }
        else {
            groups.push({
                offset: groupStart,
                length: groupEnd - groupStart,
                requests: current,
            });
            current = [req];
            groupStart = req.offset;
            groupEnd = reqEnd;
        }
    }
    groups.push({
        offset: groupStart,
        length: groupEnd - groupStart,
        requests: current,
    });
    return groups;
}
function emitFlush(onFlush, report) {
    if (!onFlush)
        return;
    try {
        onFlush(report);
    }
    catch (err) {
        console.warn("withRangeCoalescing: onFlush threw, swallowing:", err);
    }
}
/**
 * Wraps a store with microtask-tick range batching: concurrent `getRange`
 * calls within a single microtask are grouped by path, coalesced across
 * small byte gaps, and issued as a single fetch per group. The coalesced
 * blob is sliced on return and each caller receives exactly the bytes they
 * asked for.
 *
 * `withRangeCoalescing` carries no cache state. Pair with `withByteCaching`
 * if you want cross-call caching.
 *
 * ```ts
 * import * as zarr from "zarrita";
 *
 * let store = zarr.withRangeCoalescing(
 *   new zarr.FetchStore("https://example.com/data.zarr"),
 *   { coalesceSize: 32768 },
 * );
 * ```
 */
export const withRangeCoalescing = defineStoreExtension((_store, opts = {}) => {
    assertRangeCapable(_store);
    let store = _store;
    let boundGetRange = store.getRange.bind(store);
    let coalesceSize = opts.coalesceSize ?? DEFAULT_COALESCE_SIZE;
    let onFlush = opts.onFlush;
    let pending = new Map();
    let scheduled = false;
    async function flush() {
        let work = new Map(pending);
        pending.clear();
        scheduled = false;
        let pathPromises = [];
        for (let [path, requests] of work) {
            requests.sort((a, b) => a.offset - b.offset);
            let groups = groupRequests(requests, coalesceSize);
            emitFlush(onFlush, {
                path,
                groupCount: groups.length,
                requestCount: requests.length,
                bytesFetched: groups.reduce((sum, g) => sum + g.length, 0),
            });
            pathPromises.push(fetchGroups(path, groups));
        }
        await Promise.all(pathPromises);
    }
    async function fetchGroups(path, groups) {
        await Promise.all(groups.map(async (group) => {
            let signal = mergeSignals(group.requests.map((r) => r.signal));
            try {
                let data = await boundGetRange(path, { offset: group.offset, length: group.length }, { signal });
                if (data && data.length < group.length) {
                    throw new Error(`Short read: expected ${group.length} bytes but received ${data.length}`);
                }
                for (let req of group.requests) {
                    if (!data) {
                        req.resolve(undefined);
                        continue;
                    }
                    let start = req.offset - group.offset;
                    let slice = data.slice(start, start + req.length);
                    req.resolve(slice);
                }
            }
            catch (err) {
                for (let req of group.requests) {
                    req.reject(err);
                }
            }
        }));
    }
    return {
        getRange(key, range, options) {
            // Suffix reads (shard index fetches) can't be coalesced because
            // file size is unknown until the response arrives. Pass through.
            if ("suffixLength" in range) {
                return boundGetRange(key, range, options);
            }
            let { offset, length } = range;
            return new Promise((resolve, reject) => {
                let reqs = pending.get(key);
                if (!reqs) {
                    reqs = [];
                    pending.set(key, reqs);
                }
                reqs.push({
                    offset,
                    length,
                    signal: options?.signal,
                    resolve,
                    reject,
                });
                if (!scheduled) {
                    scheduled = true;
                    queueMicrotask(() => flush());
                }
            });
        },
    };
});
//# sourceMappingURL=range-coalescing.js.map