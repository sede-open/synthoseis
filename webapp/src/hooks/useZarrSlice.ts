import React from "react";
import * as zarr from "zarrita";

type SliceType = "inline" | "crossline" | "timeslice";

interface ZarrSliceResult {
  data: Float32Array | null;
  shape: [number, number] | null;
  loading: boolean;
  error: string | null;
}

/** Simple LRU cache with a fixed capacity. */
class LRUCache<K, V> {
  private capacity: number;
  private map = new Map<K, V>();

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  get(key: K): V | undefined {
    const value = this.map.get(key);
    if (value !== undefined) {
      // Move to end (most recently used)
      this.map.delete(key);
      this.map.set(key, value);
    }
    return value;
  }

  set(key: K, value: V): void {
    if (this.map.has(key)) {
      this.map.delete(key);
    } else if (this.map.size >= this.capacity) {
      // Delete least recently used (first entry)
      const firstKey = this.map.keys().next().value!;
      this.map.delete(firstKey);
    }
    this.map.set(key, value);
  }
}

// Module-level LRU cache (5 slices max ≈ 7.5 MB at ~1.5 MB each)
type CachedSlice = { data: Float32Array; shape: [number, number] };
const sliceCache = new LRUCache<string, CachedSlice>(5);

function makeCacheKey(
  folderPath: string,
  storePath: string,
  variable: string,
  sliceType: SliceType,
  index: number
): string {
  return `${folderPath}|${storePath}|${variable}|${sliceType}|${index}`;
}

/**
 * Opens a zarr v3 store (via zarrita.js FetchStore),
 * extracts the named variable, and returns a 2-D slice as a Float32Array.
 *
 * LRU cache of last 5 slices to avoid re-fetching on tab/axis toggle.
 * Crossline slice debounce is handled in RunViewer (mouseUp), not here.
 */
export default function useZarrSlice(
  folderPath: string,
  storePath: string,
  variable: string,
  sliceType: SliceType,
  sliceIndex: number
): ZarrSliceResult {
  const [result, setResult] = React.useState<ZarrSliceResult>({
    data: null,
    shape: null,
    loading: false,
    error: null,
  });

  React.useEffect(() => {
    if (!folderPath || !storePath || !variable) return;

    let cancelled = false;

    const cacheKey = makeCacheKey(folderPath, storePath, variable, sliceType, sliceIndex);
    const cached = sliceCache.get(cacheKey);
    if (cached) {
      setResult({ data: cached.data, shape: cached.shape, loading: false, error: null });
      return;
    }

    setResult((prev) => ({ ...prev, loading: true, error: null }));

    async function fetchSlice() {
      try {
        // folderPath is an absolute filesystem path (e.g. /Users/.../seismic__0517_...).
        // Build an HTTP URL via /api/zarr, which serves files from the filesystem.
        // Strip trailing slash from folderPath; storePath is already relative.
        const storeUrl =
          window.location.origin +
          "/api/zarr" +
          folderPath.replace(/\/$/, "") +
          "/" +
          storePath;

        const rawStore = new zarr.FetchStore(storeUrl);

        // Open the variable array directly by path — no consolidated metadata needed.
        // zarrita fetches {storeUrl}/{variable}/zarr.json on demand.
        const arr = await zarr.open(zarr.root(rawStore).resolve("/" + variable), { kind: "array" });

        // Build selection for the requested slice
        const sel = buildSelection(sliceType, sliceIndex);

        // Fetch the chunk
        const chunk = await zarr.get(arr, sel);

        if (cancelled) return;

        // Convert to Float32Array
        const rawData = chunk.data;
        const float32: Float32Array =
          rawData instanceof Float32Array
            ? rawData
            : new Float32Array(rawData as ArrayLike<number>);

        const shape = chunk.shape as [number, number];
        const entry: CachedSlice = { data: float32, shape };
        sliceCache.set(cacheKey, entry);

        setResult({ data: float32, shape, loading: false, error: null });
      } catch (err) {
        if (cancelled) return;
        const msg =
          err instanceof Error ? err.message : "Could not load volume chunk";
        setResult({ data: null, shape: null, loading: false, error: msg });
      }
    }

    void fetchSlice();
    return () => {
      cancelled = true;
    };
  }, [folderPath, storePath, variable, sliceType, sliceIndex]);

  return result;
}

/**
 * Build a zarrita selection for the given slice type and index.
 *
 * zarrita v0.7:
 *   null  = full axis (`:` in Python)
 *   number = scalar index (dimension is dropped from result shape)
 */
function buildSelection(
  sliceType: SliceType,
  index: number
): (null | number)[] {
  switch (sliceType) {
    case "inline":
      // [index, :, :] → shape [crossline, time|horizon]
      return [index, null, null];
    case "crossline":
      // [:, index, :] → shape [inline, time|horizon]
      return [null, index, null];
    case "timeslice":
    default:
      // [:, :, index] → shape [inline, crossline]
      return [null, null, index];
  }
}
