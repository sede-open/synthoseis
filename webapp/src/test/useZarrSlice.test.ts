/**
 * Tests for useZarrSlice.ts — zarr chunk fetch hook.
 *
 * The entire `zarrita` module is mocked so no real HTTP calls or zarr stores
 * are needed.  Each test drives `mockZarrGet` to control what the hook sees.
 */
import { renderHook, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";

// ---------------------------------------------------------------------------
// Mock zarrita before the hook is imported.
// vi.hoisted() ensures the mocks are created before vi.mock() factory runs
// (vi.mock is hoisted above all imports / const declarations).
// FetchStore must be a real class — arrow functions can't be called with `new`.
// ---------------------------------------------------------------------------

const { mockZarrGet, mockZarrOpen } = vi.hoisted(() => ({
  mockZarrGet: vi.fn(),
  mockZarrOpen: vi.fn(),
}));

vi.mock("zarrita", () => ({
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  FetchStore: class FetchStore { constructor(_url: string) {} },
  withConsolidatedMetadata: vi.fn().mockImplementation((store: unknown) => store),
  root: vi.fn().mockReturnValue({ resolve: (p: string) => p }),
  open: (...args: unknown[]) => mockZarrOpen(...args),
  get: (...args: unknown[]) => mockZarrGet(...args),
}));

import useZarrSlice from "../hooks/useZarrSlice";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeArr() {
  // A 2×2 flat Float32Array for a slice of shape [2, 2]
  return {
    data: new Float32Array([1, 2, 3, 4]),
    shape: [2, 2] as [number, number],
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useZarrSlice", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockZarrOpen.mockResolvedValue({}); // array handle (not inspected)
    mockZarrGet.mockResolvedValue(makeArr());
  });

  it("returns loading=true before the fetch resolves", () => {
    // Hold get() forever
    mockZarrGet.mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(() =>
      useZarrSlice("folder", "seismic/near.zarr", "amplitude", "inline", 0)
    );

    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("returns Float32Array data and shape on a successful inline fetch", async () => {
    const { result } = renderHook(() =>
      useZarrSlice("folder", "seismic/near.zarr", "amplitude", "inline", 1)
    );

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toBeInstanceOf(Float32Array);
    expect(result.current.shape).toEqual([2, 2]);
    expect(result.current.error).toBeNull();
  });

  it("returns an error string when zarrita throws", async () => {
    mockZarrGet.mockRejectedValueOnce(new Error("zarr fetch failed"));

    const { result } = renderHook(() =>
      useZarrSlice("folder", "seismic/near.zarr", "amplitude", "inline", 2)
    );

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toBeNull();
    expect(result.current.error).toMatch(/zarr fetch failed/);
  });

  it("calls zarr.get with [index, null, null] for inline slice", async () => {
    const { result } = renderHook(() =>
      useZarrSlice("folder", "seismic/near.zarr", "amplitude", "inline", 5)
    );

    await waitFor(() => expect(result.current.loading).toBe(false));

    // Second argument to zarr.get is the selection
    const selection = mockZarrGet.mock.calls[0][1];
    expect(selection).toEqual([5, null, null]);
  });

  it("calls zarr.get with [null, index, null] for crossline slice", async () => {
    const { result } = renderHook(() =>
      useZarrSlice("folder", "seismic/near.zarr", "amplitude", "crossline", 10)
    );

    await waitFor(() => expect(result.current.loading).toBe(false));

    const selection = mockZarrGet.mock.calls[0][1];
    expect(selection).toEqual([null, 10, null]);
  });

  it("calls zarr.get with [null, null, index] for timeslice", async () => {
    const { result } = renderHook(() =>
      useZarrSlice("folder", "seismic/near.zarr", "amplitude", "timeslice", 20)
    );

    await waitFor(() => expect(result.current.loading).toBe(false));

    const selection = mockZarrGet.mock.calls[0][1];
    expect(selection).toEqual([null, null, 20]);
  });

  it("serves the second identical request from the LRU cache (no extra zarr.get call)", async () => {
    // First render
    const { result: r1, unmount: u1 } = renderHook(() =>
      useZarrSlice("folder-cache", "seismic/cache.zarr", "amplitude", "inline", 7)
    );
    await waitFor(() => expect(r1.current.loading).toBe(false));
    expect(mockZarrGet).toHaveBeenCalledTimes(1);
    u1();

    // Second render — same key, should hit cache
    const { result: r2 } = renderHook(() =>
      useZarrSlice("folder-cache", "seismic/cache.zarr", "amplitude", "inline", 7)
    );
    await waitFor(() => expect(r2.current.loading).toBe(false));

    // zarr.get must still have been called exactly once (cache served second)
    expect(mockZarrGet).toHaveBeenCalledTimes(1);
    expect(r2.current.data).toBeInstanceOf(Float32Array);
  });
});
