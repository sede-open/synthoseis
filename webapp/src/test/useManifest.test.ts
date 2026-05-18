/**
 * Tests for useManifest.ts — the manifest.json fetch hook.
 *
 * `global.fetch` is mocked; the MANIFEST_URL constant (./manifest.json)
 * is not changed since it doesn't matter what URL is called — we always
 * control the mock response.
 */
import { renderHook, act, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";

import type { ManifestEntry } from "../types/manifest";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const MOCK_MANIFEST: ManifestEntry[] = [
  {
    run_id: "abc-123",
    folder: "seismic__20260517_abc-123",
    datestamp: "20260517",
    cube_shape: [100, 100, 500],
    volumes: [],
    parameters: {},
  },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockFetchOk(body: unknown) {
  vi.spyOn(global, "fetch").mockResolvedValueOnce({
    ok: true,
    status: 200,
    json: () => Promise.resolve(body),
  } as Response);
}

function mockFetchError(status: number, statusText = "Error") {
  vi.spyOn(global, "fetch").mockResolvedValueOnce({
    ok: false,
    status,
    statusText,
    json: () => Promise.resolve({}),
  } as Response);
}

function mockFetchNetworkFailure() {
  vi.spyOn(global, "fetch").mockRejectedValueOnce(new Error("Network failure"));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

import useManifest from "../hooks/useManifest";

describe("useManifest", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("starts in loading state with data=null and error=null", () => {
    mockFetchOk(MOCK_MANIFEST);
    const { result } = renderHook(() => useManifest());
    // Synchronously after hook creation — not yet resolved
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it("populates data and clears loading after a successful fetch", async () => {
    mockFetchOk(MOCK_MANIFEST);
    const { result } = renderHook(() => useManifest());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toEqual(MOCK_MANIFEST);
    expect(result.current.error).toBeNull();
  });

  it("sets error and clears loading on a non-200 HTTP response", async () => {
    mockFetchError(404, "Not Found");
    const { result } = renderHook(() => useManifest());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toBeNull();
    expect(result.current.error).toMatch(/HTTP 404/);
  });

  it("sets error on a network-level fetch failure", async () => {
    mockFetchNetworkFailure();
    const { result } = renderHook(() => useManifest());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.data).toBeNull();
    expect(result.current.error).toMatch(/Network failure/);
  });

  it("does not update state after the component has unmounted (cancellation)", async () => {
    // Use a deferred promise so we can unmount before the fetch resolves
    let resolveFetch!: (v: Response) => void;
    vi.spyOn(global, "fetch").mockReturnValueOnce(
      new Promise<Response>((res) => {
        resolveFetch = res;
      })
    );

    const { result, unmount } = renderHook(() => useManifest());
    unmount();

    // Now resolve after unmount — should not throw or update
    await act(async () => {
      resolveFetch({
        ok: true,
        status: 200,
        json: () => Promise.resolve(MOCK_MANIFEST),
      } as Response);
      // Give React a tick to process any state updates
      await Promise.resolve();
    });

    // State should remain as it was at unmount time (loading=true, data=null)
    expect(result.current.data).toBeNull();
  });
});
