/**
 * Tests for SliceViewer.tsx.
 *
 * Plotly is loaded from CDN in production (not bundled), so we mock the
 * react-plotly.js factory to return a simple div that captures the z-data
 * and layout props.  zarrita is mocked to return a controlled Float32Array.
 */
import React from "react";
import { render, screen, waitFor } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach } from "vitest";

import type { VolumeInfo } from "../types/manifest";

// ---------------------------------------------------------------------------
// Mock react-plotly.js factory (must be hoisted before SliceViewer import)
// ---------------------------------------------------------------------------

vi.mock("react-plotly.js/factory", () => {
  // SliceViewer.tsx references the bare global `Plotly` (CDN-injected in
  // production) at module level.  Stub it here so the ReferenceError never
  // fires — this factory runs before the SliceViewer module initialiser.
  (globalThis as Record<string, unknown>).Plotly = {};
  return {
    default: () =>
      function MockPlot(props: { data: unknown[]; layout: unknown }) {
        return (
          <div
            data-testid="plotly-heatmap"
            data-colorscale={(props.data as any)[0]?.colorscale}
            data-x-title={(props.layout as any)?.xaxis?.title?.text}
            data-y-title={(props.layout as any)?.yaxis?.title?.text}
          />
        );
      },
  };
});

// ---------------------------------------------------------------------------
// Mock useZarrSlice
// ---------------------------------------------------------------------------

const mockUseZarrSlice = vi.fn();

vi.mock("../hooks/useZarrSlice", () => ({
  default: (...args: unknown[]) => mockUseZarrSlice(...args),
}));

import SliceViewer from "../components/SliceViewer";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const VOL: VolumeInfo = {
  name: "near",
  store_path: "seismic/near.zarr",
  variable: "amplitude",
  group: "Seismic",
  shape: [4, 4, 4],
  dtype: "float32",
  dims: ["inline", "crossline", "time"],
  chunks: [4, 4, 4],
  compressor: "blosc:zstd:5",
  attrs: { angle_deg: 7 },
};

// A 2×3 flat Float32Array representing a slice with shape [2, 3]
const MOCK_DATA = new Float32Array([1, 2, 3, 4, 5, 6]);
const MOCK_SHAPE: [number, number] = [2, 3];

const DEFAULT_SLICE_STATE = {
  data: MOCK_DATA,
  shape: MOCK_SHAPE,
  loading: false,
  error: null,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("SliceViewer", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseZarrSlice.mockReturnValue(DEFAULT_SLICE_STATE);
  });

  it("renders the Plotly heatmap when slice data is available", () => {
    render(
      <SliceViewer
        folderPath="seismic__20260517_abc"
        volume={VOL}
        sliceType="inline"
        sliceIndex={0}
        colormap="RdBu"
      />
    );
    expect(screen.getByTestId("plotly-heatmap")).toBeInTheDocument();
  });

  it("passes the colormap to the Plotly component", () => {
    render(
      <SliceViewer
        folderPath="seismic__20260517_abc"
        volume={VOL}
        sliceType="inline"
        sliceIndex={0}
        colormap="Viridis"
      />
    );
    expect(screen.getByTestId("plotly-heatmap")).toHaveAttribute(
      "data-colorscale",
      "Viridis"
    );
  });

  it("shows a spinner while the slice is loading", () => {
    mockUseZarrSlice.mockReturnValue({
      data: null,
      shape: null,
      loading: true,
      error: null,
    });
    render(
      <SliceViewer
        folderPath="seismic__20260517_abc"
        volume={VOL}
        sliceType="inline"
        sliceIndex={0}
        colormap="RdBu"
      />
    );
    expect(document.querySelector(".bp5-spinner")).toBeInTheDocument();
  });

  it("shows an error non-ideal state when useZarrSlice returns an error", () => {
    mockUseZarrSlice.mockReturnValue({
      data: null,
      shape: null,
      loading: false,
      error: "Chunk fetch failed",
    });
    render(
      <SliceViewer
        folderPath="seismic__20260517_abc"
        volume={VOL}
        sliceType="inline"
        sliceIndex={0}
        colormap="RdBu"
      />
    );
    expect(
      screen.getByText(/could not load volume chunk/i)
    ).toBeInTheDocument();
  });

  it("passes inline x-axis label 'crossline' and y-axis label 'time'", () => {
    render(
      <SliceViewer
        folderPath="seismic__20260517_abc"
        volume={VOL}
        sliceType="inline"
        sliceIndex={5}
        colormap="RdBu"
      />
    );
    const chart = screen.getByTestId("plotly-heatmap");
    expect(chart).toHaveAttribute("data-x-title", "crossline");
    expect(chart).toHaveAttribute("data-y-title", "time");
  });

  it("passes crossline x-axis label 'inline' and y-axis label 'time'", () => {
    render(
      <SliceViewer
        folderPath="seismic__20260517_abc"
        volume={VOL}
        sliceType="crossline"
        sliceIndex={10}
        colormap="RdBu"
      />
    );
    const chart = screen.getByTestId("plotly-heatmap");
    expect(chart).toHaveAttribute("data-x-title", "inline");
    expect(chart).toHaveAttribute("data-y-title", "time");
  });

  it("calls useZarrSlice with the correct arguments", () => {
    render(
      <SliceViewer
        folderPath="seismic__20260517_abc"
        volume={VOL}
        sliceType="timeslice"
        sliceIndex={25}
        colormap="Greys"
      />
    );
    expect(mockUseZarrSlice).toHaveBeenCalledWith(
      "seismic__20260517_abc",
      "seismic/near.zarr",
      "amplitude",
      "timeslice",
      25
    );
  });
});
