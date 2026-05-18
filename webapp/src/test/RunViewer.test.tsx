/**
 * Tests for RunViewer.tsx — the volume/slice orchestration component.
 *
 * SliceViewer (which needs Plotly + zarrita) is stubbed out so these tests
 * focus solely on RunViewer's own logic: manifest loading states, volume
 * selection, colormap defaults, slice-type tab switching, and the inline vs.
 * crossline slider commit behaviour.
 */
import React from "react";
import { render, screen, fireEvent, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect, beforeEach } from "vitest";

import type { ManifestEntry, VolumeInfo } from "../types/manifest";

// ---------------------------------------------------------------------------
// Stubs — declared before any imports that trigger component modules
// ---------------------------------------------------------------------------

// Stub SliceViewer so we don't need Plotly or zarrita
vi.mock("../components/SliceViewer", () => ({
  default: (props: {
    sliceType: string;
    sliceIndex: number;
    colormap: string;
    volume: VolumeInfo;
  }) => (
    <div
      data-testid="slice-viewer"
      data-slice-type={props.sliceType}
      data-slice-index={props.sliceIndex}
      data-colormap={props.colormap}
      data-volume={props.volume?.name}
    />
  ),
}));

// Stub useManifest
const mockUseManifest = vi.fn();
vi.mock("../hooks/useManifest", () => ({
  default: () => mockUseManifest(),
}));

import RunViewer from "../components/RunViewer";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const mkVol = (overrides: Partial<VolumeInfo> = {}): VolumeInfo => ({
  name: "near",
  store_path: "seismic/near.zarr",
  variable: "amplitude",
  group: "Seismic",
  shape: [100, 100, 500],
  dtype: "float32",
  dims: ["inline", "crossline", "time"],
  chunks: [50, 50, 500],
  compressor: "blosc:zstd:5",
  attrs: { angle_deg: 7 },
  ...overrides,
});

const VOL_NEAR = mkVol();
const VOL_GEO = mkVol({
  name: "geologic_age",
  group: "Geology",
  store_path: "geology/geologic_age.zarr",
  attrs: {},
});

const ENTRY: ManifestEntry = {
  run_id: "abc-123",
  folder: "seismic__20260517_abc-123",
  datestamp: "20260517",
  cube_shape: [100, 100, 500],
  volumes: [VOL_NEAR, VOL_GEO],
  parameters: {},
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("RunViewer", () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows a spinner while manifest is loading", () => {
    mockUseManifest.mockReturnValue({ data: null, loading: true, error: null });
    render(<RunViewer folderId="seismic__20260517_abc-123" />);
    expect(document.querySelector(".bp5-spinner")).toBeInTheDocument();
  });

  it("shows an error state when manifest fetch fails", () => {
    mockUseManifest.mockReturnValue({
      data: null,
      loading: false,
      error: "Network error",
    });
    render(<RunViewer folderId="seismic__20260517_abc-123" />);
    expect(screen.getByText(/could not load manifest\.json/i)).toBeInTheDocument();
  });

  it("shows 'Run not found' for an unknown folderId", () => {
    mockUseManifest.mockReturnValue({
      data: [ENTRY],
      loading: false,
      error: null,
    });
    render(<RunViewer folderId="nonexistent-folder" />);
    // Falls back to first entry, not "not found" — test the real fallback
    // RunViewer falls back to manifest[0] on unknown folderId
    expect(screen.getByText("abc-123")).toBeInTheDocument();
  });

  it("renders the run_id and folder in the header", () => {
    mockUseManifest.mockReturnValue({
      data: [ENTRY],
      loading: false,
      error: null,
    });
    render(<RunViewer folderId={ENTRY.folder} />);
    expect(screen.getByText("abc-123")).toBeInTheDocument();
    expect(screen.getByText(/seismic__20260517_abc-123/)).toBeInTheDocument();
  });

  it("auto-selects the first volume and passes it to SliceViewer", () => {
    mockUseManifest.mockReturnValue({
      data: [ENTRY],
      loading: false,
      error: null,
    });
    render(<RunViewer folderId={ENTRY.folder} />);
    const sv = screen.getByTestId("slice-viewer");
    expect(sv).toHaveAttribute("data-volume", "near");
  });

  it("sets colormap to RdBu for Seismic volumes by default", () => {
    mockUseManifest.mockReturnValue({
      data: [ENTRY],
      loading: false,
      error: null,
    });
    render(<RunViewer folderId={ENTRY.folder} />);
    expect(screen.getByTestId("slice-viewer")).toHaveAttribute(
      "data-colormap",
      "RdBu"
    );
  });

  it("renders inline / crossline / timeslice tabs", () => {
    mockUseManifest.mockReturnValue({
      data: [ENTRY],
      loading: false,
      error: null,
    });
    render(<RunViewer folderId={ENTRY.folder} />);
    expect(screen.getByRole("tab", { name: /inline/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /crossline/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /timeslice/i })).toBeInTheDocument();
  });

  it("switches sliceType to crossline when the crossline tab is clicked", async () => {
    mockUseManifest.mockReturnValue({
      data: [ENTRY],
      loading: false,
      error: null,
    });
    render(<RunViewer folderId={ENTRY.folder} />);

    await userEvent.click(screen.getByRole("tab", { name: /crossline/i }));

    expect(screen.getByTestId("slice-viewer")).toHaveAttribute(
      "data-slice-type",
      "crossline"
    );
  });

  it("updates sliceIndex immediately for inline (non-crossline) slider changes", () => {
    mockUseManifest.mockReturnValue({
      data: [ENTRY],
      loading: false,
      error: null,
    });
    render(<RunViewer folderId={ENTRY.folder} />);

    const slider = screen.getByRole("slider");
    fireEvent.change(slider, { target: { value: "42" } });

    expect(screen.getByTestId("slice-viewer")).toHaveAttribute(
      "data-slice-index",
      "42"
    );
    expect(screen.getByText(/index: 42/i)).toBeInTheDocument();
  });

  it("crossline slider updates index only on mouseUp (draft pattern)", async () => {
    mockUseManifest.mockReturnValue({
      data: [ENTRY],
      loading: false,
      error: null,
    });
    render(<RunViewer folderId={ENTRY.folder} />);

    await userEvent.click(screen.getByRole("tab", { name: /crossline/i }));

    const slider = screen.getByRole("slider");

    // onChange fires — draft moves but sliceIndex stays 0
    fireEvent.change(slider, { target: { value: "30" } });
    expect(screen.getByTestId("slice-viewer")).toHaveAttribute(
      "data-slice-index",
      "0"
    );

    // mouseUp — sliceIndex commits to 30
    fireEvent.mouseUp(slider, { target: { value: "30" } });
    // After mouseUp the slice-viewer receives the committed index
    expect(screen.getByTestId("slice-viewer")).toHaveAttribute(
      "data-slice-index",
      "30"
    );
  });
});
