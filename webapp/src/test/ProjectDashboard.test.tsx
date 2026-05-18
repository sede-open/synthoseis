/**
 * Tests for ProjectDashboard.tsx and RunCard.tsx.
 *
 * ProjectDashboard delegates data-fetching to useManifest and renders a grid
 * of RunCard components.  RunCard handles click navigation and parameter display.
 */
import React from "react";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect, beforeEach } from "vitest";

import type { ManifestEntry, VolumeInfo } from "../types/manifest";

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

const MOCK_VOLUME: VolumeInfo = {
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
};

const MOCK_ENTRY: ManifestEntry = {
  run_id: "abc-123",
  folder: "seismic__20260517_abc-123",
  datestamp: "20260517",
  cube_shape: [100, 100, 500],
  volumes: [MOCK_VOLUME],
  parameters: {
    incident_angles: "[7, 15, 24]",
    closure_types: '["simple", "faulted"]',
  },
};

const MOCK_ENTRY_2: ManifestEntry = {
  run_id: "def-456",
  folder: "seismic__20260518_def-456",
  datestamp: "20260518",
  cube_shape: [300, 300, 1250],
  volumes: [MOCK_VOLUME, { ...MOCK_VOLUME, name: "mid", attrs: { angle_deg: 15 } }],
  parameters: {},
};

// ---------------------------------------------------------------------------
// Mock useManifest
// ---------------------------------------------------------------------------

const mockUseManifest = vi.fn();

vi.mock("../hooks/useManifest", () => ({
  default: () => mockUseManifest(),
}));

import ProjectDashboard from "../components/ProjectDashboard";
import RunCard from "../components/RunCard";

// ---------------------------------------------------------------------------
// ProjectDashboard
// ---------------------------------------------------------------------------

describe("ProjectDashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Set a project folder so the component skips the placeholder and
    // proceeds to the loading / error / data states under test.
    localStorage.setItem("synthoseis_project_folder", "/test/output");
  });

  afterEach(() => {
    localStorage.clear();
  });

  it("shows a spinner while loading", async () => {
    mockUseManifest.mockReturnValue({ data: null, loading: true, error: null });
    render(<ProjectDashboard />);
    expect(document.querySelector(".bp5-spinner")).toBeInTheDocument();
  });

  it("shows error non-ideal state on fetch failure", () => {
    mockUseManifest.mockReturnValue({
      data: null,
      loading: false,
      error: "HTTP 404",
    });
    render(<ProjectDashboard />);
    expect(screen.getByText(/could not load manifest/i)).toBeInTheDocument();
    expect(screen.getByText(/HTTP 404/)).toBeInTheDocument();
  });

  it("shows empty state when manifest is empty", () => {
    mockUseManifest.mockReturnValue({ data: [], loading: false, error: null });
    render(<ProjectDashboard />);
    expect(screen.getByText(/no runs found/i)).toBeInTheDocument();
  });

  it("renders a RunCard for each manifest entry", async () => {
    mockUseManifest.mockReturnValue({
      data: [MOCK_ENTRY, MOCK_ENTRY_2],
      loading: false,
      error: null,
    });
    render(<ProjectDashboard />);
    expect(screen.getByText("abc-123")).toBeInTheDocument();
    expect(screen.getByText("def-456")).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// RunCard
// ---------------------------------------------------------------------------

describe("RunCard", () => {
  it("renders run_id, folder and datestamp", () => {
    render(<RunCard entry={MOCK_ENTRY} />);
    expect(screen.getByText("abc-123")).toBeInTheDocument();
    expect(screen.getByText("seismic__20260517_abc-123")).toBeInTheDocument();
    expect(screen.getByText("20260517")).toBeInTheDocument();
  });

  it("renders cube_shape as a dimension tag", () => {
    render(<RunCard entry={MOCK_ENTRY} />);
    expect(screen.getByText("100 × 100 × 500")).toBeInTheDocument();
  });

  it("shows volume count", () => {
    render(<RunCard entry={MOCK_ENTRY} />);
    expect(screen.getByText("1 volume")).toBeInTheDocument();

    render(<RunCard entry={MOCK_ENTRY_2} />);
    expect(screen.getByText("2 volumes")).toBeInTheDocument();
  });

  it("parses and displays incident_angles from parameters", () => {
    render(<RunCard entry={MOCK_ENTRY} />);
    expect(screen.getByText(/7, 15, 24/)).toBeInTheDocument();
  });

  it("parses and displays closure_types from parameters", () => {
    render(<RunCard entry={MOCK_ENTRY} />);
    expect(screen.getByText(/simple, faulted/)).toBeInTheDocument();
  });

  it("handles missing parameters gracefully (no crash, no angles/closures shown)", () => {
    render(<RunCard entry={MOCK_ENTRY_2} />);
    expect(screen.queryByText(/Angles:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Closures:/)).not.toBeInTheDocument();
  });

  it("navigates to #/run/:folder on click", async () => {
    render(<RunCard entry={MOCK_ENTRY} />);
    await userEvent.click(screen.getByText("abc-123"));
    expect(window.location.hash).toBe(
      `#/run/${encodeURIComponent(MOCK_ENTRY.folder)}`
    );
  });
});
