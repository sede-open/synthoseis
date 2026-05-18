/**
 * Tests for ColormapSelector.tsx, VolumeSelector.tsx, and MetadataPanel.tsx.
 *
 * These are pure-presentational components with no network calls.
 */
import React from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect } from "vitest";

import ColormapSelector, { colormapForGroup } from "../components/ColormapSelector";
import VolumeSelector from "../components/VolumeSelector";
import MetadataPanel from "../components/MetadataPanel";
import type { ManifestEntry, VolumeInfo } from "../types/manifest";

// ---------------------------------------------------------------------------
// Shared fixtures
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

const VOL_SEISMIC = mkVol();
const VOL_GEOLOGY = mkVol({ name: "geologic_age", group: "Geology", store_path: "geology/geologic_age.zarr", attrs: {} });
const VOL_CLOSURE = mkVol({ name: "closure_volume", group: "Closures", store_path: "closures/closures.zarr", attrs: {} });

const MOCK_ENTRY: ManifestEntry = {
  run_id: "abc-123",
  folder: "seismic__20260517_abc-123",
  datestamp: "20260517",
  cube_shape: [100, 100, 500],
  volumes: [VOL_SEISMIC, VOL_GEOLOGY],
  parameters: { cube_shape: "[100, 100, 500]", digi: "4" },
};

// ---------------------------------------------------------------------------
// ColormapSelector
// ---------------------------------------------------------------------------

describe("ColormapSelector", () => {
  it("renders the currently selected colormap", () => {
    render(<ColormapSelector value="RdBu" onChange={() => {}} />);
    const select = screen.getByRole("combobox") as HTMLSelectElement;
    expect(select.value).toBe("RdBu");
  });

  it("lists at least the standard colormaps", () => {
    render(<ColormapSelector value="RdBu" onChange={() => {}} />);
    const options = Array.from(
      screen.getByRole("combobox").querySelectorAll("option")
    ).map((o) => o.textContent);
    expect(options).toContain("RdBu");
    expect(options).toContain("Greys");
    expect(options).toContain("Viridis");
    expect(options).toContain("Jet");
  });

  it("calls onChange with new value when selection changes", async () => {
    const onChange = vi.fn();
    render(<ColormapSelector value="RdBu" onChange={onChange} />);
    await userEvent.selectOptions(screen.getByRole("combobox"), "Viridis");
    expect(onChange).toHaveBeenCalledWith("Viridis");
  });

  it("colormapForGroup returns correct defaults per group", () => {
    expect(colormapForGroup("Seismic")).toBe("RdBu");
    expect(colormapForGroup("Geology")).toBe("Greys");
    expect(colormapForGroup("Closures")).toBe("Hot");
    expect(colormapForGroup("Horizons")).toBe("Viridis");
    expect(colormapForGroup("QC")).toBe("Viridis");
    expect(colormapForGroup("Unknown")).toBe("Viridis");
  });
});

// ---------------------------------------------------------------------------
// VolumeSelector
// ---------------------------------------------------------------------------

describe("VolumeSelector", () => {
  it("shows selected volume name on the trigger button", () => {
    render(
      <VolumeSelector
        volumes={[VOL_SEISMIC, VOL_GEOLOGY]}
        selected={VOL_SEISMIC}
        onChange={() => {}}
      />
    );
    expect(screen.getByRole("button")).toHaveTextContent("near (7°)");
  });

  it("shows 'Select volume…' when nothing is selected", () => {
    render(
      <VolumeSelector
        volumes={[VOL_SEISMIC]}
        selected={null}
        onChange={() => {}}
      />
    );
    expect(screen.getByRole("button")).toHaveTextContent("Select volume…");
  });

  it("omits angle suffix for non-seismic volumes", () => {
    render(
      <VolumeSelector
        volumes={[VOL_GEOLOGY]}
        selected={VOL_GEOLOGY}
        onChange={() => {}}
      />
    );
    expect(screen.getByRole("button")).toHaveTextContent("geologic_age");
    expect(screen.getByRole("button")).not.toHaveTextContent("°");
  });

  it("opens the popover and shows volume items on trigger click", async () => {
    render(
      <VolumeSelector
        volumes={[VOL_SEISMIC, VOL_GEOLOGY]}
        selected={null}
        onChange={() => {}}
      />
    );
    await userEvent.click(screen.getByRole("button"));
    // Blueprint Select renders menu items in a portal
    const items = document.querySelectorAll('[role="option"], .bp5-menu-item');
    const texts = Array.from(items).map((el) => el.textContent ?? "");
    expect(texts.some((t) => t.includes("near"))).toBe(true);
    expect(texts.some((t) => t.includes("geologic_age"))).toBe(true);
  });

  it("calls onChange when an item is selected", async () => {
    const onChange = vi.fn();
    render(
      <VolumeSelector
        volumes={[VOL_SEISMIC, VOL_GEOLOGY]}
        selected={null}
        onChange={onChange}
      />
    );
    await userEvent.click(screen.getByRole("button"));
    // Blueprint Select renders items into document.body via a portal
    const menuItems = document.querySelectorAll(".bp5-menu-item");
    const geoItem = Array.from(menuItems).find((el) =>
      el.textContent?.includes("geologic_age")
    ) as HTMLElement | undefined;
    expect(geoItem).toBeDefined();
    if (geoItem) {
      await userEvent.click(geoItem);
      // Blueprint Select passes (item, syntheticEvent) — match item only
      expect(onChange).toHaveBeenCalledWith(VOL_GEOLOGY, expect.anything());
    }
  });
});

// ---------------------------------------------------------------------------
// MetadataPanel
// ---------------------------------------------------------------------------

describe("MetadataPanel", () => {
  it("renders the Parameters tab by default with key/value rows", () => {
    render(<MetadataPanel entry={MOCK_ENTRY} selectedVolume={null} />);
    expect(screen.getByRole("tab", { name: /parameters/i })).toBeInTheDocument();
    expect(screen.getByText("cube_shape")).toBeInTheDocument();
    expect(screen.getByText("[100, 100, 500]")).toBeInTheDocument();
    expect(screen.getByText("digi")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
  });

  it("shows 'No parameters available' when parameters object is empty", () => {
    const emptyEntry: ManifestEntry = { ...MOCK_ENTRY, parameters: {} };
    render(<MetadataPanel entry={emptyEntry} selectedVolume={null} />);
    expect(screen.getByText(/no parameters available/i)).toBeInTheDocument();
  });

  it("renders volume info in the Volumes tab", async () => {
    render(<MetadataPanel entry={MOCK_ENTRY} selectedVolume={null} />);
    await userEvent.click(screen.getByRole("tab", { name: /volumes/i }));
    expect(screen.getByText("near")).toBeInTheDocument();
    expect(screen.getByText("geologic_age")).toBeInTheDocument();
  });

  it("highlights the selected volume with a blue border", async () => {
    render(<MetadataPanel entry={MOCK_ENTRY} selectedVolume={VOL_SEISMIC} />);
    await userEvent.click(screen.getByRole("tab", { name: /volumes/i }));
    // The selected card sets border: "1px solid #8abbff".
    // jsdom normalises #8abbff → rgb(138, 187, 255), so we use toHaveStyle.
    const nearHeading = screen.getByText("near");
    const selectedCard = nearHeading.closest("div[style]");
    expect(selectedCard).not.toBeNull();
    expect(selectedCard).toHaveStyle({ border: "1px solid #8abbff" });
  });

  it("shows shape, dtype, dims, chunks, and compressor for each volume", async () => {
    render(<MetadataPanel entry={MOCK_ENTRY} selectedVolume={null} />);
    await userEvent.click(screen.getByRole("tab", { name: /volumes/i }));
    // Shape: 100 × 100 × 500
    const shapeCells = screen.getAllByText(/100 × 100 × 500/);
    expect(shapeCells.length).toBeGreaterThan(0);
    expect(screen.getAllByText("float32").length).toBeGreaterThan(0);
  });
});
