import React from "react";
import { NonIdealState, Spinner, Button } from "@blueprintjs/core";
import useManifest from "../hooks/useManifest";
import type { ManifestEntry, VolumeInfo } from "../types/manifest";
import VolumeSelector from "./VolumeSelector";
import SliceViewer from "./SliceViewer";
import ColormapSelector from "./ColormapSelector";
import MetadataPanel from "./MetadataPanel";
import { colormapForGroup } from "./ColormapSelector";

interface RunViewerProps {
  folderId: string;
}

export default function RunViewer({ folderId }: RunViewerProps): React.ReactElement {
  // Re-use the project folder the user last loaded in ProjectDashboard so the
  // manifest fetch is scoped to the same folder (avoids a DB-only fallback).
  const projectFolder = React.useMemo(() => {
    try {
      return localStorage.getItem("synthoseis_project_folder") || null;
    } catch {
      return null;
    }
  }, []);
  const { data: manifest, loading, error } = useManifest(projectFolder);

  const [selectedVolume, setSelectedVolume] = React.useState<VolumeInfo | null>(null);
  const [sliceType, setSliceType] = React.useState<"inline" | "crossline" | "timeslice">("inline");
  const [sliceIndex, setSliceIndex] = React.useState(0);
  // crosslineDraft tracks the visual thumb position during drag without triggering zarr fetches.
  const [crosslineDraft, setCrosslineDraft] = React.useState(0);
  // Sync draft when sliceIndex resets externally (e.g. volume switch).
  React.useEffect(() => { setCrosslineDraft(sliceIndex); }, [sliceIndex]);
  const [colormap, setColormap] = React.useState<string>(() => {
    try { return localStorage.getItem("synthoseis_colormap") ?? "seismics"; } catch { return "seismics"; }
  });
  const [colormapReversed, setColormapReversed] = React.useState<boolean>(() => {
    try { return localStorage.getItem("synthoseis_colormap_reversed") === "true"; } catch { return false; }
  });

  // Persist colormap selection to localStorage
  const handleColormapChange = (value: string) => {
    setColormap(value);
    try { localStorage.setItem("synthoseis_colormap", value); } catch { /* ignore */ }
  };
  const handleReversedChange = (value: boolean) => {
    setColormapReversed(value);
    try { localStorage.setItem("synthoseis_colormap_reversed", String(value)); } catch { /* ignore */ }
  };

  const entry: ManifestEntry | undefined = React.useMemo(() => {
    if (!manifest) return undefined;
    // Find by folder name; fallback to first entry on unknown folderId
    return manifest.find((e) => e.folder === folderId) ?? manifest[0];
  }, [manifest, folderId]);

  // Auto-select first volume when entry loads
  React.useEffect(() => {
    if (entry && entry.volumes.length > 0) {
      setSelectedVolume(entry.volumes[0]);
      setSliceIndex(0);
    }
  }, [entry]);

  // Reset slice index when volume or slice type changes
  React.useEffect(() => {
    setSliceIndex(0);
  }, [selectedVolume, sliceType]);

  // Update selected volume (no colormap reset — user keeps their choice)
  const handleVolumeChange = (vol: VolumeInfo) => {
    setSelectedVolume(vol);
    setSliceIndex(0);
  };

  // Build the absolute folder path used for slice serving.
  // Guard: if projectFolder already ends with the run folder name (user entered
  // the run subfolder directly instead of the parent), don't double it.
  const runFolderPath = React.useMemo(() => {
    if (!entry) return "";
    if (!projectFolder) return entry.folder;
    const suffix = "/" + entry.folder;
    return projectFolder.endsWith(suffix) ? projectFolder : projectFolder + suffix;
  }, [projectFolder, entry]);

  if (loading) {
    return (
      <div style={{ display: "flex", justifyContent: "center", marginTop: 80 }}>
        <Spinner />
      </div>
    );
  }

  if (error) {
    return (
      <NonIdealState
        icon="error"
        title="Could not load manifest.json"
        description={error}
      />
    );
  }

  if (!entry) {
    return (
      <NonIdealState
        icon="search"
        title="Run not found"
        description={`No run with folder "${folderId}" found in the manifest.`}
      />
    );
  }

  // Determine slider max from selected volume and slice type
  const sliderMax = selectedVolume
    ? sliceType === "inline"
      ? selectedVolume.shape[0] - 1
      : sliceType === "crossline"
      ? selectedVolume.shape[1] - 1
      : selectedVolume.shape[2] - 1
    : 0;

  // Third-tab label
  const thirdTabLabel =
    selectedVolume?.dims[2] === "horizon" ? "Horizon slice" : "Timeslice";

  return (
    <div style={{ padding: 24 }}>
      {/* Back navigation */}
      <Button
        icon="arrow-left"
        minimal
        onClick={() => {
          window.location.hash = "#/";
        }}
        style={{ marginBottom: 16 }}
      >
        All Runs
      </Button>

      <h2 className="bp5-heading" style={{ marginBottom: 4 }}>
        {entry.run_id}
      </h2>
      <p style={{ color: "#8a9ba8", marginBottom: 16 }}>
        {entry.folder} &nbsp;·&nbsp; {entry.datestamp}
      </p>

      {/* Controls row */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 16,
          alignItems: "flex-end",
          marginBottom: 16,
        }}
      >
        <div>
          <label className="bp5-label" style={{ marginBottom: 4 }}>
            Volume
          </label>
          <VolumeSelector
            volumes={entry.volumes}
            selected={selectedVolume}
            onChange={handleVolumeChange}
          />
        </div>
        <div>
          <label className="bp5-label" style={{ marginBottom: 4 }}>
            Colormap
          </label>
          <ColormapSelector
            value={colormap}
            onChange={handleColormapChange}
            reversed={colormapReversed}
            onReverseChange={handleReversedChange}
          />
        </div>
      </div>

      {/* Slice-type tabs */}
      <div
        role="tablist"
        style={{ display: "flex", gap: 0, marginBottom: 12, borderBottom: "1px solid #394b59" }}
      >
        {(["inline", "crossline", "timeslice"] as const).map((st) => {
          const label =
            st === "timeslice" ? thirdTabLabel : st.charAt(0).toUpperCase() + st.slice(1);
          return (
            <button
              key={st}
              role="tab"
              aria-selected={sliceType === st}
              onClick={() => setSliceType(st)}
              style={{
                background: "none",
                border: "none",
                color: sliceType === st ? "#8abbff" : "#8a9ba8",
                borderBottom: sliceType === st ? "2px solid #8abbff" : "2px solid transparent",
                padding: "8px 16px",
                cursor: "pointer",
                fontFamily: "inherit",
                fontSize: 14,
              }}
            >
              {label}
            </button>
          );
        })}
      </div>

      {/* Slice index slider */}
      {selectedVolume && (
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
          <label className="bp5-label" style={{ minWidth: 60 }}>
            Index: {sliceIndex}
          </label>
          <input
            type="range"
            min={0}
            max={sliderMax}
            value={sliceType === "crossline" ? crosslineDraft : sliceIndex}
            // Crossline: update draft on every change (smooth thumb) but only
            // commit to sliceIndex (triggering zarr fetch) on mouseUp/touchEnd.
            onChange={(e) =>
              sliceType === "crossline"
                ? setCrosslineDraft(Number(e.target.value))
                : setSliceIndex(Number(e.target.value))
            }
            onMouseUp={(e) =>
              sliceType === "crossline" &&
              setSliceIndex(Number((e.target as HTMLInputElement).value))
            }
            onTouchEnd={(e) =>
              sliceType === "crossline" &&
              setSliceIndex(Number((e.target as HTMLInputElement).value))
            }
            style={{ flex: 1 }}
          />
          <span style={{ minWidth: 40, textAlign: "right", fontSize: 12, color: "#8a9ba8" }}>
            / {sliderMax}
          </span>
        </div>
      )}

      {/* Main content area */}
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
        <div style={{ flex: "1 1 600px", minWidth: 300 }}>
          {selectedVolume ? (
            <SliceViewer
              folderPath={runFolderPath}
              volume={selectedVolume}
              sliceType={sliceType}
              sliceIndex={sliceIndex}
              colormap={colormap}
              reversed={colormapReversed}
            />
          ) : (
            <NonIdealState icon="layers" title="Select a volume" />
          )}
        </div>
        <div style={{ flex: "0 0 360px", minWidth: 280 }}>
          <MetadataPanel entry={entry} selectedVolume={selectedVolume} />
        </div>
      </div>
    </div>
  );
}
