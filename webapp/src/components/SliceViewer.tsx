import React from "react";
import { NonIdealState, Spinner } from "@blueprintjs/core";
import createPlotlyComponent from "react-plotly.js/factory";
// plotly.js-dist-min is externalised (loaded from CDN in index.html).
// The factory pattern binds react-plotly.js to the window.Plotly global.
declare const Plotly: Parameters<typeof createPlotlyComponent>[0];
const Plot = createPlotlyComponent(Plotly);
import type { VolumeInfo } from "../types/manifest";
import useZarrSlice from "../hooks/useZarrSlice";
import { resolveColorscale } from "./ColormapSelector";

interface SliceViewerProps {
  folderPath: string;
  volume: VolumeInfo;
  sliceType: "inline" | "crossline" | "timeslice";
  sliceIndex: number;
  colormap: string;
  reversed: boolean;
}

export default function SliceViewer({
  folderPath,
  volume,
  sliceType,
  sliceIndex,
  colormap,
  reversed,
}: SliceViewerProps): React.ReactElement {
  const { data, shape, loading, error } = useZarrSlice(
    folderPath,
    volume.store_path,
    volume.variable,
    sliceType,
    sliceIndex
  );

  if (loading) {
    return (
      <div style={{ display: "flex", justifyContent: "center", alignItems: "center", minHeight: 400 }}>
        <Spinner />
      </div>
    );
  }

  if (error || !data || !shape) {
    return (
      <NonIdealState
        icon="error"
        title="Could not load volume chunk"
        description={error ?? "Unknown error loading zarr slice."}
      />
    );
  }

  // Transpose: zarrita returns [axis0, axis1] but we want the depth/time axis
  // on the y-axis (vertical) and the spatial axis on x.
  // Original shape: [rows, cols] = e.g. [crosslines, time] for an inline slice.
  // After transpose: z[col][row] so time runs along rows (y) and space along cols (x).
  const [rows, cols] = shape;
  const z: number[][] = [];
  for (let c = 0; c < cols; c++) {
    const row: number[] = [];
    for (let r = 0; r < rows; r++) {
      row.push(data[r * cols + c]);
    }
    z.push(row);
  }

  // Symmetric colour range: find the largest absolute value in the slice so
  // that 0 always maps to the centre of the colormap (standard seismic display).
  // isFinite guard skips any NaN/Infinity samples that would corrupt the range.
  let absMax = 0;
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i]);
    if (isFinite(a) && a > absMax) absMax = a;
  }
  // Guard against an all-zero slice — fall back to ±1 so the plot is still visible.
  if (absMax === 0) absMax = 1;

  // y-axis autorange "reversed" puts index 0 at the top so depth/time
  // values increase downward, matching standard seismic display convention.
  const isTimeslice = sliceType === "timeslice";

  return (
    <Plot
      data={[
        {
          type: "heatmapgl",
          z,
          colorscale: resolveColorscale(colormap, reversed),
          zmin: -absMax,
          zmax: absMax,
          showscale: true,
          hoverinfo: "z",
        } as Plotly.Data,
      ]}
      layout={{
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#f5f8fa" },
        margin: { t: 24, l: 48, r: 24, b: 48 },
        xaxis: {
          color: "#8a9ba8",
          title: { text: _xLabel(sliceType, volume) },
        },
        yaxis: {
          color: "#8a9ba8",
          title: { text: _yLabel(sliceType, volume) },
          // For inline/crossline the y-axis is depth/time — increase downward.
          // For timeslices the y-axis is a spatial dimension — normal orientation.
          autorange: isTimeslice ? true : "reversed",
          scaleanchor: undefined,
        },
        autosize: true,
      }}
      useResizeHandler
      style={{ width: "100%", minHeight: 400 }}
      config={{ responsive: true, displayModeBar: true }}
    />
  );
}

function _xLabel(sliceType: string, volume: VolumeInfo): string {
  const dims = volume.dims;
  switch (sliceType) {
    case "inline":    return dims[1] ?? "crossline";
    case "crossline": return dims[0] ?? "inline";
    case "timeslice":
    default:          return dims[1] ?? "crossline";
  }
}

function _yLabel(sliceType: string, volume: VolumeInfo): string {
  const dims = volume.dims;
  switch (sliceType) {
    case "inline":    return dims[2] ?? "time";
    case "crossline": return dims[2] ?? "time";
    case "timeslice":
    default:          return dims[0] ?? "inline";
  }
}
