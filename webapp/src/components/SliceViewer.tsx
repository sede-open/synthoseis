import React from "react";
import { NonIdealState, Spinner } from "@blueprintjs/core";
import Plot from "react-plotly.js";
import type { VolumeInfo } from "../types/manifest";
import useZarrSlice from "../hooks/useZarrSlice";

interface SliceViewerProps {
  folderPath: string;
  volume: VolumeInfo;
  sliceType: "inline" | "crossline" | "timeslice";
  sliceIndex: number;
  colormap: string;
}

export default function SliceViewer({
  folderPath,
  volume,
  sliceType,
  sliceIndex,
  colormap,
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

  // Build 2-D z array for plotly (rows × cols)
  const [rows, cols] = shape;
  const z: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row: number[] = [];
    for (let c = 0; c < cols; c++) {
      row.push(data[r * cols + c]);
    }
    z.push(row);
  }

  return (
    <Plot
      data={[
        {
          type: "heatmapgl",
          z,
          colorscale: colormap,
          showscale: true,
          hoverinfo: "z",
        } as Plotly.Data,
      ]}
      layout={{
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#f5f8fa" },
        margin: { t: 24, l: 48, r: 24, b: 48 },
        xaxis: { color: "#8a9ba8", title: { text: _xLabel(sliceType, volume) } },
        yaxis: {
          color: "#8a9ba8",
          title: { text: _yLabel(sliceType, volume) },
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
    case "inline":
      return dims[1] ?? "crossline";
    case "crossline":
      return dims[0] ?? "inline";
    case "timeslice":
    default:
      return dims[1] ?? "crossline";
  }
}

function _yLabel(sliceType: string, volume: VolumeInfo): string {
  const dims = volume.dims;
  switch (sliceType) {
    case "inline":
      return dims[2] ?? "time";
    case "crossline":
      return dims[2] ?? "time";
    case "timeslice":
    default:
      return dims[0] ?? "inline";
  }
}
