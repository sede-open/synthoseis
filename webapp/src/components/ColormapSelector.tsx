import React from "react";
import { HTMLSelect } from "@blueprintjs/core";

/**
 * Returns the default colormap for a given volume group.
 */
export function colormapForGroup(group: string): string {
  switch (group) {
    case "Seismic":
      return "RdBu";
    case "Geology":
      return "Greys";
    case "Closures":
      return "Hot";
    case "Horizons":
    case "QC":
    default:
      return "Viridis";
  }
}

const COLORMAPS = [
  "RdBu",
  "Greys",
  "Hot",
  "Viridis",
  "Plasma",
  "Inferno",
  "Magma",
  "Cividis",
  "Jet",
  "Turbo",
  "YlOrRd",
  "Blues",
  "Greens",
];

interface ColormapSelectorProps {
  value: string;
  onChange: (value: string) => void;
}

export default function ColormapSelector({
  value,
  onChange,
}: ColormapSelectorProps): React.ReactElement {
  return (
    <HTMLSelect
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      {COLORMAPS.map((cm) => (
        <option key={cm} value={cm}>
          {cm}
        </option>
      ))}
    </HTMLSelect>
  );
}
