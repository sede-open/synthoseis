import React from "react";
import { Checkbox, HTMLSelect } from "@blueprintjs/core";

// ---------------------------------------------------------------------------
// Custom colorscale definitions
// ---------------------------------------------------------------------------

/** Classic seismic diverging: dark-blue → white → dark-red */
const SEISMIC_CS: [number, string][] = [
  [0.0,   "rgb(0,0,180)"],
  [0.2,   "rgb(0,80,255)"],
  [0.35,  "rgb(100,180,255)"],
  [0.5,   "rgb(255,255,255)"],
  [0.65,  "rgb(255,160,80)"],
  [0.8,   "rgb(255,40,0)"],
  [1.0,   "rgb(160,0,0)"],
];

/** Petrel-style rainbow: blue → cyan → green → yellow → red */
const PETREL_CS: [number, string][] = [
  [0.0,   "rgb(0,0,160)"],
  [0.15,  "rgb(0,100,255)"],
  [0.3,   "rgb(0,220,255)"],
  [0.45,  "rgb(0,255,160)"],
  [0.55,  "rgb(200,255,0)"],
  [0.7,   "rgb(255,200,0)"],
  [0.85,  "rgb(255,60,0)"],
  [1.0,   "rgb(140,0,0)"],
];

/** Map from selector value → Plotly colorscale (named string or custom array) */
export const COLORSCALE_MAP: Record<string, string | [number, string][]> = {
  Seismic: SEISMIC_CS,
  RdBu:    "RdBu",
  Greys:   "Greys",
  Petrel:  PETREL_CS,
};

/**
 * Resolve a colormap key to the value Plotly expects for `colorscale`.
 * If `reversed` is true the array is flipped (or "_r" appended for named scales
 * that support it — Plotly also accepts `reversescale` on the trace, which is
 * what SliceViewer uses, so this helper is for reference only).
 */
export function resolveColorscale(
  key: string,
): string | [number, string][] {
  return COLORSCALE_MAP[key] ?? key;
}

// ---------------------------------------------------------------------------
// Default colormap per volume group
// ---------------------------------------------------------------------------

export function colormapForGroup(group: string): string {
  switch (group) {
    case "Seismic": return "Seismic";
    case "Geology": return "Petrel";
    case "Closures":
    case "Horizons":
    case "QC":
    default:        return "Greys";
  }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const COLORMAPS: { label: string; value: string }[] = [
  { label: "Seismic",       value: "Seismic" },
  { label: "Red-White-Blue", value: "RdBu"   },
  { label: "Greyscale",     value: "Greys"   },
  { label: "Petrel",        value: "Petrel"  },
];

interface ColormapSelectorProps {
  value: string;
  onChange: (value: string) => void;
  reversed: boolean;
  onReverseChange: (reversed: boolean) => void;
}

export default function ColormapSelector({
  value,
  onChange,
  reversed,
  onReverseChange,
}: ColormapSelectorProps): React.ReactElement {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <HTMLSelect
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {COLORMAPS.map((cm) => (
          <option key={cm.value} value={cm.value}>
            {cm.label}
          </option>
        ))}
      </HTMLSelect>
      <Checkbox
        checked={reversed}
        onChange={(e) => onReverseChange((e.target as HTMLInputElement).checked)}
        label="Reverse"
        style={{ marginBottom: 0 }}
      />
    </div>
  );
}
