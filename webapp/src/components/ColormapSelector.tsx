import React from "react";
import { Checkbox, HTMLSelect } from "@blueprintjs/core";

// ---------------------------------------------------------------------------
// Colormaps from https://github.com/lperozzi/Seismic_colormaps
// ---------------------------------------------------------------------------

/** yrwbc — Yellow → Red → White → Blue → Cyan */
const YRWBC_CS: [number, string][] = [
  [0.0, "rgb(255,255,0)"],
  [0.032, "rgb(255,204,0)"],
  [0.065, "rgb(255,153,0)"],
  [0.097, "rgb(255,102,0)"],
  [0.129, "rgb(255,51,0)"],
  [0.161, "rgb(255,1,1)"],
  [0.194, "rgb(255,51,51)"],
  [0.226, "rgb(255,102,102)"],
  [0.258, "rgb(255,153,153)"],
  [0.290, "rgb(255,204,204)"],
  [0.323, "rgb(255,255,255)"],
  [0.355, "rgb(245,245,255)"],
  [0.387, "rgb(224,224,255)"],
  [0.419, "rgb(204,204,255)"],
  [0.452, "rgb(163,163,255)"],
  [0.484, "rgb(122,122,255)"],
  [0.516, "rgb(82,82,255)"],
  [0.548, "rgb(41,41,255)"],
  [0.581, "rgb(0,0,255)"],
  [0.613, "rgb(0,26,255)"],
  [0.645, "rgb(0,51,255)"],
  [0.677, "rgb(0,89,255)"],
  [0.710, "rgb(0,128,255)"],
  [0.742, "rgb(0,166,255)"],
  [0.774, "rgb(0,204,255)"],
  [0.806, "rgb(0,230,255)"],
  [0.839, "rgb(0,255,255)"],
  [1.0,   "rgb(0,255,255)"],
];

/** sharp — Yellow → warm white → pale blue → deep blue → cyan */
const SHARP_CS: [number, string][] = [
  [0.0,   "rgb(255,255,0)"],
  [0.032, "rgb(255,191,0)"],
  [0.065, "rgb(255,128,0)"],
  [0.097, "rgb(255,64,0)"],
  [0.129, "rgb(255,128,3)"],
  [0.161, "rgb(255,166,64)"],
  [0.194, "rgb(255,204,128)"],
  [0.226, "rgb(255,230,191)"],
  [0.258, "rgb(255,255,255)"],
  [0.290, "rgb(245,245,255)"],
  [0.323, "rgb(224,224,255)"],
  [0.355, "rgb(204,204,255)"],
  [0.387, "rgb(173,173,255)"],
  [0.419, "rgb(143,143,255)"],
  [0.452, "rgb(112,112,255)"],
  [0.484, "rgb(82,82,255)"],
  [0.516, "rgb(51,51,255)"],
  [0.548, "rgb(20,20,255)"],
  [0.581, "rgb(0,10,255)"],
  [0.613, "rgb(0,51,255)"],
  [0.645, "rgb(0,102,255)"],
  [0.677, "rgb(0,153,255)"],
  [0.710, "rgb(0,204,255)"],
  [1.0,   "rgb(0,255,255)"],
];

/** seismics (ODT) — dark red → orange → pale → grey-blue → black */
const SEISMICS_CS: [number, string][] = [
  [0.0,   "rgb(170,0,0)"],
  [0.032, "rgb(184,10,0)"],
  [0.065, "rgb(199,26,0)"],
  [0.097, "rgb(214,46,0)"],
  [0.129, "rgb(224,64,0)"],
  [0.161, "rgb(235,85,0)"],
  [0.194, "rgb(242,102,0)"],
  [0.226, "rgb(250,128,0)"],
  [0.258, "rgb(255,153,0)"],
  [0.290, "rgb(255,178,0)"],
  [0.323, "rgb(255,199,0)"],
  [0.355, "rgb(255,207,61)"],
  [0.387, "rgb(254,212,87)"],
  [0.419, "rgb(251,220,135)"],
  [0.452, "rgb(248,230,181)"],
  [0.484, "rgb(246,237,212)"],
  [0.516, "rgb(240,240,238)"],
  [0.548, "rgb(236,236,239)"],
  [0.565, "rgb(232,232,236)"],
  [0.581, "rgb(219,220,229)"],
  [0.613, "rgb(200,201,216)"],
  [0.645, "rgb(181,182,201)"],
  [0.677, "rgb(161,163,189)"],
  [0.710, "rgb(140,143,172)"],
  [0.742, "rgb(119,122,155)"],
  [0.774, "rgb(100,104,138)"],
  [0.806, "rgb(80,85,119)"],
  [0.839, "rgb(60,65,99)"],
  [0.871, "rgb(40,43,77)"],
  [0.903, "rgb(25,26,55)"],
  [0.935, "rgb(12,12,30)"],
  [1.0,   "rgb(0,0,0)"],
];

/** petrel (ODT) — cyan → blue-grey → warm grey → brown → olive-yellow */
const PETREL_CS: [number, string][] = [
  [0.0,   "rgb(161,255,255)"],
  [0.032, "rgb(152,233,251)"],
  [0.065, "rgb(143,213,245)"],
  [0.097, "rgb(135,194,240)"],
  [0.129, "rgb(127,177,234)"],
  [0.161, "rgb(119,161,228)"],
  [0.194, "rgb(112,146,221)"],
  [0.226, "rgb(104,131,212)"],
  [0.258, "rgb(104,120,207)"],
  [0.290, "rgb(103,110,200)"],
  [0.323, "rgb(104,96,183)"],
  [0.355, "rgb(105,92,175)"],
  [0.387, "rgb(107,95,165)"],
  [0.419, "rgb(110,95,159)"],
  [0.452, "rgb(116,99,150)"],
  [0.484, "rgb(126,108,148)"],
  [0.516, "rgb(143,132,157)"],
  [0.548, "rgb(162,153,164)"],
  [0.565, "rgb(182,176,178)"],
  [0.581, "rgb(198,193,192)"],
  [0.613, "rgb(190,178,172)"],
  [0.645, "rgb(182,161,153)"],
  [0.677, "rgb(172,142,128)"],
  [0.710, "rgb(162,117,99)"],
  [0.742, "rgb(162,93,68)"],
  [0.774, "rgb(162,84,54)"],
  [0.806, "rgb(163,79,47)"],
  [0.839, "rgb(172,100,31)"],
  [0.871, "rgb(186,112,16)"],
  [0.903, "rgb(209,147,2)"],
  [0.935, "rgb(235,197,0)"],
  [1.0,   "rgb(255,255,0)"],
];

// ---------------------------------------------------------------------------
// Map name → Plotly colorscale
// ---------------------------------------------------------------------------

export const COLORSCALE_MAP: Record<string, string | [number, string][]> = {
  Seismic:       "RdBu",
  "Red-White-Blue": "RdBu",
  Greyscale:     "Greys",
  yrwbc:         YRWBC_CS,
  sharp:         SHARP_CS,
  seismics:      SEISMICS_CS,
  petrel:        PETREL_CS,
};

/**
 * Resolve a colormap key to the Plotly colorscale value.
 * When `reversed` is true the array is flipped end-to-end (or the named
 * string becomes the reversed variant). This ensures Plotly always sees a
 * changed `colorscale` value and re-renders immediately.
 */
export function resolveColorscale(
  key: string,
  reversed: boolean,
): string | [number, string][] {
  const cs = COLORSCALE_MAP[key] ?? key;
  if (!reversed) return cs;

  if (typeof cs === "string") {
    // Plotly named scales support the _r suffix for reversal.
    return cs.endsWith("_r") ? cs.slice(0, -2) : `${cs}_r`;
  }
  // Custom array: reverse and remap positions 0→1.
  const rev = [...cs].reverse();
  return rev.map(([, color], i) => [i / (rev.length - 1), color] as [number, string]);
}

export function colormapForGroup(_group: string): string {
  // Default for all groups — users choose their preferred colormap.
  return "seismics";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const COLORMAPS: { label: string; value: string }[] = [
  { label: "Seismic (RdBu)",    value: "Seismic"        },
  { label: "Red-White-Blue",    value: "Red-White-Blue" },
  { label: "Greyscale",         value: "Greyscale"      },
  { label: "yrwbc",             value: "yrwbc"          },
  { label: "sharp",             value: "sharp"          },
  { label: "seismics (ODT)",    value: "seismics"       },
  { label: "petrel (ODT)",      value: "petrel"         },
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
