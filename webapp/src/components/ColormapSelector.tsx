import React from "react";
import { Button, Checkbox, MenuDivider, MenuItem } from "@blueprintjs/core";
import { Select } from "@blueprintjs/select";

// ---------------------------------------------------------------------------
// Colorscale definitions
// ---------------------------------------------------------------------------

/** RdBu with a true symmetric white centre (blue→white→red) */
const RDBU_CS: [number, string][] = [
  [0.0,   "rgb(5,48,97)"],
  [0.125, "rgb(33,102,172)"],
  [0.25,  "rgb(92,164,220)"],
  [0.375, "rgb(175,213,240)"],
  [0.5,   "rgb(255,255,255)"],
  [0.625, "rgb(245,189,152)"],
  [0.75,  "rgb(214,96,77)"],
  [0.875, "rgb(178,24,43)"],
  [1.0,   "rgb(103,0,31)"],
];

/**
 * seismics (ODT) — dark red → orange → pale → grey-blue → black.
 * Stop at 0.500 is pure white, pinned exactly at the zero-amplitude position
 * so that symmetric zmin/zmax always renders zero as white.
 */
const SEISMICS_CS: [number, string][] = [
  [0.000, "rgb(170,0,0)"],
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
  [0.500, "rgb(255,255,255)"],  // zero = pure white (was split across 0.484/0.516)
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
  [1.000, "rgb(0,0,0)"],
];

/** Gray — black → mid-grey → white.  Reverse for white-to-black. */
const GRAY_CS: [number, string][] = [
  [0.0, "rgb(0,0,0)"],
  [0.5, "rgb(128,128,128)"],
  [1.0, "rgb(255,255,255)"],
];

/**
 * Viridis — explicit stops so heatmapgl (WebGL) renders it correctly.
 * Named-string colorscales are not reliably honoured by heatmapgl.
 * Values sampled from the canonical matplotlib Viridis LUT.
 */
const VIRIDIS_CS: [number, string][] = [
  [0.0,  "rgb(68,1,84)"],
  [0.1,  "rgb(72,29,111)"],
  [0.2,  "rgb(64,68,135)"],
  [0.3,  "rgb(52,95,141)"],
  [0.4,  "rgb(41,121,142)"],
  [0.5,  "rgb(32,146,140)"],
  [0.6,  "rgb(34,168,132)"],
  [0.7,  "rgb(70,190,112)"],
  [0.8,  "rgb(122,209,81)"],
  [0.9,  "rgb(190,222,44)"],
  [1.0,  "rgb(253,231,37)"],
];

/**
 * Magma — explicit stops for the same reason as Viridis.
 * Values sampled from the canonical matplotlib Magma LUT.
 */
const MAGMA_CS: [number, string][] = [
  [0.0,  "rgb(0,0,4)"],
  [0.1,  "rgb(21,11,53)"],
  [0.2,  "rgb(59,15,112)"],
  [0.3,  "rgb(99,26,128)"],
  [0.4,  "rgb(140,41,129)"],
  [0.5,  "rgb(183,55,121)"],
  [0.6,  "rgb(221,81,100)"],
  [0.7,  "rgb(248,125,81)"],
  [0.8,  "rgb(254,174,97)"],
  [0.9,  "rgb(254,220,135)"],
  [1.0,  "rgb(252,253,191)"],
];

export const COLORSCALE_MAP: Record<string, [number, string][]> = {
  RdBu:     RDBU_CS,
  seismics: SEISMICS_CS,
  gray:     GRAY_CS,
  Viridis:  VIRIDIS_CS,
  Magma:    MAGMA_CS,
};

/**
 * Resolve a colormap key to the Plotly colorscale value.
 * Reversal is baked in so that Plotly receives a new `colorscale` reference
 * and re-renders immediately (heatmapgl ignores `reversescale` updates).
 * All entries are explicit arrays — named strings are intentionally avoided
 * because heatmapgl (WebGL) does not reliably honour them.
 */
export function resolveColorscale(
  key: string,
  reversed: boolean,
): [number, string][] {
  const cs = COLORSCALE_MAP[key] ?? COLORSCALE_MAP["seismics"];
  if (!reversed) return cs;

  const rev = [...cs].reverse();
  return rev.map(([, color], i) => [
    i / (rev.length - 1),
    color,
  ] as [number, string]);
}

export function colormapForGroup(_group: string): string {
  return "seismics";
}

// ---------------------------------------------------------------------------
// Select item types
// ---------------------------------------------------------------------------

interface ColormapOption {
  value: string;
  label: string;
  category: string;
  isHeader?: false;
}

interface CategoryHeader {
  value: string;  // used as key — must be unique
  label: string;
  category: string;
  isHeader: true;
}

type SelectItem = ColormapOption | CategoryHeader;

const ALL_ITEMS: SelectItem[] = [
  { value: "__div__",     label: "Diverging",     category: "Diverging",  isHeader: true },
  { value: "RdBu",        label: "RdBu",          category: "Diverging"  },
  { value: "__seis__",    label: "Seismic (ODT)", category: "Seismic",    isHeader: true },
  { value: "seismics",    label: "seismics",      category: "Seismic"    },
  { value: "gray",        label: "Gray",          category: "Seismic"    },
  { value: "__seq__",     label: "Sequential",    category: "Sequential", isHeader: true },
  { value: "Viridis",     label: "Viridis",       category: "Sequential" },
  { value: "Magma",       label: "Magma",         category: "Sequential" },
];

const OPTIONS: ColormapOption[] = ALL_ITEMS.filter(
  (i): i is ColormapOption => !i.isHeader,
);

function labelForValue(value: string): string {
  return OPTIONS.find((o) => o.value === value)?.label ?? value;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

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
      <Select<SelectItem>
        items={ALL_ITEMS}
        filterable={false}
        itemDisabled={(item) => !!item.isHeader}
        itemRenderer={(item, { handleClick, modifiers }) => {
          if (item.isHeader) {
            return (
              <MenuDivider
                key={item.value}
                title={item.label}
              />
            );
          }
          return (
            <MenuItem
              key={item.value}
              text={item.label}
              active={modifiers.active}
              disabled={modifiers.disabled}
              selected={item.value === value}
              onClick={handleClick}
            />
          );
        }}
        onItemSelect={(item) => {
          if (!item.isHeader) onChange(item.value);
        }}
        popoverProps={{ minimal: true, placement: "bottom-start" }}
      >
        <Button
          text={labelForValue(value)}
          rightIcon="chevron-down"
          minimal
        />
      </Select>

      <Checkbox
        checked={reversed}
        onChange={(e) => onReverseChange((e.target as HTMLInputElement).checked)}
        label="Reverse"
        style={{ marginBottom: 0 }}
      />
    </div>
  );
}
