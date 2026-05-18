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

export const COLORSCALE_MAP: Record<string, string | [number, string][]> = {
  RdBu:     RDBU_CS,
  seismics: SEISMICS_CS,
  petrel:   PETREL_CS,
  Viridis:  "Viridis",
  Magma:    "Magma",
};

/**
 * Resolve a colormap key to the Plotly colorscale value.
 * Reversal is baked in so that Plotly receives a new `colorscale` reference
 * and re-renders immediately (heatmapgl ignores `reversescale` updates).
 */
export function resolveColorscale(
  key: string,
  reversed: boolean,
): string | [number, string][] {
  const cs = COLORSCALE_MAP[key] ?? key;
  if (!reversed) return cs;

  if (typeof cs === "string") {
    return cs.endsWith("_r") ? cs.slice(0, -2) : `${cs}_r`;
  }
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
  { value: "__div__",      label: "Diverging",      category: "Diverging",   isHeader: true },
  { value: "RdBu",         label: "RdBu",           category: "Diverging"  },
  { value: "__seismic__",  label: "Seismic (ODT)",  category: "Seismic",     isHeader: true },
  { value: "seismics",     label: "seismics",       category: "Seismic"    },
  { value: "petrel",       label: "petrel",         category: "Seismic"    },
  { value: "__seq__",      label: "Sequential",     category: "Sequential",  isHeader: true },
  { value: "Viridis",      label: "Viridis",        category: "Sequential" },
  { value: "Magma",        label: "Magma",          category: "Sequential" },
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
