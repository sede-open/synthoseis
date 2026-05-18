import React from "react";
import { Select, ItemRenderer, ItemPredicate } from "@blueprintjs/select";
import { Button, MenuItem } from "@blueprintjs/core";
import type { VolumeInfo } from "../types/manifest";

/** Group display order */
const GROUP_ORDER = ["Seismic", "Geology", "Closures", "Horizons", "QC"];

interface VolumeSelectorProps {
  volumes: VolumeInfo[];
  selected: VolumeInfo | null;
  onChange: (vol: VolumeInfo) => void;
}

const filterVolume: ItemPredicate<VolumeInfo> = (query, vol) =>
  vol.name.toLowerCase().includes(query.toLowerCase()) ||
  vol.group.toLowerCase().includes(query.toLowerCase());

const renderVolume: ItemRenderer<VolumeInfo> = (vol, { handleClick, modifiers }) => {
  if (!modifiers.matchesPredicate) return null;

  // Annotate seismic items with angle_deg if available
  const angleSuffix =
    vol.group === "Seismic" && vol.attrs.angle_deg !== undefined
      ? ` (${vol.attrs.angle_deg}°)`
      : "";

  return (
    <MenuItem
      key={vol.store_path}
      text={vol.name + angleSuffix}
      label={vol.group}
      active={modifiers.active}
      disabled={modifiers.disabled}
      onClick={handleClick}
    />
  );
};

export default function VolumeSelector({
  volumes,
  selected,
  onChange,
}: VolumeSelectorProps): React.ReactElement {
  // Group volumes for display
  const grouped: VolumeInfo[] = React.useMemo(() => {
    const byGroup: Record<string, VolumeInfo[]> = {};
    for (const vol of volumes) {
      if (!byGroup[vol.group]) byGroup[vol.group] = [];
      byGroup[vol.group].push(vol);
    }
    const result: VolumeInfo[] = [];
    for (const g of GROUP_ORDER) {
      if (byGroup[g]) result.push(...byGroup[g]);
    }
    // Any remaining groups not in GROUP_ORDER
    for (const [g, vols] of Object.entries(byGroup)) {
      if (!GROUP_ORDER.includes(g)) result.push(...vols);
    }
    return result;
  }, [volumes]);

  const buttonText = selected
    ? selected.name +
      (selected.group === "Seismic" && selected.attrs.angle_deg !== undefined
        ? ` (${selected.attrs.angle_deg}°)`
        : "")
    : "Select volume…";

  return (
    <Select<VolumeInfo>
      items={grouped}
      itemRenderer={renderVolume}
      itemPredicate={filterVolume}
      onItemSelect={onChange}
      popoverProps={{ minimal: true }}
    >
      <Button text={buttonText} rightIcon="caret-down" />
    </Select>
  );
}
