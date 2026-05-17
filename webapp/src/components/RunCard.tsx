import React from "react";
import { Card, Tag, Intent } from "@blueprintjs/core";
import type { ManifestEntry } from "../types/manifest";

interface RunCardProps {
  entry: ManifestEntry;
}

export default function RunCard({ entry }: RunCardProps): React.ReactElement {
  const handleClick = () => {
    window.location.hash = `#/run/${encodeURIComponent(entry.folder)}`;
  };

  // Parse incident_angles and closure_types from parameters if present
  let incidentAngles: string = "—";
  let closureTypes: string = "—";
  try {
    if (entry.parameters.incident_angles) {
      incidentAngles = JSON.parse(entry.parameters.incident_angles).join(", ");
    }
  } catch {
    incidentAngles = entry.parameters.incident_angles ?? "—";
  }
  try {
    if (entry.parameters.closure_types) {
      closureTypes = JSON.parse(entry.parameters.closure_types).join(", ");
    }
  } catch {
    closureTypes = entry.parameters.closure_types ?? "—";
  }

  return (
    <Card
      interactive
      onClick={handleClick}
      style={{ cursor: "pointer" }}
    >
      <h5 className="bp5-heading" style={{ marginBottom: 8 }}>
        {entry.run_id}
      </h5>
      <p style={{ marginBottom: 4, fontSize: 12, color: "#8a9ba8" }}>
        {entry.folder}
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: 8 }}>
        <Tag intent={Intent.PRIMARY} minimal>
          {entry.cube_shape.join(" × ")}
        </Tag>
        <Tag intent={Intent.NONE} minimal>
          {entry.datestamp}
        </Tag>
        <Tag intent={Intent.SUCCESS} minimal>
          {entry.volumes.length} volume{entry.volumes.length !== 1 ? "s" : ""}
        </Tag>
      </div>
      {incidentAngles !== "—" && (
        <p style={{ fontSize: 12, margin: "4px 0" }}>
          <strong>Angles:</strong> {incidentAngles}°
        </p>
      )}
      {closureTypes !== "—" && (
        <p style={{ fontSize: 12, margin: "4px 0" }}>
          <strong>Closures:</strong> {closureTypes}
        </p>
      )}
    </Card>
  );
}
