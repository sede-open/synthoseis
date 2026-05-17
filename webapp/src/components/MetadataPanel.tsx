import React from "react";
import { Tabs, Tab, HTMLTable } from "@blueprintjs/core";
import type { ManifestEntry, VolumeInfo } from "../types/manifest";

interface MetadataPanelProps {
  entry: ManifestEntry;
  selectedVolume: VolumeInfo | null;
}

export default function MetadataPanel({
  entry,
  selectedVolume,
}: MetadataPanelProps): React.ReactElement {
  return (
    <Tabs id="metadata-tabs" animate>
      <Tab
        id="parameters"
        title="Parameters"
        panel={<ParametersTab parameters={entry.parameters} />}
      />
      <Tab
        id="volumes"
        title="Volumes"
        panel={<VolumesTab volumes={entry.volumes} selected={selectedVolume} />}
      />
    </Tabs>
  );
}

function ParametersTab({
  parameters,
}: {
  parameters: Record<string, string>;
}): React.ReactElement {
  const entries = Object.entries(parameters);

  if (entries.length === 0) {
    return <p style={{ color: "#8a9ba8", fontStyle: "italic" }}>No parameters available.</p>;
  }

  return (
    <HTMLTable compact striped style={{ width: "100%", fontSize: 12 }}>
      <thead>
        <tr>
          <th>Key</th>
          <th>Value</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k}>
            <td>
              <code>{k}</code>
            </td>
            <td style={{ wordBreak: "break-word" }}>{v}</td>
          </tr>
        ))}
      </tbody>
    </HTMLTable>
  );
}

function VolumesTab({
  volumes,
  selected,
}: {
  volumes: VolumeInfo[];
  selected: VolumeInfo | null;
}): React.ReactElement {
  return (
    <div style={{ maxHeight: 480, overflowY: "auto" }}>
      {volumes.map((vol) => {
        const isSelected = selected?.store_path === vol.store_path;
        return (
          <div
            key={vol.store_path}
            style={{
              marginBottom: 12,
              padding: 8,
              borderRadius: 4,
              border: isSelected ? "1px solid #8abbff" : "1px solid #394b59",
              fontSize: 12,
            }}
          >
            <strong>{vol.name}</strong>
            <br />
            <span style={{ color: "#8a9ba8" }}>{vol.group}</span>
            <HTMLTable compact style={{ width: "100%", marginTop: 4, fontSize: 11 }}>
              <tbody>
                <tr>
                  <td>Shape</td>
                  <td>{vol.shape.join(" × ")}</td>
                </tr>
                <tr>
                  <td>Dtype</td>
                  <td>{vol.dtype}</td>
                </tr>
                <tr>
                  <td>Dims</td>
                  <td>{vol.dims.join(", ")}</td>
                </tr>
                <tr>
                  <td>Chunks</td>
                  <td>{vol.chunks.join(" × ")}</td>
                </tr>
                <tr>
                  <td>Compressor</td>
                  <td>{vol.compressor}</td>
                </tr>
                {Object.keys(vol.attrs).length > 0 && (
                  <tr>
                    <td>Attrs</td>
                    <td>{JSON.stringify(vol.attrs)}</td>
                  </tr>
                )}
              </tbody>
            </HTMLTable>
          </div>
        );
      })}
    </div>
  );
}
