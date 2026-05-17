import React from "react";
import { NonIdealState, Spinner } from "@blueprintjs/core";
import useManifest from "../hooks/useManifest";
import RunCard from "./RunCard";

export default function ProjectDashboard(): React.ReactElement {
  const { data: manifest, loading, error } = useManifest();

  if (loading) {
    return (
      <div style={{ display: "flex", justifyContent: "center", marginTop: 80 }}>
        <Spinner />
      </div>
    );
  }

  if (error) {
    return (
      <NonIdealState
        icon="error"
        title="Could not load manifest.json"
        description={error}
      />
    );
  }

  if (!manifest || manifest.length === 0) {
    return (
      <NonIdealState
        icon="folder-open"
        title="No runs found"
        description="Run generate_manifest.py to populate the manifest."
      />
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <h2 className="bp5-heading" style={{ marginBottom: 24 }}>
        Synthoseis Runs
      </h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))",
          gap: 16,
        }}
      >
        {manifest.map((entry) => (
          <RunCard key={entry.folder} entry={entry} />
        ))}
      </div>
    </div>
  );
}
