import React from "react";
import {
  Button,
  InputGroup,
  NonIdealState,
  Spinner,
  Tag,
  Intent,
} from "@blueprintjs/core";
import useManifest from "../hooks/useManifest";
import RunCard from "./RunCard";
import { browseDirectory } from "../api/client";

const STORAGE_KEY = "synthoseis_project_folder";

/** Regex for a seismic run folder name, e.g. seismic__0517_2351_<uuid> */
const RUN_FOLDER_RE = /\/seismic__(\d{8}|\d{4}_\d{4})_[^/]+\/?$/;

/**
 * If the user entered a run subfolder path directly (e.g. …/seismic__0517_…)
 * strip back to the parent project folder so the manifest scan works correctly.
 */
function normaliseFolder(raw: string): string {
  const trimmed = raw.trim().replace(/\/$/, "");
  return RUN_FOLDER_RE.test(trimmed)
    ? trimmed.replace(RUN_FOLDER_RE, "")
    : trimmed;
}

export default function ProjectDashboard(): React.ReactElement {
  // ── Folder state (persisted to localStorage) ──────────────────────────────
  const [folderInput, setFolderInput] = React.useState<string>(() => {
    try { return normaliseFolder(localStorage.getItem(STORAGE_KEY) ?? ""); } catch { return ""; }
  });
  // The committed folder that actually drives the manifest fetch.
  // Initialised from storage so the last folder auto-loads on page visit.
  const [activeFolder, setActiveFolder] = React.useState<string>(() => {
    try { return normaliseFolder(localStorage.getItem(STORAGE_KEY) ?? ""); } catch { return ""; }
  });
  const [browsing, setBrowsing] = React.useState(false);

  const { data: manifest, loading, error } = useManifest(activeFolder || null);

  // ── Handlers ──────────────────────────────────────────────────────────────
  function handleLoad() {
    const normalised = normaliseFolder(folderInput);
    setFolderInput(normalised);
    setActiveFolder(normalised);
    if (normalised) {
      localStorage.setItem(STORAGE_KEY, normalised);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }

  async function handleBrowse() {
    setBrowsing(true);
    try {
      const picked = await browseDirectory(folderInput || undefined);
      if (picked) {
        const normalised = normaliseFolder(picked);
        setFolderInput(normalised);
        setActiveFolder(normalised);
        localStorage.setItem(STORAGE_KEY, normalised);
      }
    } catch {
      // Server-side picker unavailable — user can type the path manually
    } finally {
      setBrowsing(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") handleLoad();
  }

  // ── Render ────────────────────────────────────────────────────────────────
  const showPlaceholder = !activeFolder;

  return (
    <div style={{ padding: 24 }}>
      {/* ── Folder picker ─────────────────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "flex-end",
          marginBottom: 24,
          maxWidth: 700,
        }}
      >
        <div style={{ flex: 1 }}>
          <label
            className="bp5-label"
            style={{ marginBottom: 4, display: "block" }}
          >
            Project folder
          </label>
          <InputGroup
            value={folderInput}
            onChange={(e) => setFolderInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="~/synthoseis_output"
            rightElement={
              <Button
                icon="folder-open"
                minimal
                loading={browsing}
                title="Browse…"
                onClick={handleBrowse}
              />
            }
          />
        </div>
        <Button icon="refresh" onClick={handleLoad} disabled={loading}>
          Load
        </Button>
      </div>

      {/* ── Content area ──────────────────────────────────────────────────── */}
      {showPlaceholder ? (
        <NonIdealState
          icon="folder-open"
          title="Choose a project folder"
          description="Enter the path to a synthoseis output folder above and click Load, or use the browse button."
        />
      ) : loading ? (
        <div style={{ display: "flex", justifyContent: "center", marginTop: 80 }}>
          <Spinner />
        </div>
      ) : error ? (
        <NonIdealState
          icon="error"
          title="Could not load manifest"
          description={error}
          action={
            <Button icon="refresh" onClick={handleLoad}>
              Retry
            </Button>
          }
        />
      ) : !manifest || manifest.length === 0 ? (
        <NonIdealState
          icon="search"
          title="No runs found"
          description={`No seismic__* run folders were found in "${activeFolder}".`}
          action={
            <Button icon="refresh" onClick={handleLoad}>
              Refresh
            </Button>
          }
        />
      ) : (
        <>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 16,
            }}
          >
            <h2 className="bp5-heading" style={{ margin: 0 }}>
              Synthoseis Runs
            </h2>
            <Tag minimal>{manifest.length} run{manifest.length !== 1 ? "s" : ""}</Tag>
            <Tag minimal intent={Intent.NONE} style={{ color: "#8a9ba8" }}>
              {activeFolder}
            </Tag>
            <Button
              icon="refresh"
              minimal
              small
              onClick={handleLoad}
              title="Re-scan folder"
            />
          </div>
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
        </>
      )}
    </div>
  );
}
