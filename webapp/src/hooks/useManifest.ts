import React from "react";
import type { ManifestEntry } from "../types/manifest";
import { MANIFEST_URL } from "../config";

interface UseManifestResult {
  data: ManifestEntry[] | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fetches and parses the manifest from MANIFEST_URL (the /api/manifest endpoint).
 *
 * When *projectFolder* is provided it is forwarded as `?project_folder=…` so
 * the API scans that specific folder on disk rather than reading from the DB.
 * Pass `null` / `undefined` to use the DB-backed fallback.
 */
export default function useManifest(projectFolder?: string | null): UseManifestResult {
  const [data, setData] = React.useState<ManifestEntry[] | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const url = projectFolder
          ? `${MANIFEST_URL}?project_folder=${encodeURIComponent(projectFolder)}`
          : MANIFEST_URL;
        const resp = await fetch(url);
        if (!resp.ok) {
          // Prefer the API's error detail when available
          let detail = resp.statusText;
          try {
            const body = await resp.json();
            if (body?.detail) detail = body.detail;
          } catch { /* ignore */ }
          throw new Error(`manifest responded with HTTP ${resp.status}: ${detail}`);
        }
        const json = (await resp.json()) as ManifestEntry[];
        if (!cancelled) {
          setData(json);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Could not load manifest"
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [projectFolder]);

  return { data, loading, error };
}
