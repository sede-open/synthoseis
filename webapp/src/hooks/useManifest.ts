import React from "react";
import type { ManifestEntry } from "../types/manifest";
import { MANIFEST_URL } from "../config";

interface UseManifestResult {
  data: ManifestEntry[] | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fetches and parses manifest.json from MANIFEST_URL.
 * Returns the parsed manifest entries, a loading flag, and an error string.
 */
export default function useManifest(): UseManifestResult {
  const [data, setData] = React.useState<ManifestEntry[] | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const resp = await fetch(MANIFEST_URL);
        if (!resp.ok) {
          throw new Error(
            `manifest.json responded with HTTP ${resp.status}: ${resp.statusText}`
          );
        }
        const json = (await resp.json()) as ManifestEntry[];
        if (!cancelled) {
          setData(json);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : "Could not load manifest.json"
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
  }, []);

  return { data, loading, error };
}
