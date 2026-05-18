/** TypeScript interfaces mirroring the manifest.json schema. */

export interface VolumeInfo {
  /** Human-readable volume name (derived from store filename stem). */
  name: string;
  /** Store path relative to the run folder. e.g. "seismic/foo.zarr" */
  store_path: string;
  /** xarray variable name inside the zarr store. e.g. "amplitude" */
  variable: string;
  /** UI group: "Seismic" | "Geology" | "Horizons" | "Closures" | "QC" */
  group: string;
  /** Array shape as [inline, crossline, time|horizon] */
  shape: [number, number, number];
  /** numpy dtype string, e.g. "float32" */
  dtype: string;
  /** Dimension names, e.g. ["inline", "crossline", "time"] */
  dims: string[];
  /** Chunk shape matching dims order */
  chunks: number[];
  /** Compressor description string, e.g. "blosc:zstd:5:bitshuffle" */
  compressor: string;
  /** Store-level attributes, e.g. { angle_deg: 7, sample_rate_ms: 4 } */
  attrs: Record<string, unknown>;
}

export interface ManifestEntry {
  /** Run identifier extracted from folder name */
  run_id: string;
  /** Full run folder name, e.g. "seismic__20260517_my_run" */
  folder: string;
  /** 8-digit date string, e.g. "20260517" */
  datestamp: string;
  /** Shape of the primary 3-D cube [inline, crossline, time] */
  cube_shape: number[];
  /** All discovered volumes for this run */
  volumes: VolumeInfo[];
  /** Flat key/value parameters from parameters.db */
  parameters: Record<string, string>;
}
