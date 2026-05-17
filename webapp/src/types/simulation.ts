/**
 * Types for the Synthoseis simulation launcher API.
 */

export type RunStatus = "QUEUED" | "RUNNING" | "COMPLETE" | "FAILED";

export interface SandFraction {
  min: number;
  max: number;
}

export interface SimulationConfig {
  project: string;
  project_folder: string;
  work_folder: string;
  run_id?: string;
  cube_shape: [number, number, number];
  incident_angles: number[];
  digi: number;
  infill_factor: number;
  initial_layer_stdev: [number, number];
  thickness_min: number;
  thickness_max: number;
  seabed_min_depth: [number, number];
  signal_to_noise_ratio_db: [number, number, number];
  bandwidth_low: [number, number];
  bandwidth_high: [number, number];
  bandwidth_ord: number;
  dip_factor_max: number;
  min_number_faults: number;
  max_number_faults: number;
  pad_samples: number;
  max_column_height: [number, number];
  closure_types: Array<"simple" | "faulted" | "onlap">;
  min_closure_voxels_simple: number;
  min_closure_voxels_faulted: number;
  min_closure_voxels_onlap: number;
  sand_layer_thickness: number;
  sand_layer_fraction: SandFraction;
  extra_qc_plots: boolean;
  verbose: boolean;
  partial_voxels: boolean;
  variable_shale_ng: boolean;
  basin_floor_fans: boolean;
  include_channels: boolean;
  include_salt: boolean;
  write_to_hdf: boolean;
  broadband_qc_volume: boolean;
  model_qc_volumes: boolean;
  multiprocess_bp: boolean;
  model_store_in_memory: boolean;
  cleanup_intermediates: boolean;
}

export interface RunRecord {
  run_id: string;
  status: RunStatus;
  config: SimulationConfig;
  started_at: string;
  ended_at: string | null;
  output_folder: string | null;
}
