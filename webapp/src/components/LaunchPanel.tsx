/**
 * LaunchPanel — 5-section Blueprint.js form for submitting a new simulation.
 *
 * Sections:
 *   1. Output   — project, project_folder, work_folder, run_id
 *   2. Geometry — cube_shape, incident_angles, digi, infill_factor, pad_samples
 *   3. Geology  — layer / thickness / fault / closure params
 *   4. Seismic  — SNR, bandwidth
 *   5. Flags    — all boolean switches
 */
import React from "react";
import {
  Button,
  Card,
  Checkbox,
  Elevation,
  FormGroup,
  HTMLSelect,
  InputGroup,
  Intent,
  NumericInput,
  Switch,
  Tag,
  Tooltip,
} from "@blueprintjs/core";
import { fetchModels, submitRun, browseDirectory } from "../api/client";
import type { SimulationConfig } from "../types/simulation";

// ---------------------------------------------------------------------------
// Defaults (hard-coded from config/example.json)
// ---------------------------------------------------------------------------

const DEFAULT_CONFIG: SimulationConfig = {
  project: "rpm_example",
  project_folder: "~/synthoseis_output",
  work_folder: "~/synthoseis_work",
  run_id: undefined,
  cube_shape: [300, 300, 1250],
  incident_angles: [7, 15, 24],
  digi: 4,
  infill_factor: 10,
  initial_layer_stdev: [7.0, 25.0],
  thickness_min: 2,
  thickness_max: 12,
  seabed_min_depth: [20, 50],
  signal_to_noise_ratio_db: [7.5, 12.5, 17.5],
  bandwidth_low: [3.0, 6.0],
  bandwidth_high: [20.0, 35.0],
  bandwidth_ord: 4,
  dip_factor_max: 2.0,
  min_number_faults: 1,
  max_number_faults: 6,
  pad_samples: 10,
  max_column_height: [150.0, 150.0],
  closure_types: ["simple", "faulted", "onlap"],
  min_closure_voxels_simple: 500,
  min_closure_voxels_faulted: 2500,
  min_closure_voxels_onlap: 500,
  sand_layer_thickness: 2,
  sand_layer_fraction: { min: 0.05, max: 0.25 },
  extra_qc_plots: true,
  verbose: true,
  partial_voxels: true,
  variable_shale_ng: false,
  basin_floor_fans: false,
  include_channels: false,
  include_salt: true,
  write_to_hdf: false,
  broadband_qc_volume: false,
  model_qc_volumes: true,
  multiprocess_bp: true,
  model_store_in_memory: false,
  cleanup_intermediates: true,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function Row({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>{children}</div>
  );
}

function SectionCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card elevation={Elevation.ONE} style={{ marginBottom: 16 }}>
      <h4 style={{ marginTop: 0, marginBottom: 12 }}>{title}</h4>
      {children}
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface LaunchPanelProps {
  onNavigate?: (hash: string) => void;
}

export default function LaunchPanel({
  onNavigate,
}: LaunchPanelProps): React.ReactElement {
  // ---- State ---------------------------------------------------------------
  const [models, setModels] = React.useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = React.useState(true);
  const [submitting, setSubmitting] = React.useState(false);
  const [submitError, setSubmitError] = React.useState<string | null>(null);

  const [project, setProject] = React.useState(DEFAULT_CONFIG.project);
  const [projectFolder, setProjectFolder] = React.useState(
    DEFAULT_CONFIG.project_folder
  );
  const [workFolder, setWorkFolder] = React.useState(DEFAULT_CONFIG.work_folder);
  const [runId, setRunId] = React.useState("");

  // Geometry
  const [cubeX, setCubeX] = React.useState(DEFAULT_CONFIG.cube_shape[0]);
  const [cubeY, setCubeY] = React.useState(DEFAULT_CONFIG.cube_shape[1]);
  const [cubeZ, setCubeZ] = React.useState(DEFAULT_CONFIG.cube_shape[2]);
  const [incidentAngles, setIncidentAngles] = React.useState<number[]>(
    DEFAULT_CONFIG.incident_angles
  );
  const [angleInputStr, setAngleInputStr] = React.useState("");
  const [digi, setDigi] = React.useState(DEFAULT_CONFIG.digi);
  const [infillFactor, setInfillFactor] = React.useState(
    DEFAULT_CONFIG.infill_factor
  );
  const [padSamples, setPadSamples] = React.useState(DEFAULT_CONFIG.pad_samples);

  // Geology
  const [initLayerLow, setInitLayerLow] = React.useState(
    DEFAULT_CONFIG.initial_layer_stdev[0]
  );
  const [initLayerHigh, setInitLayerHigh] = React.useState(
    DEFAULT_CONFIG.initial_layer_stdev[1]
  );
  const [thicknessMin, setThicknessMin] = React.useState(
    DEFAULT_CONFIG.thickness_min
  );
  const [thicknessMax, setThicknessMax] = React.useState(
    DEFAULT_CONFIG.thickness_max
  );
  const [seabedMin, setSeabedMin] = React.useState(
    DEFAULT_CONFIG.seabed_min_depth[0]
  );
  const [seabedMax, setSeabedMax] = React.useState(
    DEFAULT_CONFIG.seabed_min_depth[1]
  );
  const [dipFactorMax, setDipFactorMax] = React.useState(
    DEFAULT_CONFIG.dip_factor_max
  );
  const [minFaults, setMinFaults] = React.useState(
    DEFAULT_CONFIG.min_number_faults
  );
  const [maxFaults, setMaxFaults] = React.useState(
    DEFAULT_CONFIG.max_number_faults
  );
  const [maxColLow, setMaxColLow] = React.useState(
    DEFAULT_CONFIG.max_column_height[0]
  );
  const [maxColHigh, setMaxColHigh] = React.useState(
    DEFAULT_CONFIG.max_column_height[1]
  );
  const [sandLayerThickness, setSandLayerThickness] = React.useState(
    DEFAULT_CONFIG.sand_layer_thickness
  );
  const [sandFracMin, setSandFracMin] = React.useState(
    DEFAULT_CONFIG.sand_layer_fraction.min
  );
  const [sandFracMax, setSandFracMax] = React.useState(
    DEFAULT_CONFIG.sand_layer_fraction.max
  );
  const [closureSimple, setClosureSimple] = React.useState(
    DEFAULT_CONFIG.closure_types.includes("simple")
  );
  const [closureFaulted, setClosureFaulted] = React.useState(
    DEFAULT_CONFIG.closure_types.includes("faulted")
  );
  const [closureOnlap, setClosureOnlap] = React.useState(
    DEFAULT_CONFIG.closure_types.includes("onlap")
  );
  const [minVoxelsSimple, setMinVoxelsSimple] = React.useState(
    DEFAULT_CONFIG.min_closure_voxels_simple
  );
  const [minVoxelsFaulted, setMinVoxelsFaulted] = React.useState(
    DEFAULT_CONFIG.min_closure_voxels_faulted
  );
  const [minVoxelsOnlap, setMinVoxelsOnlap] = React.useState(
    DEFAULT_CONFIG.min_closure_voxels_onlap
  );

  // Seismic
  const [snrLeft, setSnrLeft] = React.useState(
    DEFAULT_CONFIG.signal_to_noise_ratio_db[0]
  );
  const [snrMode, setSnrMode] = React.useState(
    DEFAULT_CONFIG.signal_to_noise_ratio_db[1]
  );
  const [snrRight, setSnrRight] = React.useState(
    DEFAULT_CONFIG.signal_to_noise_ratio_db[2]
  );
  const [bwLowLow, setBwLowLow] = React.useState(
    DEFAULT_CONFIG.bandwidth_low[0]
  );
  const [bwLowHigh, setBwLowHigh] = React.useState(
    DEFAULT_CONFIG.bandwidth_low[1]
  );
  const [bwHighLow, setBwHighLow] = React.useState(
    DEFAULT_CONFIG.bandwidth_high[0]
  );
  const [bwHighHigh, setBwHighHigh] = React.useState(
    DEFAULT_CONFIG.bandwidth_high[1]
  );
  const [bwOrd, setBwOrd] = React.useState(DEFAULT_CONFIG.bandwidth_ord);

  // Flags
  const [extraQcPlots, setExtraQcPlots] = React.useState(
    DEFAULT_CONFIG.extra_qc_plots
  );
  const [verbose, setVerbose] = React.useState(DEFAULT_CONFIG.verbose);
  const [partialVoxels, setPartialVoxels] = React.useState(
    DEFAULT_CONFIG.partial_voxels
  );
  const [variableShaleNg, setVariableShaleNg] = React.useState(
    DEFAULT_CONFIG.variable_shale_ng
  );
  const [basinFloorFans, setBasinFloorFans] = React.useState(
    DEFAULT_CONFIG.basin_floor_fans
  );
  const [includeChannels, setIncludeChannels] = React.useState(
    DEFAULT_CONFIG.include_channels
  );
  const [includeSalt, setIncludeSalt] = React.useState(
    DEFAULT_CONFIG.include_salt
  );
  const [writeToHdf, setWriteToHdf] = React.useState(DEFAULT_CONFIG.write_to_hdf);
  const [broadbandQcVolume, setBroadbandQcVolume] = React.useState(
    DEFAULT_CONFIG.broadband_qc_volume
  );
  const [modelQcVolumes, setModelQcVolumes] = React.useState(
    DEFAULT_CONFIG.model_qc_volumes
  );
  const [multiprocessBp, setMultiprocessBp] = React.useState(
    DEFAULT_CONFIG.multiprocess_bp
  );
  const [modelStoreInMemory, setModelStoreInMemory] = React.useState(
    DEFAULT_CONFIG.model_store_in_memory
  );
  const [cleanupIntermediates, setCleanupIntermediates] = React.useState(
    DEFAULT_CONFIG.cleanup_intermediates
  );

  // ---- Load models on mount ------------------------------------------------
  React.useEffect(() => {
    fetchModels()
      .then((ms) => {
        setModels(ms);
        if (ms.length > 0 && !ms.includes(project)) {
          setProject(ms[0]);
        }
      })
      .catch(() => setModels([]))
      .finally(() => setModelsLoading(false));
  }, []);

  // ---- Derived validation --------------------------------------------------
  const thicknessError =
    thicknessMin >= thicknessMax
      ? "thickness_min must be less than thickness_max"
      : null;

  const selectedClosureTypes: Array<"simple" | "faulted" | "onlap"> = [
    ...(closureSimple ? (["simple"] as const) : []),
    ...(closureFaulted ? (["faulted"] as const) : []),
    ...(closureOnlap ? (["onlap"] as const) : []),
  ];
  const closureError =
    selectedClosureTypes.length === 0 ? "At least one closure type required" : null;

  const anglesError =
    incidentAngles.length === 0 ? "At least one incident angle required" : null;

  const sandFracError =
    sandFracMin >= sandFracMax
      ? "sand_layer_fraction min must be less than max"
      : null;

  const hasErrors =
    !!thicknessError || !!closureError || !!anglesError || !!sandFracError;

  // ---- Handlers ------------------------------------------------------------
  function handleLoadDefaults() {
    setProject(DEFAULT_CONFIG.project);
    setProjectFolder(DEFAULT_CONFIG.project_folder);
    setWorkFolder(DEFAULT_CONFIG.work_folder);
    setRunId("");
    setCubeX(DEFAULT_CONFIG.cube_shape[0]);
    setCubeY(DEFAULT_CONFIG.cube_shape[1]);
    setCubeZ(DEFAULT_CONFIG.cube_shape[2]);
    setIncidentAngles(DEFAULT_CONFIG.incident_angles);
    setDigi(DEFAULT_CONFIG.digi);
    setInfillFactor(DEFAULT_CONFIG.infill_factor);
    setPadSamples(DEFAULT_CONFIG.pad_samples);
    setInitLayerLow(DEFAULT_CONFIG.initial_layer_stdev[0]);
    setInitLayerHigh(DEFAULT_CONFIG.initial_layer_stdev[1]);
    setThicknessMin(DEFAULT_CONFIG.thickness_min);
    setThicknessMax(DEFAULT_CONFIG.thickness_max);
    setSeabedMin(DEFAULT_CONFIG.seabed_min_depth[0]);
    setSeabedMax(DEFAULT_CONFIG.seabed_min_depth[1]);
    setDipFactorMax(DEFAULT_CONFIG.dip_factor_max);
    setMinFaults(DEFAULT_CONFIG.min_number_faults);
    setMaxFaults(DEFAULT_CONFIG.max_number_faults);
    setMaxColLow(DEFAULT_CONFIG.max_column_height[0]);
    setMaxColHigh(DEFAULT_CONFIG.max_column_height[1]);
    setSandLayerThickness(DEFAULT_CONFIG.sand_layer_thickness);
    setSandFracMin(DEFAULT_CONFIG.sand_layer_fraction.min);
    setSandFracMax(DEFAULT_CONFIG.sand_layer_fraction.max);
    setClosureSimple(true);
    setClosureFaulted(true);
    setClosureOnlap(true);
    setMinVoxelsSimple(DEFAULT_CONFIG.min_closure_voxels_simple);
    setMinVoxelsFaulted(DEFAULT_CONFIG.min_closure_voxels_faulted);
    setMinVoxelsOnlap(DEFAULT_CONFIG.min_closure_voxels_onlap);
    setSnrLeft(DEFAULT_CONFIG.signal_to_noise_ratio_db[0]);
    setSnrMode(DEFAULT_CONFIG.signal_to_noise_ratio_db[1]);
    setSnrRight(DEFAULT_CONFIG.signal_to_noise_ratio_db[2]);
    setBwLowLow(DEFAULT_CONFIG.bandwidth_low[0]);
    setBwLowHigh(DEFAULT_CONFIG.bandwidth_low[1]);
    setBwHighLow(DEFAULT_CONFIG.bandwidth_high[0]);
    setBwHighHigh(DEFAULT_CONFIG.bandwidth_high[1]);
    setBwOrd(DEFAULT_CONFIG.bandwidth_ord);
    setExtraQcPlots(DEFAULT_CONFIG.extra_qc_plots);
    setVerbose(DEFAULT_CONFIG.verbose);
    setPartialVoxels(DEFAULT_CONFIG.partial_voxels);
    setVariableShaleNg(DEFAULT_CONFIG.variable_shale_ng);
    setBasinFloorFans(DEFAULT_CONFIG.basin_floor_fans);
    setIncludeChannels(DEFAULT_CONFIG.include_channels);
    setIncludeSalt(DEFAULT_CONFIG.include_salt);
    setWriteToHdf(DEFAULT_CONFIG.write_to_hdf);
    setBroadbandQcVolume(DEFAULT_CONFIG.broadband_qc_volume);
    setModelQcVolumes(DEFAULT_CONFIG.model_qc_volumes);
    setMultiprocessBp(DEFAULT_CONFIG.multiprocess_bp);
    setModelStoreInMemory(DEFAULT_CONFIG.model_store_in_memory);
    setCleanupIntermediates(DEFAULT_CONFIG.cleanup_intermediates);
  }

  async function handleSubmit() {
    if (hasErrors) return;
    setSubmitting(true);
    setSubmitError(null);
    const config: SimulationConfig = {
      project,
      project_folder: projectFolder,
      work_folder: workFolder,
      run_id: runId || undefined,
      cube_shape: [cubeX, cubeY, cubeZ],
      incident_angles: incidentAngles,
      digi,
      infill_factor: infillFactor,
      initial_layer_stdev: [initLayerLow, initLayerHigh],
      thickness_min: thicknessMin,
      thickness_max: thicknessMax,
      seabed_min_depth: [seabedMin, seabedMax],
      signal_to_noise_ratio_db: [snrLeft, snrMode, snrRight],
      bandwidth_low: [bwLowLow, bwLowHigh],
      bandwidth_high: [bwHighLow, bwHighHigh],
      bandwidth_ord: bwOrd,
      dip_factor_max: dipFactorMax,
      min_number_faults: minFaults,
      max_number_faults: maxFaults,
      pad_samples: padSamples,
      max_column_height: [maxColLow, maxColHigh],
      closure_types: selectedClosureTypes,
      min_closure_voxels_simple: minVoxelsSimple,
      min_closure_voxels_faulted: minVoxelsFaulted,
      min_closure_voxels_onlap: minVoxelsOnlap,
      sand_layer_thickness: sandLayerThickness,
      sand_layer_fraction: { min: sandFracMin, max: sandFracMax },
      extra_qc_plots: extraQcPlots,
      verbose,
      partial_voxels: partialVoxels,
      variable_shale_ng: variableShaleNg,
      basin_floor_fans: basinFloorFans,
      include_channels: includeChannels,
      include_salt: includeSalt,
      write_to_hdf: writeToHdf,
      broadband_qc_volume: broadbandQcVolume,
      model_qc_volumes: modelQcVolumes,
      multiprocess_bp: multiprocessBp,
      model_store_in_memory: modelStoreInMemory,
      cleanup_intermediates: cleanupIntermediates,
    };
    try {
      const result = await submitRun(config);
      const nav = onNavigate ?? ((h: string) => { window.location.hash = h; });
      nav(`#/runs/${result.run_id}/logs`);
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  function handleAngleInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      const val = parseInt(angleInputStr.trim(), 10);
      if (!isNaN(val) && !incidentAngles.includes(val)) {
        setIncidentAngles([...incidentAngles, val]);
      }
      setAngleInputStr("");
    }
  }

  // ---- Render --------------------------------------------------------------
  return (
    <div style={{ padding: 24, maxWidth: 900, margin: "0 auto" }}>
      <h2 style={{ marginBottom: 16 }}>Launch Simulation</h2>

      {/* ── Section 1: Output ── */}
      <SectionCard title="1 · Output">
        <FormGroup label="Rock-physics model (project)" labelFor="project-select">
          {modelsLoading ? (
            <Tag minimal>Loading models…</Tag>
          ) : (
            <HTMLSelect
              id="project-select"
              value={project}
              onChange={(e) => setProject(e.target.value)}
              options={models.length > 0 ? models : [project]}
            />
          )}
        </FormGroup>
        <FormGroup label="Project folder" labelFor="project-folder">
          <InputGroup
            id="project-folder"
            value={projectFolder}
            onChange={(e) => setProjectFolder(e.target.value)}
            placeholder="~/synthoseis_output"
            rightElement={
              <Button
                icon="folder-open"
                minimal
                title="Browse for folder"
                onClick={async () => {
                  try {
                    const picked = await browseDirectory(projectFolder || undefined);
                    if (picked) setProjectFolder(picked);
                  } catch {
                    // tkinter unavailable — user can type the path manually
                  }
                }}
              />
            }
          />
        </FormGroup>
        <FormGroup label="Work folder" labelFor="work-folder">
          <InputGroup
            id="work-folder"
            value={workFolder}
            onChange={(e) => setWorkFolder(e.target.value)}
            placeholder="~/synthoseis_work"
            rightElement={
              <Button
                icon="folder-open"
                minimal
                title="Browse for folder"
                onClick={async () => {
                  try {
                    const picked = await browseDirectory(workFolder || undefined);
                    if (picked) setWorkFolder(picked);
                  } catch {
                    // tkinter unavailable — user can type the path manually
                  }
                }}
              />
            }
          />
        </FormGroup>
        <FormGroup label="Run ID (optional)" labelFor="run-id">
          <InputGroup
            id="run-id"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            placeholder="auto-generated"
          />
        </FormGroup>
      </SectionCard>

      {/* ── Section 2: Geometry ── */}
      <SectionCard title="2 · Geometry">
        <FormGroup label="Cube shape (X / Y / Z)">
          <Row>
            <NumericInput
              value={cubeX}
              onValueChange={setCubeX}
              min={1}
              style={{ width: 100 }}
              placeholder="X"
            />
            <NumericInput
              value={cubeY}
              onValueChange={setCubeY}
              min={1}
              style={{ width: 100 }}
              placeholder="Y"
            />
            <NumericInput
              value={cubeZ}
              onValueChange={setCubeZ}
              min={1}
              style={{ width: 100 }}
              placeholder="Z"
            />
          </Row>
        </FormGroup>

        <FormGroup
          label="Incident angles (press Enter to add)"
          labelFor="angle-input"
          helperText={anglesError ?? undefined}
          intent={anglesError ? Intent.DANGER : Intent.NONE}
        >
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 8 }}>
            {incidentAngles.map((a) => (
              <Tag
                key={a}
                onRemove={() =>
                  setIncidentAngles(incidentAngles.filter((x) => x !== a))
                }
                intent={anglesError ? Intent.DANGER : Intent.NONE}
              >
                {a}°
              </Tag>
            ))}
          </div>
          <InputGroup
            id="angle-input"
            value={angleInputStr}
            onChange={(e) => setAngleInputStr(e.target.value)}
            onKeyDown={handleAngleInputKeyDown}
            placeholder="Type angle + Enter"
            style={{ width: 180 }}
          />
        </FormGroup>

        <Row>
          <FormGroup label="Digitisation (digi)">
            <NumericInput
              value={digi}
              onValueChange={setDigi}
              min={1}
              style={{ width: 120 }}
            />
          </FormGroup>
          <FormGroup label="Infill factor">
            <NumericInput
              value={infillFactor}
              onValueChange={setInfillFactor}
              min={1}
              style={{ width: 120 }}
            />
          </FormGroup>
          <FormGroup label="Pad samples">
            <NumericInput
              value={padSamples}
              onValueChange={setPadSamples}
              min={0}
              style={{ width: 120 }}
            />
          </FormGroup>
        </Row>
      </SectionCard>

      {/* ── Section 3: Geology ── */}
      <SectionCard title="3 · Geology">
        <Row>
          <FormGroup label="Initial layer stdev (Low)">
            <NumericInput
              value={initLayerLow}
              onValueChange={setInitLayerLow}
              min={0}
              stepSize={0.1}
              style={{ width: 120 }}
            />
          </FormGroup>
          <FormGroup label="Initial layer stdev (High)">
            <NumericInput
              value={initLayerHigh}
              onValueChange={setInitLayerHigh}
              min={0}
              stepSize={0.1}
              style={{ width: 120 }}
            />
          </FormGroup>
        </Row>

        <Row>
          <FormGroup
            label="Thickness min"
            helperText={thicknessError ?? undefined}
            intent={thicknessError ? Intent.DANGER : Intent.NONE}
          >
            <NumericInput
              value={thicknessMin}
              onValueChange={setThicknessMin}
              min={1}
              intent={thicknessError ? Intent.DANGER : Intent.NONE}
              style={{ width: 120 }}
            />
          </FormGroup>
          <FormGroup label="Thickness max">
            <NumericInput
              value={thicknessMax}
              onValueChange={setThicknessMax}
              min={1}
              intent={thicknessError ? Intent.DANGER : Intent.NONE}
              style={{ width: 120 }}
            />
          </FormGroup>
        </Row>

        <Row>
          <FormGroup label="Seabed depth min">
            <NumericInput
              value={seabedMin}
              onValueChange={setSeabedMin}
              style={{ width: 120 }}
            />
          </FormGroup>
          <FormGroup label="Seabed depth max">
            <NumericInput
              value={seabedMax}
              onValueChange={setSeabedMax}
              style={{ width: 120 }}
            />
          </FormGroup>
          <FormGroup label="Dip factor max">
            <NumericInput
              value={dipFactorMax}
              onValueChange={setDipFactorMax}
              min={0}
              stepSize={0.1}
              style={{ width: 120 }}
            />
          </FormGroup>
        </Row>

        <Row>
          <FormGroup label="Min faults">
            <NumericInput
              value={minFaults}
              onValueChange={setMinFaults}
              min={0}
              style={{ width: 100 }}
            />
          </FormGroup>
          <FormGroup label="Max faults">
            <NumericInput
              value={maxFaults}
              onValueChange={setMaxFaults}
              min={0}
              style={{ width: 100 }}
            />
          </FormGroup>
        </Row>

        <Row>
          <FormGroup label="Max column height (Low)">
            <NumericInput
              value={maxColLow}
              onValueChange={setMaxColLow}
              stepSize={10}
              style={{ width: 130 }}
            />
          </FormGroup>
          <FormGroup label="Max column height (High)">
            <NumericInput
              value={maxColHigh}
              onValueChange={setMaxColHigh}
              stepSize={10}
              style={{ width: 130 }}
            />
          </FormGroup>
        </Row>

        <Row>
          <FormGroup label="Sand layer thickness">
            <NumericInput
              value={sandLayerThickness}
              onValueChange={setSandLayerThickness}
              min={1}
              style={{ width: 130 }}
            />
          </FormGroup>
          <FormGroup
            label="Sand fraction min"
            helperText={sandFracError ?? undefined}
            intent={sandFracError ? Intent.DANGER : Intent.NONE}
          >
            <NumericInput
              value={sandFracMin}
              onValueChange={setSandFracMin}
              min={0.01}
              max={0.99}
              stepSize={0.01}
              intent={sandFracError ? Intent.DANGER : Intent.NONE}
              style={{ width: 130 }}
            />
          </FormGroup>
          <FormGroup label="Sand fraction max">
            <NumericInput
              value={sandFracMax}
              onValueChange={setSandFracMax}
              min={0.01}
              max={0.99}
              stepSize={0.01}
              intent={sandFracError ? Intent.DANGER : Intent.NONE}
              style={{ width: 130 }}
            />
          </FormGroup>
        </Row>

        <FormGroup
          label="Closure types"
          helperText={closureError ?? undefined}
          intent={closureError ? Intent.DANGER : Intent.NONE}
        >
          <Row>
            <Checkbox
              label="Simple"
              checked={closureSimple}
              onChange={(e) =>
                setClosureSimple((e.target as HTMLInputElement).checked)
              }
            />
            <Checkbox
              label="Faulted"
              checked={closureFaulted}
              onChange={(e) =>
                setClosureFaulted((e.target as HTMLInputElement).checked)
              }
            />
            <Checkbox
              label="Onlap"
              checked={closureOnlap}
              onChange={(e) =>
                setClosureOnlap((e.target as HTMLInputElement).checked)
              }
            />
          </Row>
        </FormGroup>

        <Row>
          <FormGroup label="Min voxels (simple)">
            <NumericInput
              value={minVoxelsSimple}
              onValueChange={setMinVoxelsSimple}
              min={1}
              style={{ width: 130 }}
            />
          </FormGroup>
          <FormGroup label="Min voxels (faulted)">
            <NumericInput
              value={minVoxelsFaulted}
              onValueChange={setMinVoxelsFaulted}
              min={1}
              style={{ width: 130 }}
            />
          </FormGroup>
          <FormGroup label="Min voxels (onlap)">
            <NumericInput
              value={minVoxelsOnlap}
              onValueChange={setMinVoxelsOnlap}
              min={1}
              style={{ width: 130 }}
            />
          </FormGroup>
        </Row>
      </SectionCard>

      {/* ── Section 4: Seismic ── */}
      <SectionCard title="4 · Seismic">
        <FormGroup label="Signal-to-noise ratio dB (Left / Mode / Right)">
          <Row>
            <NumericInput
              value={snrLeft}
              onValueChange={setSnrLeft}
              stepSize={0.5}
              style={{ width: 120 }}
              placeholder="Left"
            />
            <NumericInput
              value={snrMode}
              onValueChange={setSnrMode}
              stepSize={0.5}
              style={{ width: 120 }}
              placeholder="Mode"
            />
            <NumericInput
              value={snrRight}
              onValueChange={setSnrRight}
              stepSize={0.5}
              style={{ width: 120 }}
              placeholder="Right"
            />
          </Row>
        </FormGroup>

        <Row>
          <FormGroup label="Bandwidth low (Low / High)">
            <Row>
              <NumericInput
                value={bwLowLow}
                onValueChange={setBwLowLow}
                stepSize={0.5}
                style={{ width: 100 }}
              />
              <NumericInput
                value={bwLowHigh}
                onValueChange={setBwLowHigh}
                stepSize={0.5}
                style={{ width: 100 }}
              />
            </Row>
          </FormGroup>
          <FormGroup label="Bandwidth high (Low / High)">
            <Row>
              <NumericInput
                value={bwHighLow}
                onValueChange={setBwHighLow}
                stepSize={0.5}
                style={{ width: 100 }}
              />
              <NumericInput
                value={bwHighHigh}
                onValueChange={setBwHighHigh}
                stepSize={0.5}
                style={{ width: 100 }}
              />
            </Row>
          </FormGroup>
          <FormGroup label="Bandwidth order">
            <NumericInput
              value={bwOrd}
              onValueChange={setBwOrd}
              min={1}
              style={{ width: 100 }}
            />
          </FormGroup>
        </Row>
      </SectionCard>

      {/* ── Section 5: Flags ── */}
      <SectionCard title="5 · Flags">
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "4px 24px",
          }}
        >
          <Switch
            label="Extra QC plots"
            checked={extraQcPlots}
            onChange={(e) => setExtraQcPlots((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Verbose"
            checked={verbose}
            onChange={(e) => setVerbose((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Partial voxels"
            checked={partialVoxels}
            onChange={(e) => setPartialVoxels((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Variable shale N:G"
            checked={variableShaleNg}
            onChange={(e) => setVariableShaleNg((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Basin floor fans"
            checked={basinFloorFans}
            onChange={(e) => setBasinFloorFans((e.target as HTMLInputElement).checked)}
          />
          <Tooltip content="Deprecated — channels are always disabled" placement="top">
            <Switch
              label="Include channels"
              checked={false}
              disabled
              onChange={() => { /* deprecated, always false */ }}
            />
          </Tooltip>
          <Switch
            label="Include salt"
            checked={includeSalt}
            onChange={(e) => setIncludeSalt((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Write to HDF"
            checked={writeToHdf}
            onChange={(e) => setWriteToHdf((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Broadband QC volume"
            checked={broadbandQcVolume}
            onChange={(e) => setBroadbandQcVolume((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Model QC volumes"
            checked={modelQcVolumes}
            onChange={(e) => setModelQcVolumes((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Multiprocess BP"
            checked={multiprocessBp}
            onChange={(e) => setMultiprocessBp((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Model store in memory"
            checked={modelStoreInMemory}
            onChange={(e) => setModelStoreInMemory((e.target as HTMLInputElement).checked)}
          />
          <Switch
            label="Cleanup intermediates"
            checked={cleanupIntermediates}
            onChange={(e) => setCleanupIntermediates((e.target as HTMLInputElement).checked)}
          />
        </div>
      </SectionCard>

      {/* ── Action buttons ── */}
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <Button icon="reset" onClick={handleLoadDefaults} disabled={submitting}>
          Load defaults
        </Button>
        <Button
          intent={Intent.SUCCESS}
          icon="play"
          large
          loading={submitting}
          disabled={hasErrors}
          onClick={handleSubmit}
        >
          Run simulation
        </Button>
        {submitError && (
          <Tag intent={Intent.DANGER} minimal>
            {submitError}
          </Tag>
        )}
      </div>
    </div>
  );
}
