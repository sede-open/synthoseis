/**
 * RunsPanel — polls GET /api/runs every 5 seconds and displays a table.
 */
import React from "react";
import {
  AnchorButton,
  Button,
  HTMLTable,
  Intent,
  NonIdealState,
  Spinner,
  Tag,
} from "@blueprintjs/core";
import { deleteRun, listRuns } from "../api/client";
import type { RunRecord, RunStatus } from "../types/simulation";

function statusIntent(status: RunStatus): Intent {
  switch (status) {
    case "RUNNING":
      return Intent.PRIMARY;
    case "COMPLETE":
      return Intent.SUCCESS;
    case "FAILED":
      return Intent.DANGER;
    default:
      return Intent.NONE;
  }
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

export default function RunsPanel(): React.ReactElement {
  const [runs, setRuns] = React.useState<RunRecord[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const fetchRuns = React.useCallback(async () => {
    try {
      const data = await listRuns();
      setRuns(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchRuns();
    const interval = setInterval(fetchRuns, 5000);
    return () => clearInterval(interval);
  }, [fetchRuns]);

  async function handleCancel(runId: string) {
    try {
      await deleteRun(runId);
      await fetchRuns();
    } catch (err) {
      alert(`Failed to cancel run: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

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
        title="Could not load runs"
        description={error}
        action={<Button onClick={fetchRuns}>Retry</Button>}
      />
    );
  }

  if (runs.length === 0) {
    return (
      <NonIdealState
        icon="history"
        title="No simulation runs yet"
        description='Click "Launch" to start your first simulation.'
        action={
          <AnchorButton href="#/launch" intent={Intent.PRIMARY} icon="play">
            Launch
          </AnchorButton>
        }
      />
    );
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
        <h2 style={{ margin: 0 }}>Simulation Runs</h2>
        <Button icon="refresh" minimal onClick={fetchRuns}>
          Refresh
        </Button>
      </div>

      <HTMLTable striped style={{ width: "100%" }}>
        <thead>
          <tr>
            <th>Run ID</th>
            <th>Status</th>
            <th>Started</th>
            <th>Ended</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const outputBasename = run.output_folder
              ? run.output_folder.split("/").pop() ?? run.output_folder
              : null;
            return (
              <tr key={run.run_id}>
                <td>
                  <code style={{ fontSize: 12 }}>{run.run_id}</code>
                </td>
                <td>
                  <Tag
                    intent={statusIntent(run.status)}
                    minimal={run.status === "QUEUED"}
                  >
                    {run.status}
                  </Tag>
                </td>
                <td>{formatDate(run.started_at)}</td>
                <td>{formatDate(run.ended_at)}</td>
                <td>
                  <div style={{ display: "flex", gap: 6 }}>
                    <AnchorButton
                      href={`#/runs/${run.run_id}/logs`}
                      icon="document"
                      small
                      minimal
                    >
                      View logs
                    </AnchorButton>

                    {run.status === "COMPLETE" && outputBasename && (
                      <AnchorButton
                        href={`#/run/${encodeURIComponent(outputBasename)}`}
                        icon="eye-open"
                        intent={Intent.SUCCESS}
                        small
                        minimal
                      >
                        Open in viewer
                      </AnchorButton>
                    )}

                    {run.status === "RUNNING" && (
                      <Button
                        icon="stop"
                        intent={Intent.DANGER}
                        small
                        minimal
                        onClick={() => handleCancel(run.run_id)}
                      >
                        Cancel
                      </Button>
                    )}
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </HTMLTable>
    </div>
  );
}
