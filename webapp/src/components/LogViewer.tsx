/**
 * LogViewer — streams live stdout from a simulation run via SSE.
 *
 * - Opens EventSource on mount, closes on unmount.
 * - Appends lines to a pre element and auto-scrolls to the bottom.
 * - Shows a Blueprint Alert warning if the user tries to close while
 *   the run is still RUNNING.
 */
import React from "react";
import {
  AnchorButton,
  Alert,
  Button,
  Card,
  Elevation,
  Intent,
  NonIdealState,
  Spinner,
  Tag,
} from "@blueprintjs/core";
import { streamLogs } from "../api/client";
import type { RunStatus } from "../types/simulation";

interface LogViewerProps {
  runId: string;
}

function statusIntent(status: RunStatus | null): Intent {
  switch (status) {
    case "RUNNING":
      return Intent.PRIMARY;
    case "COMPLETE":
      return Intent.SUCCESS;
    case "FAILED":
      return Intent.DANGER;
    case "QUEUED":
      return Intent.NONE;
    default:
      return Intent.NONE;
  }
}

export default function LogViewer({ runId }: LogViewerProps): React.ReactElement {
  const [lines, setLines] = React.useState<string[]>([]);
  const [status, setStatus] = React.useState<RunStatus | null>("RUNNING");
  const [error, setError] = React.useState<string | null>(null);
  const [showCloseAlert, setShowCloseAlert] = React.useState(false);

  const preRef = React.useRef<HTMLPreElement>(null);

  // Auto-scroll to bottom when new lines arrive
  React.useEffect(() => {
    const pre = preRef.current;
    if (pre) {
      pre.scrollTop = pre.scrollHeight;
    }
  }, [lines]);

  // Open SSE stream on mount
  React.useEffect(() => {
    const es = streamLogs(runId);

    es.onmessage = (event) => {
      setLines((prev) => [...prev, event.data]);
    };

    es.addEventListener("status", (event: Event) => {
      const msgEvent = event as MessageEvent;
      const newStatus = msgEvent.data as RunStatus;
      setStatus(newStatus);
      if (newStatus === "COMPLETE" || newStatus === "FAILED") {
        es.close();
      }
    });

    es.onerror = () => {
      setError("SSE connection lost.");
      es.close();
    };

    return () => {
      es.close();
    };
  }, [runId]);

  function handleCopyLog() {
    navigator.clipboard
      .writeText(lines.join("\n"))
      .catch(() => alert("Could not copy to clipboard."));
  }

  function handleCloseAttempt() {
    if (status === "RUNNING") {
      setShowCloseAlert(true);
    } else {
      window.location.hash = "#/runs";
    }
  }

  function handleAlertConfirm() {
    setShowCloseAlert(false);
    window.location.hash = "#/runs";
  }

  function handleAlertCancel() {
    setShowCloseAlert(false);
  }

  return (
    <>
      <Alert
        isOpen={showCloseAlert}
        intent={Intent.WARNING}
        icon="warning-sign"
        confirmButtonText="Leave anyway"
        cancelButtonText="Stay"
        onConfirm={handleAlertConfirm}
        onCancel={handleAlertCancel}
      >
        <p>
          <strong>The simulation is still running.</strong> Leaving this page
          will not cancel it — it will continue running in the background.
        </p>
      </Alert>

      <div style={{ padding: 24, maxWidth: 1100, margin: "0 auto" }}>
        <Card elevation={Elevation.TWO}>
          {/* Header */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 12,
              flexWrap: "wrap",
              gap: 8,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <h3 style={{ margin: 0 }}>Run: {runId}</h3>
              {status && (
                <Tag intent={statusIntent(status)} large>
                  {status}
                </Tag>
              )}
              {status === "RUNNING" && (
                <Spinner size={16} intent={Intent.PRIMARY} />
              )}
            </div>

            <div style={{ display: "flex", gap: 8 }}>
              <Button icon="clipboard" small onClick={handleCopyLog}>
                Copy log
              </Button>
              <Button icon="cross" small onClick={handleCloseAttempt}>
                Close
              </Button>
            </div>
          </div>

          {/* Log output */}
          {error && (
            <div style={{ marginBottom: 8 }}>
              <Tag intent={Intent.DANGER}>{error}</Tag>
            </div>
          )}

          <pre
            ref={preRef}
            style={{
              maxHeight: 600,
              overflow: "auto",
              background: "#1a1a2e",
              color: "#e0e0e0",
              padding: 16,
              borderRadius: 4,
              fontSize: 12,
              lineHeight: 1.5,
              whiteSpace: "pre-wrap",
              wordBreak: "break-all",
              margin: 0,
            }}
          >
            {lines.length === 0 && status === "RUNNING"
              ? "Waiting for output…"
              : lines.join("\n")}
          </pre>
        </Card>
      </div>
    </>
  );
}
