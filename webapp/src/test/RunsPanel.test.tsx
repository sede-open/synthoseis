/**
 * Unit tests for RunsPanel.tsx
 *
 * Tests: polling, status tags, action buttons
 */
import React from "react";
import { render, screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import RunsPanel from "../components/RunsPanel";
import type { RunRecord } from "../types/simulation";

// Top-level mock — vi.mock is hoisted, so these stubs are available before imports run
const mockListRuns = vi.fn();
const mockDeleteRun = vi.fn().mockResolvedValue(undefined);

vi.mock("../api/client", () => ({
  listRuns: (...args: unknown[]) => mockListRuns(...args),
  deleteRun: (...args: unknown[]) => mockDeleteRun(...args),
}));

const mockRuns: RunRecord[] = [
  {
    run_id: "run-001",
    status: "RUNNING",
    config: {} as any,
    started_at: "2026-05-17T10:00:00",
    ended_at: null,
    output_folder: null,
  },
  {
    run_id: "run-002",
    status: "COMPLETE",
    config: {} as any,
    started_at: "2026-05-17T09:00:00",
    ended_at: "2026-05-17T09:30:00",
    output_folder: "/tmp/results/my_run",
  },
  {
    run_id: "run-003",
    status: "FAILED",
    config: {} as any,
    started_at: "2026-05-17T08:00:00",
    ended_at: "2026-05-17T08:05:00",
    output_folder: null,
  },
];

describe("RunsPanel", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("shows empty state when no runs", async () => {
    mockListRuns.mockResolvedValue([]);

    await act(async () => {
      render(<RunsPanel />);
    });

    await waitFor(
      () => {
        expect(screen.getByText(/no simulation runs yet/i)).toBeInTheDocument();
      },
      { timeout: 3000 }
    );
  });

  it("renders runs table with status tags", async () => {
    mockListRuns.mockResolvedValue(mockRuns);

    await act(async () => {
      render(<RunsPanel />);
    });

    await waitFor(
      () => {
        expect(screen.getByText("RUNNING")).toBeInTheDocument();
        expect(screen.getByText("COMPLETE")).toBeInTheDocument();
        expect(screen.getByText("FAILED")).toBeInTheDocument();
      },
      { timeout: 3000 }
    );
  });

  it("shows Cancel button only for RUNNING runs", async () => {
    mockListRuns.mockResolvedValue(mockRuns);

    await act(async () => {
      render(<RunsPanel />);
    });

    await waitFor(
      () => {
        expect(screen.getByText("RUNNING")).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    const cancelButtons = screen.getAllByRole("button", { name: /cancel/i });
    expect(cancelButtons).toHaveLength(1);
  });

  it("shows 'Open in viewer' button only for COMPLETE runs with output", async () => {
    mockListRuns.mockResolvedValue(mockRuns);

    await act(async () => {
      render(<RunsPanel />);
    });

    await waitFor(
      () => {
        expect(screen.getByText("COMPLETE")).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    // Blueprint AnchorButton renders <a> with role="button"
    const viewerButtons = screen.getAllByRole("button", { name: /open in viewer/i });
    expect(viewerButtons).toHaveLength(1);
    expect(viewerButtons[0].closest("a")).toHaveAttribute("href", "#/run/my_run");
  });

  it("shows 'View logs' button for all runs", async () => {
    mockListRuns.mockResolvedValue(mockRuns);

    await act(async () => {
      render(<RunsPanel />);
    });

    await waitFor(
      () => {
        // Blueprint AnchorButton renders <a> with role="button"
        const logButtons = screen.getAllByRole("button", { name: /view logs/i });
        expect(logButtons).toHaveLength(mockRuns.length);
      },
      { timeout: 3000 }
    );
  });

  it("calls deleteRun when Cancel is clicked", async () => {
    mockListRuns.mockResolvedValue(mockRuns);

    await act(async () => {
      render(<RunsPanel />);
    });

    await waitFor(
      () => {
        expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    await userEvent.click(screen.getByRole("button", { name: /cancel/i }));

    expect(mockDeleteRun).toHaveBeenCalledWith("run-001");
  });

  it("polls again after 5 seconds", async () => {
    mockListRuns.mockResolvedValue([]);

    vi.useFakeTimers({ shouldAdvanceTime: true });

    await act(async () => {
      render(<RunsPanel />);
    });

    // Initial call
    expect(mockListRuns).toHaveBeenCalledTimes(1);

    // Advance by 5 seconds
    await act(async () => {
      vi.advanceTimersByTime(5000);
      // Give React a tick to process
      await Promise.resolve();
    });

    expect(mockListRuns.mock.calls.length).toBeGreaterThanOrEqual(2);

    vi.useRealTimers();
  });
});
