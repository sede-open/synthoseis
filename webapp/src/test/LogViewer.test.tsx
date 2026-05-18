/**
 * Unit tests for LogViewer.tsx
 *
 * Tests: SSE event handling, auto-scroll, status updates, close alert
 */
import React from "react";
import { render, screen, act, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import LogViewer from "../components/LogViewer";

// ---------------------------------------------------------------------------
// Mock EventSource
// ---------------------------------------------------------------------------

type MockEventSourceInstance = {
  onmessage: ((e: MessageEvent) => void) | null;
  onerror: ((e: Event) => void) | null;
  _listeners: Record<string, ((e: Event) => void)[]>;
  close: () => void;
  addEventListener: (type: string, handler: (e: Event) => void) => void;
  dispatchMessage: (data: string) => void;
  dispatchStatus: (status: string) => void;
  dispatchError: () => void;
};

let mockEventSourceInstance: MockEventSourceInstance | null = null;

class MockEventSource {
  onmessage: ((e: MessageEvent) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  _listeners: Record<string, ((e: Event) => void)[]> = {};

  constructor(_url: string) {
    mockEventSourceInstance = this as unknown as MockEventSourceInstance;
  }

  close() {
    // no-op in mock
  }

  addEventListener(type: string, handler: (e: Event) => void) {
    if (!this._listeners[type]) this._listeners[type] = [];
    this._listeners[type].push(handler);
  }

  dispatchMessage(data: string) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent("message", { data }));
    }
  }

  dispatchStatus(status: string) {
    const handlers = this._listeners["status"] ?? [];
    handlers.forEach((h) => h(new MessageEvent("status", { data: status })));
  }

  dispatchError() {
    if (this.onerror) {
      this.onerror(new Event("error"));
    }
  }
}

vi.mock("../api/client", () => ({
  streamLogs: vi.fn((runId: string) => new MockEventSource(`/api/runs/${runId}/logs`)),
}));

describe("LogViewer", () => {
  beforeEach(() => {
    mockEventSourceInstance = null;
    vi.clearAllMocks();
    // Mock clipboard
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  it("renders the run ID in the header", () => {
    render(<LogViewer runId="test-run-abc" />);
    expect(screen.getByText(/run: test-run-abc/i)).toBeInTheDocument();
  });

  it("appends incoming SSE message lines", async () => {
    render(<LogViewer runId="test-run-abc" />);

    await act(async () => {
      mockEventSourceInstance?.dispatchMessage("Line 1 of output");
      mockEventSourceInstance?.dispatchMessage("Line 2 of output");
    });

    expect(screen.getByText(/Line 1 of output[\s\S]*Line 2 of output/)).toBeInTheDocument();
  });

  it("updates status tag on SSE status event", async () => {
    render(<LogViewer runId="test-run-abc" />);

    // Initially RUNNING
    expect(screen.getByText("RUNNING")).toBeInTheDocument();

    await act(async () => {
      mockEventSourceInstance?.dispatchStatus("COMPLETE");
    });

    await waitFor(() => {
      expect(screen.getByText("COMPLETE")).toBeInTheDocument();
    });
  });

  it("shows error tag when SSE connection fails", async () => {
    render(<LogViewer runId="test-run-abc" />);

    await act(async () => {
      mockEventSourceInstance?.dispatchError();
    });

    await waitFor(() => {
      expect(screen.getByText(/SSE connection lost/i)).toBeInTheDocument();
    });
  });

  it("shows confirmation alert when closing while RUNNING", async () => {
    render(<LogViewer runId="test-run-abc" />);

    // Status is RUNNING
    expect(screen.getByText("RUNNING")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /close/i }));

    // Alert should appear
    await waitFor(() => {
      expect(
        screen.getByText(/the simulation is still running/i)
      ).toBeInTheDocument();
    });
  });

  it("does NOT show confirmation alert when closing after COMPLETE", async () => {
    render(<LogViewer runId="test-run-abc" />);

    // Simulate completion
    await act(async () => {
      mockEventSourceInstance?.dispatchStatus("COMPLETE");
    });

    // No alert should appear when clicking Close
    const originalHash = window.location.hash;
    await userEvent.click(screen.getByRole("button", { name: /close/i }));

    // Alert should NOT be shown
    expect(
      screen.queryByText(/the simulation is still running/i)
    ).not.toBeInTheDocument();
  });

  it("copies log content to clipboard", async () => {
    render(<LogViewer runId="test-run-abc" />);

    await act(async () => {
      mockEventSourceInstance?.dispatchMessage("log line 1");
    });

    await userEvent.click(screen.getByRole("button", { name: /copy log/i }));

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("log line 1");
  });
});
