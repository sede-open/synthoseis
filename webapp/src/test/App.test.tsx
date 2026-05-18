/**
 * Tests for App.tsx — hash-based router.
 *
 * Every child component is replaced with a lightweight stub so this file
 * only tests routing logic, not component implementation.
 */
import React from "react";
import { render, screen, act } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";

// Stub every routed component — must be declared before importing App.
vi.mock("../components/ProjectDashboard", () => ({
  default: () => <div data-testid="project-dashboard" />,
}));
vi.mock("../components/RunViewer", () => ({
  default: ({ folderId }: { folderId: string }) => (
    <div data-testid="run-viewer" data-folder-id={folderId} />
  ),
}));
vi.mock("../components/LaunchPanel", () => ({
  default: () => <div data-testid="launch-panel" />,
}));
vi.mock("../components/RunsPanel", () => ({
  default: () => <div data-testid="runs-panel" />,
}));
vi.mock("../components/LogViewer", () => ({
  default: ({ runId }: { runId: string }) => (
    <div data-testid="log-viewer" data-run-id={runId} />
  ),
}));

import App from "../App";

describe("App — hash-based router", () => {
  afterEach(() => {
    window.location.hash = "";
  });

  it("renders ProjectDashboard for empty hash", () => {
    window.location.hash = "";
    render(<App />);
    expect(screen.getByTestId("project-dashboard")).toBeInTheDocument();
  });

  it("renders ProjectDashboard for #/", () => {
    window.location.hash = "#/";
    render(<App />);
    // hashchange not fired during render — re-render reflects initial hash
    expect(screen.getByTestId("project-dashboard")).toBeInTheDocument();
  });

  it("renders LaunchPanel for #/launch", () => {
    window.location.hash = "#/launch";
    render(<App />);
    expect(screen.getByTestId("launch-panel")).toBeInTheDocument();
  });

  it("renders RunsPanel for #/runs", () => {
    window.location.hash = "#/runs";
    render(<App />);
    expect(screen.getByTestId("runs-panel")).toBeInTheDocument();
  });

  it("renders LogViewer for #/runs/:runId/logs with decoded runId", () => {
    window.location.hash = "#/runs/my-run-abc-123/logs";
    render(<App />);
    const viewer = screen.getByTestId("log-viewer");
    expect(viewer).toBeInTheDocument();
    expect(viewer).toHaveAttribute("data-run-id", "my-run-abc-123");
  });

  it("renders RunViewer for #/run/:folderId with decoded folderId", () => {
    window.location.hash = "#/run/seismic__20260517_abc";
    render(<App />);
    const viewer = screen.getByTestId("run-viewer");
    expect(viewer).toBeInTheDocument();
    expect(viewer).toHaveAttribute("data-folder-id", "seismic__20260517_abc");
  });

  it("switches to LaunchPanel on hashchange event", () => {
    window.location.hash = "#/runs";
    render(<App />);
    expect(screen.getByTestId("runs-panel")).toBeInTheDocument();

    act(() => {
      window.location.hash = "#/launch";
      window.dispatchEvent(new Event("hashchange"));
    });

    expect(screen.getByTestId("launch-panel")).toBeInTheDocument();
  });

  it("renders Synthoseis nav with Viewer / Runs / Launch links", () => {
    window.location.hash = "";
    render(<App />);
    expect(screen.getByText("Synthoseis")).toBeInTheDocument();
    // Blueprint AnchorButton renders as <a role="button">, not role="link"
    const viewer = screen.getByRole("button", { name: /viewer/i });
    const runs   = screen.getByRole("button", { name: /^runs$/i });
    const launch = screen.getByRole("button", { name: /launch/i });
    expect(viewer.closest("a")).toHaveAttribute("href", "#/");
    expect(runs.closest("a")).toHaveAttribute("href", "#/runs");
    expect(launch.closest("a")).toHaveAttribute("href", "#/launch");
  });
});
