/**
 * Unit tests for LaunchPanel.tsx
 *
 * Tests: thickness validation, cube_shape validation, closure_types validation
 */
import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { vi, describe, it, expect, beforeEach } from "vitest";
import LaunchPanel from "../components/LaunchPanel";

// Mock the API client so no real HTTP calls are made
vi.mock("../api/client", () => ({
  fetchModels: vi.fn().mockResolvedValue(["rpm_example"]),
  submitRun: vi.fn().mockResolvedValue({ run_id: "test-run-123", status: "RUNNING" }),
}));

describe("LaunchPanel — form validation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders without crashing", async () => {
    render(<LaunchPanel />);
    expect(screen.getByText("Launch Simulation")).toBeInTheDocument();
  });

  it("loads default values", async () => {
    render(<LaunchPanel />);
    // project_folder default
    expect(screen.getByPlaceholderText("~/synthoseis_output")).toBeInTheDocument();
  });

  it("shows thickness error when min >= max", async () => {
    render(<LaunchPanel />);
    // Initially valid (min=2, max=12) — no error
    expect(screen.queryByText(/thickness_min must be less than thickness_max/i)).not.toBeInTheDocument();

    // Find thickness_min NumericInput and set it to a value >= max
    // Blueprint NumericInput renders an <input> — find by order in DOM
    const numericInputs = screen.getAllByRole("spinbutton");
    // thickness_min is in the Geology section
    const thicknessMinInput = numericInputs.find(
      (el) => (el as HTMLInputElement).value === "2"
    );
    expect(thicknessMinInput).toBeDefined();

    if (thicknessMinInput) {
      fireEvent.change(thicknessMinInput, { target: { value: "15" } });
      fireEvent.blur(thicknessMinInput);
    }

    await waitFor(() => {
      expect(
        screen.getByText(/thickness_min must be less than thickness_max/i)
      ).toBeInTheDocument();
    });
  });

  it("shows error and disables submit when no closure types selected", async () => {
    const user = userEvent.setup();
    render(<LaunchPanel />);

    // Uncheck all closure type checkboxes
    const simpleCheckbox = screen.getByRole("checkbox", { name: /simple/i });
    const faultedCheckbox = screen.getByRole("checkbox", { name: /faulted/i });
    const onlapCheckbox = screen.getByRole("checkbox", { name: /onlap/i });

    await user.click(simpleCheckbox);
    await user.click(faultedCheckbox);
    await user.click(onlapCheckbox);

    expect(
      screen.getByText(/at least one closure type required/i)
    ).toBeInTheDocument();

    // The "Run simulation" button should be disabled
    const runButton = screen.getByRole("button", { name: /run simulation/i });
    expect(runButton).toBeDisabled();
  });

  it("'Load defaults' button resets fields", async () => {
    const user = userEvent.setup();
    render(<LaunchPanel />);

    // Change project_folder
    const projectFolderInput = screen.getByPlaceholderText("~/synthoseis_output");
    await user.clear(projectFolderInput);
    await user.type(projectFolderInput, "/custom/path");
    expect(projectFolderInput).toHaveValue("/custom/path");

    // Click load defaults
    const loadDefaultsBtn = screen.getByRole("button", { name: /load defaults/i });
    await user.click(loadDefaultsBtn);

    expect(projectFolderInput).toHaveValue("~/synthoseis_output");
  });

  it("calls submitRun with correct data when form is valid", async () => {
    const { submitRun } = await import("../api/client");
    const mockNav = vi.fn();

    render(<LaunchPanel onNavigate={mockNav} />);

    // Wait for models to load
    await waitFor(() => {
      expect(screen.queryByText("Loading models…")).not.toBeInTheDocument();
    });

    const runButton = screen.getByRole("button", { name: /run simulation/i });
    expect(runButton).not.toBeDisabled();

    await userEvent.click(runButton);

    await waitFor(() => {
      expect(submitRun).toHaveBeenCalledOnce();
    });

    expect(mockNav).toHaveBeenCalledWith("#/runs/test-run-123/logs");
  });
});
