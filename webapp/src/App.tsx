import React from "react";
import {
  Alignment,
  AnchorButton,
  Navbar,
  NavbarDivider,
  NavbarGroup,
  NavbarHeading,
} from "@blueprintjs/core";
import ProjectDashboard from "./components/ProjectDashboard";
import RunViewer from "./components/RunViewer";
import LaunchPanel from "./components/LaunchPanel";
import RunsPanel from "./components/RunsPanel";
import LogViewer from "./components/LogViewer";

/**
 * Hash-based router:
 *   #/                       → ProjectDashboard
 *   #/run/:folderId          → RunViewer
 *   #/launch                 → LaunchPanel
 *   #/runs                   → RunsPanel
 *   #/runs/:runId/logs       → LogViewer
 */
function App(): React.ReactElement {
  const [hash, setHash] = React.useState(window.location.hash);

  React.useEffect(() => {
    const handler = () => setHash(window.location.hash);
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  // Parse #/runs/:runId/logs
  const logsMatch = hash.match(/^#\/runs\/([^/]+)\/logs$/);
  // Parse #/run/:folderId (original viewer route)
  const runMatch = hash.match(/^#\/run\/(.+)$/);

  function renderContent(): React.ReactElement {
    if (logsMatch) {
      const runId = decodeURIComponent(logsMatch[1]);
      return <LogViewer runId={runId} />;
    }
    if (hash === "#/launch") {
      return <LaunchPanel />;
    }
    if (hash === "#/runs") {
      return <RunsPanel />;
    }
    if (runMatch) {
      const folderId = decodeURIComponent(runMatch[1]);
      return <RunViewer folderId={folderId} />;
    }
    // Default
    return <ProjectDashboard />;
  }

  return (
    <>
      <Navbar>
        <NavbarGroup align={Alignment.LEFT}>
          <NavbarHeading>Synthoseis</NavbarHeading>
          <NavbarDivider />
        </NavbarGroup>
        <NavbarGroup align={Alignment.RIGHT}>
          <AnchorButton href="#/" minimal icon="eye-open">
            Viewer
          </AnchorButton>
          <AnchorButton href="#/runs" minimal icon="history">
            Runs
          </AnchorButton>
          <AnchorButton href="#/launch" minimal icon="play">
            Launch
          </AnchorButton>
        </NavbarGroup>
      </Navbar>

      {renderContent()}
    </>
  );
}

export default App;
