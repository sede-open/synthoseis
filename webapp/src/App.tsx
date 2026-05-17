import React from "react";
import ProjectDashboard from "./components/ProjectDashboard";
import RunViewer from "./components/RunViewer";

/**
 * Hash-based router:
 *   #/              → ProjectDashboard
 *   #/run/:folderId → RunViewer
 */
function App(): React.ReactElement {
  const [hash, setHash] = React.useState(window.location.hash);

  React.useEffect(() => {
    const handler = () => setHash(window.location.hash);
    window.addEventListener("hashchange", handler);
    return () => window.removeEventListener("hashchange", handler);
  }, []);

  // Parse #/run/:folderId
  const runMatch = hash.match(/^#\/run\/(.+)$/);
  if (runMatch) {
    const folderId = decodeURIComponent(runMatch[1]);
    return <RunViewer folderId={folderId} />;
  }

  // Default → project dashboard
  return <ProjectDashboard />;
}

export default App;
