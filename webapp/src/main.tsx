import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Blueprint CSS is loaded via index.html CDN links.
// Import Blueprint's normalize reset so font sizes apply correctly.
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/select/lib/css/blueprint-select.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
