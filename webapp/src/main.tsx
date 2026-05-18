import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Blueprint CSS — imported via Vite (bundled, version-locked, correct load order).
// Order matters per https://blueprintjs.com/docs/#blueprint/getting-started:
//   1. blueprint.css        — base component styles
//   2. blueprint-icons.css  — icon font faces (missing this = blank icons)
//   3. blueprint-select.css — Select / Select2 / MultiSelect styles
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/icons/lib/css/blueprint-icons.css";
import "@blueprintjs/select/lib/css/blueprint-select.css";
// Global baseline: sets dark background-color on body/html so areas not
// covered by Blueprint components don't show through as white.
import "./index.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
