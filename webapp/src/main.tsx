import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Blueprint CSS loaded from CDN links in index.html — do not re-import here.

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
