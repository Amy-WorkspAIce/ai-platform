import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
// Brand foundation, vendored from github.com/Amy-WorkspAIce/brand: fonts, then tokens.
import "./brand/fonts/fonts.css";
import "./brand/tokens.css";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
