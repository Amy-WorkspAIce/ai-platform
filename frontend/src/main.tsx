import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { takeKeyFromUrl } from "./api";
// Brand foundation, vendored from github.com/Amy-WorkspAIce/brand: fonts, then tokens.
import "./brand/fonts/fonts.css";
import "./brand/tokens.css";
import "./styles.css";

takeKeyFromUrl();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
