/* Site-specific integration layered on the unmodified PDF.js Generic Viewer. */
import { PDFViewerApplication } from "./viewer.mjs";

const filename = new URLSearchParams(location.search).get("filename");

if (filename) {
  // GitHub Pages serves the source as .pdf.bin so mobile browsers do not
  // intercept it. Preserve the original .pdf name for the viewer download.
  PDFViewerApplication._contentDispositionFilename = filename;
  document.title = filename;
}
