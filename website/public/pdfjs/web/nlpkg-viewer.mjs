/* Site-specific integration layered on the unmodified PDF.js Generic Viewer. */
import { PDFViewerApplication } from "./viewer.mjs";
import { startViewTracking } from '../../analytics/paper-views.mjs';

const filename = new URLSearchParams(location.search).get("filename");
if (new URLSearchParams(location.search).get('host') === 'reader') {
  document.documentElement.classList.add('kg-reader');
}

if (filename) {
  // GitHub Pages serves the source as .pdf.bin so mobile browsers do not
  // intercept it. Preserve the original .pdf name for the viewer download.
  PDFViewerApplication._contentDispositionFilename = filename;
  document.title = filename;
}

const paperId = new URLSearchParams(location.search).get('paper');
PDFViewerApplication.initializedPromise.then(() => {
  const onRendered = ({ error }) => {
    if (error) return;
    PDFViewerApplication.eventBus.off('pagerendered', onRendered);
    startViewTracking(paperId);
  };
  PDFViewerApplication.eventBus.on('pagerendered', onRendered);
}).catch(() => { /* Statistics must never block the reader. */ });
