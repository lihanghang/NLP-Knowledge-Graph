export const ENDPOINT = 'https://kg-stats.lihanghang.top/v1/view';
const SITE_ORIGIN = 'https://kg.lihanghang.top';
const VISITOR_KEY = 'kg-analytics:visitor:v1';

export function isReaderVisible(win) {
  if (win.document.visibilityState !== 'visible') return false;
  try {
    const frame = win.frameElement;
    if (!frame) return true;
    const rect = frame.getBoundingClientRect();
    const visibleHeight = Math.max(0, Math.min(rect.bottom, win.parent.innerHeight) - Math.max(rect.top, 0));
    const visibleWidth = Math.max(0, Math.min(rect.right, win.parent.innerWidth) - Math.max(rect.left, 0));
    return rect.width > 0 && rect.height > 0 && visibleHeight * visibleWidth >= rect.height * rect.width * 0.25;
  } catch { return false; }
}

// Called only after a PDF page is actually rendered. No tracking on local
// previews, failed PDF loads, background tabs, or readers scrolled out of view.
export function startViewTracking(paperId, win = window, endpoint = ENDPOINT, allowedOrigin = SITE_ORIGIN) {
  if (!/^[a-f0-9]{16}$/.test(paperId || '') || win.location.origin !== allowedOrigin ||
      win.navigator.globalPrivacyControl || win.navigator.doNotTrack === '1') return () => {};
  const eventKey = `kg-analytics:next:${paperId}`;
  let visitorId, nextEligible;
  try {
    visitorId = win.localStorage.getItem(VISITOR_KEY);
    if (!/^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i.test(visitorId || '')) {
      visitorId = win.crypto.randomUUID();
      win.localStorage.setItem(VISITOR_KEY, visitorId);
    }
    nextEligible = Number(win.localStorage.getItem(eventKey)) || 0;
  } catch {
    // Without browser storage we cannot deduplicate reliably; skip counting.
    return () => {};
  }
  if (Date.now() < nextEligible) return () => {};
  let elapsed = 0, last = win.performance.now(), stopped = false;
  const stop = () => {
    stopped = true; win.clearInterval(timer);
    win.removeEventListener('pagehide', stop);
  };
  const send = async () => {
    // Other tabs may already have counted this paper during the five-second wait.
    try { if (Date.now() < Number(win.localStorage.getItem(eventKey))) return; } catch { return; }
    for (let attempt = 0; attempt < 2 && !stopped; attempt++) {
      try {
        const response = await win.fetch(endpoint, {
          method: 'POST', credentials: 'omit', referrerPolicy: 'no-referrer',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ paperId, visitorId }), signal: AbortSignal.timeout(5000),
        });
        if (!response.ok) {
          if (response.status >= 500) throw new Error('Unavailable');
          return;
        }
        const result = await response.json();
        if (Number.isFinite(result.nextEligibleAt)) win.localStorage.setItem(eventKey, String(result.nextEligibleAt));
        if (Number.isSafeInteger(result.totalViews) && result.totalViews >= 0) {
          win.parent?.postMessage?.({ type: 'kg:paper-views', paperId, totalViews: result.totalViews }, win.location.origin);
        }
        return;
      } catch {
        if (attempt === 0) await new Promise(resolve => win.setTimeout(resolve, 2000));
      }
    }
  };
  const timer = win.setInterval(() => {
    const now = win.performance.now();
    // Cap gaps so a throttled/suspended tab never earns minutes of visibility.
    if (isReaderVisible(win)) elapsed += Math.min(now - last, 500);
    last = now;
    if (elapsed >= 5000) {
      win.clearInterval(timer);
      void send().finally(stop);
    }
  }, 250);
  win.addEventListener('pagehide', stop, { once: true });
  return stop;
}
