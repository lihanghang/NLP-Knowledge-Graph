import test from 'node:test';
import assert from 'node:assert/strict';
import { startViewTracking, isReaderVisible } from '../../../website/public/analytics/paper-views.mjs';
const id = 'f8d56fb77b411557';
const flush = () => new Promise(resolve => setImmediate(resolve));
function browser() {
  let clock = 0, callback;
  const storage = new Map(), calls = [];
  const win = {
    document: { visibilityState: 'visible' }, navigator: {}, location: { origin: 'https://kg.lihanghang.top' },
    performance: { now: () => clock }, crypto,
    localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    setInterval: fn => { callback = fn; return 1; }, clearInterval: () => { callback = null; },
    setTimeout: fn => { fn(); }, addEventListener() {}, removeEventListener() {},
    fetch: async (_url, request) => { calls.push(JSON.parse(request.body)); return Response.json({ nextEligibleAt: Date.now() + 1800000 }); },
  };
  return { win, calls, storage, tick(n) { for (let i = 0; i < n; i++) { clock += 250; callback?.(); } } };
}
test('five visible seconds; refresh cooldown; same anonymous ID shared by readers', async () => {
  const b = browser(); startViewTracking(id, b.win);
  b.tick(19); await flush(); assert.equal(b.calls.length, 0);
  b.tick(1); await flush(); assert.equal(b.calls.length, 1);
  startViewTracking(id, b.win); b.tick(40); await flush(); assert.equal(b.calls.length, 1);
  assert.equal(b.calls[0].visitorId, b.storage.get('kg-analytics:visitor:v1'));
});
test('hidden tab and offscreen iframe never count; visible dwell resumes', async () => {
  const b = browser(); b.win.document.visibilityState = 'hidden'; startViewTracking(id, b.win);
  b.tick(100); await flush(); assert.equal(b.calls.length, 0);
  b.win.document.visibilityState = 'visible';
  b.win.parent = { innerHeight: 800, innerWidth: 390 };
  b.win.frameElement = { getBoundingClientRect: () => ({ top: 900, bottom: 1700, left: 0, right: 390, width: 390, height: 800 }) };
  assert.equal(isReaderVisible(b.win), false); b.tick(100); await flush(); assert.equal(b.calls.length, 0);
  b.win.frameElement = null; b.tick(20); await flush(); assert.equal(b.calls.length, 1);
});
test('privacy settings, local previews, missing storage, and cancelled reads skip counting', async () => {
  for (const configure of [
    b => b.win.navigator.globalPrivacyControl = true,
    b => b.win.navigator.doNotTrack = '1',
    b => b.win.location.origin = 'http://localhost:4325',
    b => Object.defineProperty(b.win, 'localStorage', { get() { throw Error('blocked'); } }),
  ]) { const b = browser(); configure(b); startViewTracking(id, b.win); b.tick(40); await flush(); assert.equal(b.calls.length, 0); }
  const b = browser(), stop = startViewTracking(id, b.win); stop(); b.tick(40); await flush(); assert.equal(b.calls.length, 0);
});
test('network failure retries at most once and does not mark a failed view as counted', async () => {
  const b = browser(); let attempts = 0;
  b.win.fetch = async () => { attempts++; throw Error('offline'); };
  startViewTracking(id, b.win); b.tick(20); await flush(); await flush();
  assert.equal(attempts, 2); assert.equal(b.storage.has(`kg-analytics:next:${id}`), false);
});
