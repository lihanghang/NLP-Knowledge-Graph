// Run against a built preview and a disposable Chrome with --remote-debugging-port.
// READER_BASE_URL=http://127.0.0.1:4325 CHROME_DEBUG_URL=http://127.0.0.1:9224 node scripts/check_reader.mjs
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
const base = process.env.READER_BASE_URL || 'http://127.0.0.1:4325';
const debug = process.env.CHROME_DEBUG_URL || 'http://127.0.0.1:9224';
const output = process.env.READER_SCREENSHOTS || '/private/tmp/kg-pdf-reader-check';
await mkdir(output, { recursive: true });
const manifest = await (await fetch(`${base}/read/papers.json`)).json();
assert.ok(Object.keys(manifest).length > 0);
const cases = [
  ['A Translation Approach', 27],
  ['Bag of Tricks', 5],
  ['BERT(中文翻译)', 23],
  ['Hierarchical Attention Networks', 10],
  ['As We May Think', 9],
].map(([title, pages]) => {
  const entry = Object.entries(manifest).find(([, paper]) => paper.title.includes(title));
  assert.ok(entry, `Missing test paper: ${title}`);
  return { id: entry[0], ...entry[1], pages };
});
assert.equal((await fetch(`${base}/read/samples/`)).status, 404);
assert.equal((await fetch(`${base}/reading/pilot.json`)).status, 404);
const version = await (await fetch(`${debug}/json/version`)).json();
const socket = new WebSocket(version.webSocketDebuggerUrl);
await new Promise(resolve => socket.addEventListener('open', resolve, { once: true }));
let serial = 0; const pending = new Map();
socket.addEventListener('message', event => {
  const m = JSON.parse(event.data), handler = pending.get(m.id);
  if (handler) { pending.delete(m.id); m.error ? handler.reject(Error(JSON.stringify(m.error))) : handler.resolve(m.result); }
});
function call(method, params = {}, sessionId) {
  return new Promise((resolve, reject) => {
    const id = ++serial;
    const timeout = setTimeout(() => { pending.delete(id); reject(Error(`Chrome did not respond: ${method}`)); }, 15000);
    pending.set(id, { resolve: value => { clearTimeout(timeout); resolve(value); }, reject: error => { clearTimeout(timeout); reject(error); } });
    socket.send(JSON.stringify({ id, method, params, sessionId }));
  });
}
const { targetId } = await call('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await call('Target.attachToTarget', { targetId, flatten: true });
const send = (method, params = {}) => call(method, params, sessionId);
const evaluate = async expression => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true, userGesture: true });
  if (r.exceptionDetails) throw Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
};
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(expression) {
  for (let i = 0; i < 160; i++) { if (await evaluate(expression)) return; await pause(200); }
  throw Error(`Timed out: ${expression}; page=${await evaluate('document.body.innerText')}`);
}
async function navigate(path) { await send('Page.navigate', { url: base + path }); await until('document.readyState === "complete"'); }
async function reload() {
  const previous = await evaluate('performance.timeOrigin');
  await send('Page.reload');
  await until(`performance.timeOrigin !== ${previous} && document.readyState === 'complete'`);
  await ready();
}
async function ready() { await until('document.getElementById("page-input") && !document.getElementById("page-input").disabled'); }
const click = id => evaluate(`document.getElementById(${JSON.stringify(id)}).click()`);
async function screenshot(name) { const shot = await send('Page.captureScreenshot', { format: 'png' }); await writeFile(`${output}/${name}.png`, Buffer.from(shot.data, 'base64')); }
async function layout(width) {
  const state = await evaluate(`({width:innerWidth,overflow:document.documentElement.scrollWidth>innerWidth,outerScroll:document.documentElement.scrollHeight>innerHeight+1,controls:[...document.querySelectorAll('button,input,select,a.icon-button')].filter(e=>!e.closest('dialog')&&!e.closest('[hidden]')&&!e.classList.contains('sr-only')&&e.getBoundingClientRect().width).map(e=>({id:e.id,w:e.getBoundingClientRect().width,h:e.getBoundingClientRect().height}))})`);
  assert.equal(state.width, width); assert.equal(state.overflow, false); assert.equal(state.outerScroll, false);
  for (const c of state.controls) assert.ok(c.w >= 44 && c.h >= 44, `${c.id} touch target ${c.w}×${c.h}`);
}
try {
  await send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
  await send('Emulation.setTouchEmulationEnabled', { enabled: true });
  await navigate('/read/');
  await until('!document.getElementById("error").hidden');
  await evaluate('Object.keys(localStorage).filter(k=>k.startsWith("kg-reader:")).forEach(k=>localStorage.removeItem(k))');
  await navigate('/read/?paper=0000000000000000');
  await until('document.getElementById("error-message").textContent.includes("未找到")');
  for (const [i, paper] of cases.entries()) {
    await navigate(`/read/?paper=${paper.id}`); await ready();
    await until('document.getElementById("pdf-frame").contentDocument.querySelector(".page canvas")?.width>0');
    assert.equal(await evaluate('document.getElementById("page-count").textContent'), `/ ${paper.pages}`);
    assert.equal(await evaluate('Boolean(document.querySelector("#text-mode, #text-view, #font-controls"))'), false);
    assert.equal(await evaluate('getComputedStyle(document.getElementById("pdf-frame").contentDocument.querySelector(".toolbar")).display'), 'none');
    await layout(390);
    await click('next'); await until('document.getElementById("page-input").value === "2"');
    await pause(400); await reload();
    assert.equal(await evaluate('document.getElementById("page-input").value'), '2', 'PDF resume');
    await screenshot(`mobile-${i}`);
    const scale = await evaluate('document.getElementById("pdf-frame").contentWindow.PDFViewerApplication.pdfViewer.currentScale');
    await click('zoom-in');
    assert.ok(await evaluate(`document.getElementById('pdf-frame').contentWindow.PDFViewerApplication.pdfViewer.currentScale > ${scale}`));
    await click('zoom-out'); await click('fit-width');
    assert.equal(await evaluate('document.getElementById("pdf-frame").contentWindow.PDFViewerApplication.pdfViewer.currentScaleValue'), 'page-width');
    await evaluate(`document.getElementById('page-input').value='${paper.pages}';document.getElementById('page-form').requestSubmit()`);
    await until(`document.getElementById('pdf-frame').contentWindow.PDFViewerApplication.page === ${paper.pages}`);
    assert.equal(await evaluate('document.getElementById("next").disabled'), true);
    await click('contents-open'); await until('document.querySelectorAll("#outline button").length>0');
    await evaluate('document.querySelector("#outline button").click()');
    assert.equal(await evaluate('document.getElementById("contents").open'), false);
    await click('menu-open'); await click('restart');
    assert.equal(await evaluate('document.getElementById("page-input").value'), '1');
    console.log(`PASS ${paper.title}: navigation, resume, zoom, outline, restart, touch targets`);
  }
  await navigate(cases[0].details.replace('?details=1', ''));
  await until('location.pathname === "/read/"'); await ready();
  await click('back'); await until('location.search.includes("details=1")');
  assert.ok(await evaluate('Boolean(document.querySelector("[data-reader-link]"))'));
  for (const width of [320, 1440]) {
    await send('Emulation.setDeviceMetricsOverride', { width, height: width === 320 ? 740 : 900, deviceScaleFactor: 1, mobile: width === 320 });
    await navigate(`/read/?paper=${cases[2].id}&page=2`); await ready(); await pause(300);
    await layout(width); await screenshot(`reader-${width}`);
  }
  await navigate(`/read/?paper=${cases[2].id}&page=1`); await ready();
  await click('next'); await pause(400); await reload();
  assert.equal(await evaluate('document.getElementById("page-input").value'), '2', 'deep link must not override later progress');
  await evaluate(`void Object.defineProperty(window, 'localStorage', { configurable: true, get() { throw new DOMException('Storage unavailable', 'SecurityError'); } })`);
  await click('next');
  await until('document.getElementById("progress-status").textContent === "本次阅读"');
  assert.equal(await evaluate('document.getElementById("error").hidden'), true);
  console.log(`PASS mobile entry/back, 320/390/1440px, deep-link resume, disabled storage. Screenshots: ${output}`);
} finally { await call('Target.closeTarget', { targetId }); socket.close(); }
