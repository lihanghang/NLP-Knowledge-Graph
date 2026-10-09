type Paper = { title: string; file: string; filename: string; details: string };
type Saved = { page?: number; pdfTop?: number; pdfLeft?: number; scale?: number | string };
const element = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const button = (id: string) => element<HTMLButtonElement>(id);
const frame = element<HTMLIFrameElement>('pdf-frame');
const pageInput = element<HTMLInputElement>('page-input');
const params = new URLSearchParams(location.search);
const id = params.get('paper') || '';
const storageKey = `kg-reader:v1:${id}`;
let storageAvailable = true;
function readStorage(key: string) {
  try { return JSON.parse(localStorage.getItem(key) || '{}'); } catch { return {}; }
}
const raw = readStorage(storageKey);
const saved: Saved = raw && typeof raw === 'object' ? raw : {};
let page = 1, total = 0, ready = false;
// PDF.js is a vendored application with a same-origin API, not a native PDF embed.
let app: any;
let saveTimer: ReturnType<typeof setTimeout>;
let restoring = true;
function save() {
  if (!ready || restoring) return;
  const loc = app.pdfViewer._location;
  const record: Saved = { page, scale: app.pdfViewer.currentScaleValue };
  if (loc?.pageNumber === page) { record.pdfTop = loc.top; record.pdfLeft = loc.left; }
  try { localStorage.setItem(storageKey, JSON.stringify(record)); }
  catch { storageAvailable = false; }
  element('progress-status').textContent = storageAvailable ? '进度已保存' : '本次阅读';
}
function queueSave() { clearTimeout(saveTimer); saveTimer = setTimeout(save, 250); }
function updateControls() {
  pageInput.value = String(page); pageInput.max = String(total);
  element('page-count').textContent = `/ ${total}`;
  button('previous').disabled = !ready || page <= 1;
  button('next').disabled = !ready || page >= total;
}
function goToPage(value: number) {
  if (!ready) return;
  page = Math.max(1, Math.min(total, Math.trunc(value) || page));
  app.pdfViewer.scrollPageIntoView({ pageNumber: page });
  updateControls(); queueSave();
}
function failure(message: string) {
  ready = false; element('loading').hidden = true; element('error').hidden = false;
  element('error-message').textContent = message;
  element('progress-status').textContent = '加载失败';
  ['zoom-in', 'zoom-out', 'contents-open', 'menu-open', 'fit-width'].forEach(id => button(id).disabled = true);
  pageInput.disabled = true; updateControls();
}
function dialog(id: string) { return element<HTMLDialogElement>(id); }
button('menu-open').onclick = () => dialog('settings').showModal();
button('contents-open').onclick = () => dialog('contents').showModal();
document.querySelectorAll<HTMLButtonElement>('[data-close]').forEach(b => b.onclick = () => dialog(b.dataset.close!).close());
document.querySelectorAll('dialog').forEach(d => d.addEventListener('click', e => { if (e.target === d) { const r = d.getBoundingClientRect(); if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) d.close(); } }));
button('previous').onclick = () => goToPage(page - 1);
button('next').onclick = () => goToPage(page + 1);
element('page-form').onsubmit = e => { e.preventDefault(); goToPage(Number(pageInput.value)); pageInput.blur(); };
pageInput.onchange = () => goToPage(Number(pageInput.value));
button('fit-width').onclick = () => app.pdfViewer.currentScaleValue = 'page-width';
button('zoom-in').onclick = () => app.zoomIn();
button('zoom-out').onclick = () => app.zoomOut();
button('restart').onclick = () => { goToPage(1); dialog('settings').close(); };
button('retry').onclick = () => location.reload();
window.addEventListener('pagehide', save);
document.addEventListener('visibilitychange', () => { if (document.hidden) save(); });

async function buildOutline() {
  const list = element('outline');
  try {
    const outline = await app.pdfDocument.getOutline();
    const entries: { title: string; page: number; depth: number }[] = [];
    async function visit(items: any[], depth = 0) {
      for (const item of items || []) {
        try {
          const dest = typeof item.dest === 'string' ? await app.pdfDocument.getDestination(item.dest) : item.dest;
          if (dest) {
            const index = typeof dest[0] === 'number' ? dest[0] : await app.pdfDocument.getPageIndex(dest[0]);
            entries.push({ title: item.title, page: index + 1, depth });
          }
        } catch { /* An invalid destination should not hide the rest of the outline. */ }
        await visit(item.items, depth + 1);
      }
    }
    await visit(outline);
    if (!entries.length) {
      element('outline-note').textContent = '这份 PDF 没有内置目录，可按页跳转。';
      for (let n = 1; n <= total; n++) entries.push({ title: `第 ${n} 页`, page: n, depth: 0 });
    } else element('outline-note').textContent = '使用论文内置目录，页码对应原版 PDF。';
    for (const entry of entries) {
      const li = document.createElement('li'), b = document.createElement('button');
      b.textContent = `${entry.title} · ${entry.page}`;
      b.style.paddingLeft = `${12 + Math.min(entry.depth, 3) * 14}px`;
      b.onclick = () => { goToPage(entry.page); dialog('contents').close(); };
      li.append(b); list.append(li);
    }
  } catch { element('outline-note').textContent = '目录暂时无法读取，可使用底部页码跳转。'; }
}

async function start() {
  if (!/^[a-f0-9]{16}$/.test(id)) throw new Error('请从论文详情页进入阅读器。');
  const response = await fetch('/read/papers.json', { signal: AbortSignal.timeout(15000) });
  if (!response.ok) throw new Error('论文列表暂时无法加载，请检查网络后重试。');
  const paper: Paper = (await response.json())[id];
  if (!paper) throw new Error('未找到这篇论文，请返回列表重新选择。');
  document.title = `${paper.title} · 论文阅读`;
  element('paper-title').textContent = paper.title; element('paper-title').title = paper.title;
  element<HTMLAnchorElement>('back').href = paper.details;
  for (const key of ['download', 'error-source']) {
    const link = element<HTMLAnchorElement>(key); link.href = paper.file; link.download = paper.filename; link.hidden = false;
  }
  const viewer = new URL('/pdfjs/web/viewer.html', location.origin);
  viewer.search = new URLSearchParams({ file: paper.file, filename: paper.filename, locale: 'zh-CN' }).toString();
  viewer.hash = 'zoom=page-width'; element<HTMLAnchorElement>('full-viewer').href = viewer.href;
  viewer.searchParams.set('host', 'reader');
  // Feed the same page to PDF.js's own initial-view pass. It can run after the
  // document proxy is ready (and again for documents with unequal page sizes).
  const requested = Number(params.get('page') || new URLSearchParams(location.hash.slice(1)).get('page'));
  const initialPage = Math.max(1, Math.trunc(requested || Number(saved.page) || 1));
  viewer.hash = `page=${initialPage}&zoom=page-width`;
  frame.src = viewer.href; frame.hidden = false;
  await new Promise<void>((resolve, reject) => {
    const deadline = Date.now() + 30000;
    let attached = false;
    const timer = setInterval(() => {
      app = (frame.contentWindow as any)?.PDFViewerApplication;
      if (app?.eventBus && !attached) {
        attached = true;
        app.eventBus.on('documenterror', () => { clearInterval(timer); reject(new Error('PDF 加载失败，可重试或下载原文。')); });
      }
      if (app?.pdfDocument && app.pdfViewer?.pagesCount && app.isInitialViewSet) { clearInterval(timer); resolve(); }
      else if (Date.now() > deadline) { clearInterval(timer); reject(new Error('加载时间较长，请检查网络后重试，也可以下载原文。')); }
    }, 100);
  });
  total = app.pdfDocument.numPages;
  // Use the whole phone width rather than PDF.js's desktop side gutters.
  app.pdfViewer.removePageBorders = true;
  app.pdfViewer.viewer.classList.add('removePageBorders');
  page = Math.max(1, Math.min(total, Math.trunc(requested || Number(saved.page) || 1)));
  app.pdfViewer.currentScaleValue = 'page-width';
  const savedScale = Number(saved.scale);
  if (savedScale >= 0.25 && savedScale <= 5 && !requested) app.pdfViewer.currentScaleValue = String(savedScale);
  app.pdfViewer.scrollPageIntoView({ pageNumber: page, ...(!requested && Number.isFinite(saved.pdfTop) ? { destArray: [null, { name: 'XYZ' }, saved.pdfLeft || 0, saved.pdfTop, null] } : {}) });
  ready = true;
  app.eventBus.on('pagechanging', ({ pageNumber }: { pageNumber: number }) => {
    if (restoring) return;
    page = pageNumber; updateControls(); queueSave();
  });
  app.eventBus.on('updateviewarea', queueSave);
  for (const key of ['zoom-in', 'zoom-out', 'contents-open', 'menu-open', 'fit-width']) button(key).disabled = false;
  pageInput.disabled = false;
  element('loading').hidden = true;
  updateControls(); restoring = false;
  // A shared page is an entry destination, not a permanent override of
  // subsequent reading progress when this same URL is refreshed.
  const cleanURL = new URL(location.href);
  cleanURL.searchParams.delete('page');
  if (new URLSearchParams(cleanURL.hash.slice(1)).has('page')) cleanURL.hash = '';
  history.replaceState(null, '', cleanURL);
  element('progress-status').textContent = page > 1 ? '已恢复进度' : '开始阅读';
  void buildOutline();
}
start().catch(error => failure(error instanceof Error ? error.message : '请检查网络后重试。'));
