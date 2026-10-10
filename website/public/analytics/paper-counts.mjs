const API = 'https://kg-stats.lihanghang.top/v1/papers/';
const format = new Intl.NumberFormat('zh-CN');

for (const element of document.querySelectorAll('[data-paper-views]')) {
  const paperId = element.dataset.paperViews || new URLSearchParams(location.search).get('paper');
  if (!/^[a-f0-9]{16}$/.test(paperId || '')) continue;
  let displayed = -1;
  function render(total) {
    if (!Number.isSafeInteger(total) || total < 0) return;
    // An earlier GET must not overwrite a newer successful reading event.
    displayed = Math.max(displayed, total);
    element.textContent = `阅读 ${format.format(displayed)} 次`;
    element.hidden = false;
  }
  window.addEventListener('message', event => {
    if (event.origin !== location.origin || event.data?.type !== 'kg:paper-views' || event.data.paperId !== paperId) return;
    const reader = [...document.querySelectorAll('iframe')].find(frame => frame.contentWindow === event.source);
    if (!reader || new URL(reader.src, location.href).searchParams.get('paper') !== paperId) return;
    render(event.data.totalViews);
  });
  // Read-only, anonymous lookup. Failure hides the number, never invents zero.
  fetch(`${API}${paperId}/views`, { credentials: 'omit', referrerPolicy: 'no-referrer', signal: AbortSignal.timeout(5000) })
    .then(response => { if (!response.ok) throw new Error('Unavailable'); return response.json(); })
    .then(data => { if (data.paperId === paperId) render(data.totalViews); })
    .catch(() => {});
}
