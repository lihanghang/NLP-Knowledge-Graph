// Public Web Analytics site token; this is not an account/API credential.
const token = '420a22ef21fa4283a039b948a3f06e92';

function start() {
  if (location.hostname !== 'kg.lihanghang.top' || window.top !== window) return;
  if (navigator.globalPrivacyControl || navigator.doNotTrack === '1' || window.doNotTrack === '1') return;
  if (document.querySelector('script[data-cf-beacon]')) return;

  // On phones the detail page immediately redirects to /read/. Count the reader
  // destination, not the intermediate page. Embedded PDF.js has no beacon.
  if (matchMedia('(max-width: 50rem)').matches &&
      !new URLSearchParams(location.search).has('details') &&
      document.querySelector('[data-reader-link]')) return;
  if (document.title.startsWith('404 |')) return;

  const script = document.createElement('script');
  script.type = 'module';
  script.src = 'https://static.cloudflareinsights.com/beacon.min.js';
  script.dataset.cfBeacon = JSON.stringify({ token });
  document.head.append(script);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', start, { once: true });
} else {
  start();
}
