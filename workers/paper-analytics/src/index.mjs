const WINDOW_SECONDS = 30 * 60;
const ID = /^[a-f0-9]{16}$/;
const UUID = /^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/i;
let cachedCatalog, catalogExpires = 0;

export async function visitorKey(paperId, visitorId) {
  const hash = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`${paperId}:${visitorId}`));
  return Array.from(new Uint8Array(hash), b => b.toString(16).padStart(2, '0')).join('');
}

export async function recordView(db, paperId, key, paper, now) {
  const results = await db.batch([
    db.prepare(`INSERT INTO papers(id, title, url) VALUES (?, ?, ?)
      ON CONFLICT(id) DO UPDATE SET title = excluded.title, url = excluded.url
      WHERE papers.title != excluded.title OR papers.url != excluded.url`)
      .bind(paperId, paper.title, paper.url),
    db.prepare(`INSERT INTO recent_views(visitor_key, paper_id, last_seen) VALUES (?, ?, ?)
      ON CONFLICT(visitor_key) DO UPDATE SET last_seen = excluded.last_seen
      WHERE excluded.last_seen - recent_views.last_seen >= ? RETURNING last_seen`)
      .bind(key, paperId, now, WINDOW_SECONDS),
    db.prepare('SELECT last_seen FROM recent_views WHERE visitor_key = ?').bind(key),
  ]);
  return { counted: results[1].results.length > 0, nextEligibleAt: (results[2].results[0].last_seen + WINDOW_SECONDS) * 1000 };
}

async function loadCatalog(env) {
  if (!cachedCatalog || Date.now() >= catalogExpires) {
    const response = await fetch(`${env.SITE_ORIGIN}/read/papers.json`, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error(`Catalog unavailable: HTTP ${response.status}`);
    const data = await response.json();
    if (!data || Array.isArray(data) || typeof data !== 'object') throw new Error('Invalid catalog');
    cachedCatalog = data;
    catalogExpires = Date.now() + 5 * 60 * 1000;
  }
  return cachedCatalog;
}

function json(body, status, origin) {
  const headers = { 'Content-Type': 'application/json', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', 'Vary': 'Origin' };
  if (origin) headers['Access-Control-Allow-Origin'] = origin;
  return new Response(JSON.stringify(body), { status, headers });
}

export async function handleRequest(request, env, catalogLoader = loadCatalog) {
  const path = new URL(request.url).pathname;
  if (path === '/health' && request.method === 'GET') return json({ ok: true }, 200);
  if (path !== '/v1/view') return json({ error: 'Not found' }, 404);
  const origin = request.headers.get('Origin');
  if (origin !== env.SITE_ORIGIN) return json({ error: 'Origin not allowed' }, 403);
  if (request.method === 'OPTIONS') return new Response(null, { status: 204, headers: {
    'Access-Control-Allow-Origin': origin, 'Access-Control-Allow-Methods': 'POST',
    'Access-Control-Allow-Headers': 'Content-Type', 'Access-Control-Max-Age': '86400', 'Vary': 'Origin',
  } });
  if (request.method !== 'POST') return json({ error: 'Method not allowed' }, 405, origin);
  if (!request.headers.get('Content-Type')?.toLowerCase().startsWith('application/json')) return json({ error: 'JSON required' }, 415, origin);
  // Keep the public endpoint small; never read an unbounded request body.
  const reader = request.body?.getReader();
  let length = 0, body = '';
  if (!reader) return json({ error: 'Missing body' }, 400, origin);
  const decoder = new TextDecoder();
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    length += value.byteLength;
    if (length > 512) { await reader.cancel(); return json({ error: 'Payload too large' }, 413, origin); }
    body += decoder.decode(value, { stream: true });
  }
  let input;
  try { input = JSON.parse(body + decoder.decode()); } catch { return json({ error: 'Invalid JSON' }, 400, origin); }
  if (!input || !ID.test(input.paperId) || !UUID.test(input.visitorId)) return json({ error: 'Invalid event' }, 400, origin);
  if (request.headers.get('Sec-GPC') === '1' || request.headers.get('DNT') === '1') return json({ skipped: true }, 200, origin);
  try {
    const catalog = await catalogLoader(env);
    const paper = catalog[input.paperId];
    if (!paper || typeof paper.title !== 'string' || !paper.details?.startsWith('/')) return json({ error: 'Unknown paper' }, 404, origin);
    const url = new URL(paper.details, env.SITE_ORIGIN);
    if (url.origin !== env.SITE_ORIGIN) return json({ error: 'Invalid paper' }, 404, origin);
    url.search = '';
    const result = await recordView(env.DB, input.paperId, await visitorKey(input.paperId, input.visitorId), {
      title: paper.title.slice(0, 600), url: url.href,
    }, Math.floor(Date.now() / 1000));
    return json(result, 200, origin);
  } catch {
    // Never log request bodies or identifiers. A failed counter is not a failed read.
    return json({ error: 'Temporarily unavailable' }, 503, origin);
  }
}

export default {
  fetch(request, env) { return handleRequest(request, env); },
  async scheduled(_event, env) {
    // Retain anonymous deduplication keys for at most ~25 hours, keep aggregates.
    await env.DB.prepare('DELETE FROM recent_views WHERE last_seen < ?')
      .bind(Math.floor(Date.now() / 1000) - 86400).run();
  },
};
