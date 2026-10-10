import test from 'node:test';
import assert from 'node:assert/strict';
import { DatabaseSync } from 'node:sqlite';
import { readFileSync } from 'node:fs';
import worker, { handleRequest, recordView, visitorKey } from '../src/index.mjs';

const id = 'f8d56fb77b411557', other = 'be6e998579f8a389';
const visitor = '12345678-1234-4234-8234-123456789abc';
const origin = 'https://kg.lihanghang.top';
const paper = { title: 'A paper', url: `${origin}/paper/` };
function database() {
  const sqlite = new DatabaseSync(':memory:');
  sqlite.exec('PRAGMA foreign_keys = ON');
  sqlite.exec(readFileSync(new URL('../migrations/0001_views.sql', import.meta.url), 'utf8'));
  const db = { prepare(sql) { return { bind(...values) { return {
    sql, values,
    async run() { return sqlite.prepare(sql).run(...values); },
    async first() { return sqlite.prepare(sql).get(...values); },
  }; } }; }, async batch(statements) {
    sqlite.exec('BEGIN');
    try {
      const results = statements.map(({ sql, values }) => ({ results: sqlite.prepare(sql).all(...values) }));
      sqlite.exec('COMMIT'); return results;
    } catch (error) { sqlite.exec('ROLLBACK'); throw error; }
  } };
  return { sqlite, db };
}
test('atomic duplicate suppression, exact rolling boundary, independent visitors and papers', async () => {
  const { sqlite, db } = database(), now = 1791540000;
  const key = await visitorKey(id, visitor);
  assert.notEqual(key, await visitorKey(other, visitor));
  assert.equal((await recordView(db, id, key, paper, now)).counted, true);
  assert.equal((await recordView(db, id, key, paper, now + 1799)).counted, false);
  assert.equal((await recordView(db, id, key, paper, now + 1800)).counted, true);
  const results = await Promise.all(Array.from({ length: 20 }, () => recordView(db, id, key, paper, now + 1801)));
  assert.equal(results.filter(r => r.counted).length, 0);
  assert.equal((await recordView(db, id, await visitorKey(id, crypto.randomUUID()), paper, now)).counted, true);
  assert.equal((await recordView(db, other, await visitorKey(other, visitor), paper, now)).counted, true);
  assert.equal(sqlite.prepare('SELECT SUM(views) n FROM paper_daily').get().n, 4);
  assert.equal(sqlite.prepare('SELECT COUNT(*) n FROM paper_stats').get().n, 2);
  sqlite.close();
});
test('UTC+8 day boundary does not reset deduplication; cleanup leaves aggregates intact', async () => {
  const { sqlite, db } = database();
  const beforeMidnight = Date.parse('2026-10-10T15:59:00Z') / 1000;
  const key = await visitorKey(id, visitor);
  await recordView(db, id, key, paper, beforeMidnight);
  await recordView(db, id, key, paper, beforeMidnight + 120);
  await recordView(db, id, key, paper, beforeMidnight + 1800);
  assert.deepEqual(sqlite.prepare('SELECT day, views FROM paper_daily ORDER BY day').all().map(x => ({ ...x })), [
    { day: '2026-10-10', views: 1 }, { day: '2026-10-11', views: 1 },
  ]);
  sqlite.prepare('DELETE FROM recent_views').run();
  assert.equal(sqlite.prepare('SELECT COUNT(*) n FROM paper_daily').get().n, 2);
  assert.equal(sqlite.prepare('SELECT SUM(views) n FROM paper_daily').get().n, 2);
  sqlite.close();
});
test('validation, private reports, CORS, oversized body, unknown IDs, privacy signals, unavailable service', async () => {
  const { sqlite, db } = database(), env = { DB: db, SITE_ORIGIN: origin };
  const catalog = async () => ({ [id]: { title: paper.title, details: '/paper/?details=1' } });
  const req = (body = { paperId: id, visitorId: visitor }, headers = {}, method = 'POST') => new Request('https://kg-stats.lihanghang.top/v1/view', {
    method, headers: { Origin: origin, 'Content-Type': 'application/json', ...headers },
    ...(method === 'POST' ? { body: typeof body === 'string' ? body : JSON.stringify(body) } : {}),
  });
  assert.equal((await handleRequest(req(), env, catalog)).status, 200);
  assert.equal((await handleRequest(req({}, {}, 'OPTIONS'), env, catalog)).headers.get('Access-Control-Allow-Origin'), origin);
  assert.equal((await handleRequest(req({}, { Origin: 'https://unrelated.test' }), env, catalog)).status, 403);
  assert.equal((await handleRequest(req('x'.repeat(513)), env, catalog)).status, 413);
  assert.equal((await handleRequest(req('{'), env, catalog)).status, 400);
  assert.equal((await handleRequest(req({ paperId: 'bad', visitorId: visitor }), env, catalog)).status, 400);
  assert.equal((await handleRequest(req({ paperId: other, visitorId: visitor }), env, catalog)).status, 404);
  assert.equal((await handleRequest(req({}, {}, 'GET'), env, catalog)).status, 405);
  assert.equal((await handleRequest(req({}, { 'Content-Type': 'text/plain' }), env, catalog)).status, 415);
  assert.equal((await handleRequest(req(undefined, { 'Sec-GPC': '1' }), env, catalog)).status, 200);
  assert.equal((await handleRequest(req(), env, async () => { throw Error('offline'); })).status, 503);
  assert.equal((await worker.fetch(new Request('https://example.com/stats'), env, {})).status, 404);
  assert.equal(sqlite.prepare('SELECT SUM(views) n FROM paper_daily').get().n, 1);
  sqlite.close();
});
test('Worker entrypoint receives execution context without treating it as catalog loader', async () => {
  const { sqlite, db } = database(), originalFetch = globalThis.fetch;
  globalThis.fetch = async () => Response.json({ [id]: { title: paper.title, details: '/paper/' } });
  try {
    const response = await worker.fetch(new Request('https://kg-stats.lihanghang.top/v1/view', {
      method: 'POST', headers: { Origin: origin, 'Content-Type': 'application/json' }, body: JSON.stringify({ paperId: id, visitorId: visitor }),
    }), { DB: db, SITE_ORIGIN: origin }, { waitUntil() {} });
    assert.equal(response.status, 200);
    assert.equal((await response.json()).counted, true);
    await worker.scheduled({}, { DB: db });
  } finally { globalThis.fetch = originalFetch; sqlite.close(); }
});

test('public paper counts return only the aggregate, include zero, and never record a read', async () => {
  const { sqlite, db } = database(), env = { DB: db, SITE_ORIGIN: origin };
  const catalog = async () => ({ [id]: {}, [other]: {} });
  const request = (paperId = id, method = 'GET') => new Request(`https://kg-stats.lihanghang.top/v1/papers/${paperId}/views`, { method });
  const zero = await handleRequest(request(), env, catalog);
  assert.equal(zero.headers.get('Access-Control-Allow-Origin'), origin);
  assert.equal(zero.headers.get('Cache-Control'), 'no-store');
  assert.deepEqual(await zero.json(), { paperId: id, totalViews: 0 });
  assert.equal(sqlite.prepare('SELECT COUNT(*) n FROM recent_views').get().n, 0);
  const key = await visitorKey(id, visitor), now = Math.floor(Date.now() / 1000);
  assert.equal((await recordView(db, id, key, paper, now)).totalViews, 1);
  assert.equal((await recordView(db, id, key, paper, now + 1)).totalViews, 1);
  assert.equal((await recordView(db, id, key, paper, now + 1800)).totalViews, 2);
  for (let n = 0; n < 3; n++) assert.deepEqual(await (await handleRequest(request(), env, catalog)).json(), { paperId: id, totalViews: 2 });
  assert.deepEqual(await (await handleRequest(request(other), env, catalog)).json(), { paperId: other, totalViews: 0 });
  assert.equal((await handleRequest(request('0000000000000000'), env, catalog)).status, 404);
  assert.equal((await handleRequest(request('bad'), env, catalog)).status, 404);
  assert.equal((await handleRequest(request(id, 'POST'), env, catalog)).status, 405);
  assert.equal((await handleRequest(new Request('https://example.com/v1/ranking'), env, catalog)).status, 404);
  const unavailable = await handleRequest(request(), { ...env, DB: { prepare() { throw Error('offline'); } } }, catalog);
  assert.equal(unavailable.status, 503);
  assert.equal(Object.hasOwn(await unavailable.json(), 'totalViews'), false);
  assert.equal(sqlite.prepare('SELECT SUM(views) n FROM paper_daily').get().n, 2);
  sqlite.close();
});
