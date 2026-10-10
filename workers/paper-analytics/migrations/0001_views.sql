CREATE TABLE papers (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  url TEXT NOT NULL
);

-- Short-lived, per-paper anonymous keys. No IP, user agent, or raw browser ID.
CREATE TABLE recent_views (
  visitor_key TEXT PRIMARY KEY,
  paper_id TEXT NOT NULL REFERENCES papers(id),
  last_seen INTEGER NOT NULL
);
CREATE INDEX recent_views_expiry ON recent_views(last_seen);

CREATE TABLE paper_daily (
  paper_id TEXT NOT NULL REFERENCES papers(id),
  day TEXT NOT NULL,
  views INTEGER NOT NULL DEFAULT 0 CHECK (views >= 0),
  PRIMARY KEY (paper_id, day)
);

-- Counting and deduplication are one atomic SQLite write, including concurrent
-- tabs and requests arriving on different Worker instances.
CREATE TRIGGER count_first_view AFTER INSERT ON recent_views BEGIN
  INSERT INTO paper_daily(paper_id, day, views)
  VALUES (NEW.paper_id, date(NEW.last_seen, 'unixepoch', '+8 hours'), 1)
  ON CONFLICT(paper_id, day) DO UPDATE SET views = views + 1;
END;
CREATE TRIGGER count_return_view AFTER UPDATE OF last_seen ON recent_views BEGIN
  INSERT INTO paper_daily(paper_id, day, views)
  VALUES (NEW.paper_id, date(NEW.last_seen, 'unixepoch', '+8 hours'), 1)
  ON CONFLICT(paper_id, day) DO UPDATE SET views = views + 1;
END;

-- Private reports, accessible in the authenticated Cloudflare D1 console.
CREATE VIEW paper_stats AS
SELECT p.id AS paper_id, p.title, p.url,
       SUM(d.views) AS total_views,
       SUM(CASE WHEN d.day = date('now', '+8 hours') THEN d.views ELSE 0 END) AS today_views,
       SUM(CASE WHEN d.day >= date('now', '+8 hours', '-6 days') THEN d.views ELSE 0 END) AS last_7_days_views
FROM papers p JOIN paper_daily d ON p.id = d.paper_id
GROUP BY p.id
ORDER BY last_7_days_views DESC, total_views DESC, p.id;

CREATE VIEW daily_stats AS
SELECT day, SUM(views) AS views, COUNT(*) AS papers_viewed
FROM paper_daily GROUP BY day ORDER BY day DESC;
