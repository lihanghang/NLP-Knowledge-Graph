import { execFileSync } from 'node:child_process';

// Read once during the static build. Rebuilding the same commit keeps its date
// and version, and no Git or API requests run in the visitor's browser.
function getRelease() {
  try {
    const [sha, timestamp] = execFileSync('git', ['log', '-1', '--format=%H%n%cI'], {
      encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'],
    }).trim().split('\n');
    const date = new Date(timestamp);
    if (!/^[a-f0-9]{40}$/.test(sha) || Number.isNaN(date.getTime())) throw new Error('Invalid Git metadata');
    return {
      version: sha.slice(0, 7),
      url: `https://github.com/lihanghang/NLP-Knowledge-Graph/commit/${sha}`,
      datetime: date.toISOString(),
      updated: new Intl.DateTimeFormat('zh-CN', {
        timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
      }).format(date).replaceAll('/', '-'),
    };
  } catch {
    // Source archives without .git can still be previewed without inventing a release.
    return { version: '本地预览', url: null, datetime: null, updated: null };
  }
}

export const release = getRelease();
