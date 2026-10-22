import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';

const site = 'https://kg.lihanghang.top';

function escapeXml(value: string): string {
	return value
		.replaceAll('&', '&amp;')
		.replaceAll('<', '&lt;')
		.replaceAll('>', '&gt;')
		.replaceAll('"', '&quot;')
		.replaceAll("'", '&apos;');
}

export const GET: APIRoute = async () => {
	const papers = (await getCollection('docs', ({ data }) => Boolean(data.collectedAt)))
		.sort((a, b) => Date.parse(b.data.collectedAt!) - Date.parse(a.data.collectedAt!));
	const updated = papers[0]?.data.collectedAt ?? new Date().toISOString();

	const items = papers.map((paper) => {
		const url = `${site}/${paper.id.split('/').map(encodeURIComponent).join('/')}/`;
		const date = paper.data.collectedAt!;
		return `
    <item>
      <title>${escapeXml(paper.data.title)}</title>
      <link>${escapeXml(url)}</link>
      <guid isPermaLink="true">${escapeXml(url)}</guid>
      <pubDate>${new Date(date).toUTCString()}</pubDate>
      <description>${escapeXml(paper.data.description ?? '')}</description>
      <category>${escapeXml(paper.data.paperCategory ?? '')}</category>
    </item>`;
	}).join('');

	const xml = `<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>知识图谱与语言智能 · 论文更新</title>
    <link>${site}/updates/</link>
    <description>自然语言处理与知识图谱论文库新增论文</description>
    <language>zh-CN</language>
    <lastBuildDate>${new Date(updated).toUTCString()}</lastBuildDate>${items}
  </channel>
</rss>`;

	return new Response(xml, {
		headers: { 'Content-Type': 'application/rss+xml; charset=utf-8' },
	});
};
