import { getCollection } from 'astro:content';

export async function GET() {
  const papers = await getCollection('docs', ({ data }) => Boolean(data.readerId));
  return new Response(JSON.stringify(Object.fromEntries(papers.map(({ id, data }) => [data.readerId, {
    title: data.title,
    file: data.pdfUrl,
    filename: data.pdfFilename,
    details: '/' + id.split('/').map(encodeURIComponent).join('/') + '/?details=1',
  }]))), { headers: { 'Content-Type': 'application/json; charset=utf-8' } });
}
