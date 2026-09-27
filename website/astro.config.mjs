// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// 分类顺序与 scripts/generate_site.py 中的 CATEGORIES 保持一致
const categories = [
	'自然语言处理',
	'知识库构建',
	'事理图谱',
	'基于知识图谱的对话系统',
	'知识图谱基础',
	'中文金融文档智能处理',
	'机器学习',
	'知识图谱技术总结',
	'语义计算',
	'知识存储',
	'数据集',
];

export default defineConfig({
	site: 'https://lihanghang.github.io',
	base: '/NLP-Knowledge-Graph',
	integrations: [
		starlight({
			title: 'NLP-Knowledge-Graph',
			description: '自然语言处理与知识图谱论文在线阅读',
			defaultLocale: 'root',
			locales: { root: { label: '简体中文', lang: 'zh-CN' } },
			logo: { src: './src/assets/logo.svg' },
			favicon: '/favicon.svg',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/lihanghang/NLP-Knowledge-Graph' },
			],
			customCss: ['./src/styles/theme.css'],
			components: {
				// 在每页注入 PDF 全屏脚本
				Head: './src/components/Head.astro',
			},
			sidebar: categories.map((name) => ({
				label: name,
				collapsed: true,
				items: [{ autogenerate: { directory: name, collapsed: true } }],
			})),
			pagination: false,
			lastUpdated: false,
		}),
	],
});
