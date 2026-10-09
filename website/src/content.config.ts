import { defineCollection, z } from 'astro:content';
import { docsLoader, i18nLoader } from '@astrojs/starlight/loaders';
import { docsSchema, i18nSchema } from '@astrojs/starlight/schema';

export const collections = {
	docs: defineCollection({
		loader: docsLoader(),
		schema: docsSchema({
			extend: z.object({
				paperCategory: z.string().optional(),
				collectedAt: z.string().datetime({ offset: true }).optional(),
				readerId: z.string().optional(),
				pdfUrl: z.string().optional(),
				pdfFilename: z.string().optional(),
			}),
		}),
	}),
	i18n: defineCollection({ loader: i18nLoader(), schema: i18nSchema() }),
};
