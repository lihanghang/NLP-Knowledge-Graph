# 论文阅读统计

GitHub Pages 上的 PDF.js 阅读器向 Cloudflare Worker 上报，D1 保存按论文、按日汇总的阅读次数。PDF 文件继续由原站点提供；前台不显示次数或排行榜。

## 统计口径

- PDF 成功渲染一页后，阅读器累计可见停留 5 秒才上报；嵌入式阅读器至少 25% 面积在视口内。翻页本身不增加次数。
- 同一浏览器、同一论文，距上一次计数不足 30 分钟的上报不再计数。刷新、手机阅读页、桌面嵌入页、完整阅读器共用规则。持续打开页面不会每 30 分钟自动增加次数。
- 服务器使用原子写入与数据库触发器去重，避免并发标签页重复累加。失败的请求最多重试一次；统计故障不影响阅读。
- 日报使用北京时间（UTC+8）；近 7 天包括今天及之前 6 个自然日。上线前的阅读不回填。报表只列出已有阅读的论文。
- 本地预览、PDF 渲染失败、后台或视口外的阅读器不会贡献可见时长。启用 GPC / DNT、禁用浏览器存储或拦截统计接口时不计数。
- 这是阅读次数，不是独立人数。更换设备、清除浏览器数据可能再次计数；接口校验来源和论文目录，但不能阻止伪造请求，不应用于结算或严格反作弊场景。

## 查看数据

登录自己的 Cloudflare 账号，进入 **存储和数据库 → D1 → kg-paper-analytics → Console / Studio**。运行：

```sql
-- 各论文：标题、链接、累计、今日、近 7 天阅读次数
SELECT * FROM paper_stats;

-- 全站逐日阅读次数、当日被读过的论文数
SELECT * FROM daily_stats;
```

[打开数据库控制台](https://dash.cloudflare.com/0e7319c81818adb53ec94b3341afb5a8/workers/d1/databases/6767d78c-bd0a-4bd4-bdf0-529fb624e8b5/console)

报表只在已登录的 Cloudflare 后台提供，Worker 不暴露公开查询接口。也可在本目录运行：

```sh
npx wrangler d1 execute kg-paper-analytics --remote --command 'SELECT * FROM paper_stats'
```

## 数据与隐私

前端在本地生成随机浏览器 ID，仅发送该 ID 和论文 ID；服务端只保存二者组合后的 SHA-256 值，不保存原始浏览器 ID、IP、User-Agent 或姓名。每小时清理超过 24 小时的去重记录（最长约 25 小时），按日汇总数据继续保留。浏览器 ID 会保留在本站 localStorage，清除站点数据即可重置。

Worker 应用日志和可观测性采集关闭。网络请求仍由 Cloudflare 基础设施处理，以上说明针对本应用保存的数据。

## 部署与维护

```sh
cd workers/paper-analytics
npm ci
npx wrangler login
npm test
npx wrangler d1 migrations apply kg-paper-analytics --remote
npm run deploy
```

配置中的账号、数据库 ID 均为资源标识，不是密钥。OAuth 凭据由 Wrangler 在本机管理，不入仓库。`kg-stats.lihanghang.top` 是 Worker 的自定义域名，生产站点为 `kg.lihanghang.top`。

Worker 从生产站点 `/read/papers.json` 获取合法论文目录，缓存最多 5 分钟。新增论文发布后无需重新部署 Worker。`GET /health` 仅检查 Worker 可达性，不验证数据库；`POST /v1/view` 返回本次是否计数及下次可计数时间。

Cloudflare 的页面规则 `https://kg.lihanghang.top/read/papers.json` 设置为 **SSL: 严格**（规则 ID：`1decf5bb168048a7ac4264b04fac4a1d`）。这是对目录接口的精确匹配，解决域名默认“灵活”回源与 GitHub Pages 强制 HTTPS 导致的 Worker 子请求重定向循环；其他地址和子域名不受这条规则影响。迁移域名时需一并核对 HTTPS 回源。参见 [Cloudflare 重定向循环说明](https://developers.cloudflare.com/ssl/troubleshooting/too-many-redirects/)。

网站推送后由 GitHub Actions 执行统计单元测试并发布前端。Worker 和数据库迁移需单独运行上述命令；GitHub 不保存 Cloudflare 密钥。修改统计规则时，先部署兼容的 Worker，再发布前端。

暂停采集：移除 `website/public/pdfjs/web/nlpkg-viewer.mjs` 中的统计调用并发布网站；无需删除已有汇总。不要通过更新 `recent_views.last_seen` 修正报表，因为更新会触发新的计数。
