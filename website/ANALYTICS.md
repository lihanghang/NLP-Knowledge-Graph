# 网站访问统计

生产站点 `kg.lihanghang.top` 手动加载 Cloudflare Web Analytics，不依赖 DNS 代理模式。

[打开 Web Analytics](https://dash.cloudflare.com/0e7319c81818adb53ec94b3341afb5a8/web-analytics/sites)，选择 **kg.lihanghang.top** 和日期范围，查看 Page views（页面浏览量）与 Visits（外部来源或直接进入的访问次数）。Visits 不是去重人数 UV；HTTP 请求数、DNS 查询数和论文阅读次数均不是页面浏览量。

站点 ID：`d671923677d9447f945f8f2994546834`。配置为手动安装（`auto_install: false`）。前端 token 是公开站点标识，不是 Cloudflare API 密钥。

## 采集范围

- 首页、论文更新、分类、桌面论文详情、独立手机阅读页。
- 手机详情页自动跳转阅读页时，仅在目标阅读页加载统计脚本。
- 内嵌 PDF.js 和直接打开的 PDF.js 完整阅读器不加载此脚本，PDF 翻页不增加 PV。
- 本地预览、非生产域名、iframe、404 页面、启用 GPC / DNT 的浏览器不加载脚本。
- `/read/?paper=...` 统一显示为 `/read/`；Cloudflare Web Analytics 不记录查询参数。单篇阅读继续由 Workers + D1 独立计数。

脚本位于 `public/analytics/site-traffic.mjs`，由 Starlight 公共 Head 和 `/read/` 页面引入。发布后验证浏览器加载 `beacon.min.js`，并向 `cloudflareinsights.com/cdn-cgi/rum` 成功发送请求，再确认后台出现数据。首次上线验证会产生少量实际测试访问，不回填上线前数据。

前端拦截、网络失败和 Cloudflare 报表采样可能影响数字，适合运营趋势观察，不是完整访问日志。Cloudflare 当前提供最近六个月的数据。不要将主域名 `lihanghang.top` 的统计直接当作本站统计。

暂停统计：移除两个页面入口中的 `site-traffic.mjs` 引用并发布。切换到自动注入前应移除手动接入并重新检查 PDF 和跳转页面的重复计数。

参考：[手动接入](https://developers.cloudflare.com/web-analytics/get-started/#sites-not-proxied-through-cloudflare)、[指标定义](https://developers.cloudflare.com/web-analytics/data-metrics/high-level-metrics/)、[限制与 FAQ](https://developers.cloudflare.com/web-analytics/faq/)。
