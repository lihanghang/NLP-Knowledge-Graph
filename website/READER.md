# PDF 专注阅读器

`/read/?paper=<id>` 是手机和桌面共用的 PDF 阅读页。手机打开论文详情时自动进入；返回按钮使用 `?details=1`，避免循环跳转。桌面保留完整 PDF.js，同时提供专注阅读入口。

- 顶部提供缩小、放大与适合宽度；底部提供目录、翻页和页码输入，主要操作的点击区域不小于 44 像素。
- 阅读区使用整个可用视口，去掉 PDF.js 的重复工具栏和多余左右留白。
- 页码、缩放与 PDF 阅读位置保存在当前浏览器的 `kg-reader:v1:<id>` 中。浏览器禁止存储时仍可阅读，不承诺跨设备同步。
- 设置中提供下载、完整阅读器（搜索、标注）、从第一页重新阅读。
- 目录使用 PDF 自带书签，没有书签时提供按页跳转。页码按 PDF 的物理页码计算。
- 自定义 PDF.js 样式只在 `host=reader` 时生效。
- 所有论文统一展示原版 PDF，没有文字重排、逐篇适配、试读页面或转换数据的维护流程。
- 生产环境在 PDF 渲染后累计可见停留 5 秒才上报阅读，同一浏览器、同一论文 30 分钟内去重；统计故障不影响阅读。[统计口径与后台使用说明](../workers/paper-analytics/README.md)。

## 构建与验证

```sh
python3 scripts/generate_site.py
ASTRO_TELEMETRY_DISABLED=1 npm --prefix website run build
```

先完成构建，再通过静态 HTTP 服务发布 `website/dist`。用独立 Chrome 测试配置目录开放调试端口后运行：

```sh
READER_BASE_URL=http://127.0.0.1:4325 CHROME_DEBUG_URL=http://127.0.0.1:9224 node scripts/check_reader.mjs
```

脚本只操作新建测试标签页，并清除该预览域名中 `kg-reader:` 开头的阅读设置。覆盖五份现有 PDF（单栏、双栏、中文、图表、扫描件）的翻页、跳页、缩放、目录、刷新续读和重新开始，以及手机入口/返回、320/390/1440 像素布局、禁止存储时的回退和已移除页面返回 404。截图默认保存到 `/private/tmp/kg-pdf-reader-check`，可用 `READER_SCREENSHOTS` 指定其它目录。

浏览器模拟不能代替 iPhone Safari/Android 真机、双指缩放、屏幕阅读器和弱网测试。
