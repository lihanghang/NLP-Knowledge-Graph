// PDF viewer initialization - loads pdf.js from CDN when needed
document.addEventListener('DOMContentLoaded', function() {
    var containers = document.querySelectorAll('.pdf-viewer-container[data-pdf]');
    if (containers.length === 0) return;

    var script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js';
    script.onload = function() {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        containers.forEach(function(container) {
            var pdfUrl = container.getAttribute('data-pdf');
            renderPdf(container, pdfUrl);
        });
    };
    document.head.appendChild(script);
});

function renderPdf(container, url) {
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#ccc;">加载中...</div>';

    pdfjsLib.getDocument(url).promise.then(function(pdf) {
        container.innerHTML = '';

        var toolbar = document.createElement('div');
        toolbar.className = 'pdf-toolbar';
        toolbar.style.cssText = 'display:flex;align-items:center;gap:8px;padding:6px 12px;background:#323639;color:#fff;font-size:13px;';

        var pageInfo = document.createElement('span');
        pageInfo.textContent = '共 ' + pdf.numPages + ' 页';

        var prevBtn = document.createElement('button');
        prevBtn.textContent = '◀ 上一页';
        prevBtn.style.cssText = 'background:#555;color:#fff;border:none;padding:4px 10px;border-radius:3px;cursor:pointer;';

        var pageInput = document.createElement('input');
        pageInput.type = 'number';
        pageInput.min = 1;
        pageInput.max = pdf.numPages;
        pageInput.value = 1;
        pageInput.style.cssText = 'width:50px;text-align:center;background:#444;color:#fff;border:1px solid #666;border-radius:3px;padding:3px;';

        var nextBtn = document.createElement('button');
        nextBtn.textContent = '下一页 ▶';
        nextBtn.style.cssText = 'background:#555;color:#fff;border:none;padding:4px 10px;border-radius:3px;cursor:pointer;';

        var downloadBtn = document.createElement('a');
        downloadBtn.href = url;
        downloadBtn.download = '';
        downloadBtn.textContent = '⬇ 下载PDF';
        downloadBtn.style.cssText = 'margin-left:auto;color:#8ab4f8;text-decoration:none;';

        toolbar.append(prevBtn, pageInfo, pageInput, nextBtn, downloadBtn);

        var viewer = document.createElement('div');
        viewer.style.cssText = 'overflow:auto;height:calc(100% - 40px);background:#525659;text-align:center;padding:10px 0;';

        container.append(toolbar, viewer);

        var currentPage = 1;
        var rendering = false;

        function renderPage(num) {
            if (rendering) return;
            rendering = true;
            pdf.getPage(num).then(function(page) {
                var scale = 1.5;
                var viewport = page.getViewport({ scale: scale });
                var canvas = document.createElement('canvas');
                var ctx = canvas.getContext('2d');
                canvas.width = viewport.width;
                canvas.height = viewport.height;
                canvas.style.cssText = 'max-width:100%;height:auto;margin:5px auto;display:block;box-shadow:0 1px 4px rgba(0,0,0,0.3);';
                page.render({ canvasContext: ctx, viewport: viewport }).promise.then(function() {
                    rendering = false;
                });
                viewer.appendChild(canvas);
            });
            pageInput.value = num;
            pageInfo.textContent = '第 ' + num + ' / ' + pdf.numPages + ' 页';
        }

        prevBtn.onclick = function() {
            if (currentPage > 1) { currentPage--; renderPage(currentPage); }
        };
        nextBtn.onclick = function() {
            if (currentPage < pdf.numPages) { currentPage++; renderPage(currentPage); }
        };
        pageInput.onchange = function() {
            var v = parseInt(this.value);
            if (v >= 1 && v <= pdf.numPages) { currentPage = v; renderPage(currentPage); }
        };

        renderPage(1);
    }).catch(function(err) {
        container.innerHTML = '<div style="padding:2em;color:#ff6b6b;text-align:center;">PDF加载失败: ' + err.message + '</div>';
    });
}
