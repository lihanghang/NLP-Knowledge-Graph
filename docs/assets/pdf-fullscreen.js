// PDF Viewer Fullscreen Functionality
document.addEventListener('DOMContentLoaded', function() {
    // Add fullscreen button to all PDF viewer containers
    const pdfContainers = document.querySelectorAll('.pdf-viewer-container');

    pdfContainers.forEach(container => {
        // Create fullscreen button
        const fullscreenBtn = document.createElement('button');
        fullscreenBtn.className = 'pdf-fullscreen-btn';
        fullscreenBtn.innerHTML = '⛶ 全屏';
        fullscreenBtn.setAttribute('aria-label', '全屏查看PDF');
        fullscreenBtn.setAttribute('type', 'button');

        // Insert button before iframe
        container.insertBefore(fullscreenBtn, container.firstChild);

        // Handle fullscreen toggle
        fullscreenBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();

            if (!document.fullscreenElement && !document.webkitFullscreenElement && !document.mozFullScreenElement) {
                // Enter fullscreen
                enterFullscreen(container);
            } else {
                // Exit fullscreen
                exitFullscreen();
            }
        });

        // Touch event optimization for mobile
        let touchStartY = 0;
        fullscreenBtn.addEventListener('touchstart', function(e) {
            touchStartY = e.touches[0].clientY;
        }, { passive: true });

        fullscreenBtn.addEventListener('touchend', function(e) {
            const touchEndY = e.changedTouches[0].clientY;
            // Prevent accidental scrolling from triggering button
            if (Math.abs(touchEndY - touchStartY) < 10) {
                e.preventDefault();
            }
        });
    });

    // Fullscreen helper functions with cross-browser support
    function enterFullscreen(element) {
        if (element.requestFullscreen) {
            element.requestFullscreen();
        } else if (element.webkitRequestFullscreen) {
            element.webkitRequestFullscreen();
        } else if (element.mozRequestFullScreen) {
            element.mozRequestFullScreen();
        } else if (element.msRequestFullscreen) {
            element.msRequestFullscreen();
        }
    }

    function exitFullscreen() {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.mozCancelFullScreen) {
            document.mozCancelFullScreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
    }

    // Update button text when fullscreen state changes
    document.addEventListener('fullscreenchange', updateFullscreenButton);
    document.addEventListener('webkitfullscreenchange', updateFullscreenButton);
    document.addEventListener('mozfullscreenchange', updateFullscreenButton);
    document.addEventListener('MSFullscreenChange', updateFullscreenButton);

    function updateFullscreenButton() {
        const isFullscreen = document.fullscreenElement || document.webkitFullscreenElement || document.mozFullScreenElement;
        const buttons = document.querySelectorAll('.pdf-fullscreen-btn');

        buttons.forEach(btn => {
            if (isFullscreen) {
                btn.innerHTML = '✕ 退出全屏';
                btn.classList.add('is-fullscreen');
                btn.setAttribute('aria-label', '退出全屏');
            } else {
                btn.innerHTML = '⛶ 全屏';
                btn.classList.remove('is-fullscreen');
                btn.setAttribute('aria-label', '全屏查看PDF');
            }
        });
    }

    // Handle ESC key to exit fullscreen
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && (document.fullscreenElement || document.webkitFullscreenElement)) {
            exitFullscreen();
        }
    });

    // Double-tap to fullscreen on mobile (optional enhancement)
    let lastTap = 0;
    pdfContainers.forEach(container => {
        container.addEventListener('dblclick', function() {
            if (!document.fullscreenElement) {
                enterFullscreen(container);
            }
        });

        // Mobile double-tap detection
        container.addEventListener('touchend', function(e) {
            const currentTime = new Date().getTime();
            const tapLength = currentTime - lastTap;
            if (tapLength < 500 && tapLength > 0) {
                // Double tap detected
                if (!document.fullscreenElement && !document.webkitFullscreenElement) {
                    enterFullscreen(container);
                }
                e.preventDefault();
            }
            lastTap = currentTime;
        });
    });
});

// Mobile viewport height fix (for iOS)
function setMobileVH() {
    let vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty('--vh', `${vh}px`);
}

// Initialize on mobile devices
if (window.innerWidth <= 768 || /iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
    setMobileVH();

    // Update on resize and orientation change with debouncing
    let resizeTimer;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(setMobileVH, 100);
    });

    window.addEventListener('orientationchange', function() {
        setTimeout(setMobileVH, 300);
    });
}

// Prevent iOS Safari bounce when in fullscreen
document.addEventListener('touchmove', function(e) {
    if (document.fullscreenElement || document.webkitFullscreenElement) {
        const target = e.target;
        const iframe = target.closest('iframe');
        if (!iframe) {
            e.preventDefault();
        }
    }
}, { passive: false });

