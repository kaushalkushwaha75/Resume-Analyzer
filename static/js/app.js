/* ═══════════════════════════════════════════════════════════════════════════
   Resume Analyzer — Frontend Interactivity & Animations
   ═══════════════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    // ─── Drag & Drop File Upload ────────────────────────────────────────────
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('resume-input');
    const dropContent = document.getElementById('drop-zone-content');
    const dropPreview = document.getElementById('drop-zone-preview');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const fileRemove = document.getElementById('file-remove');

    if (dropZone && fileInput) {
        // Drag events
        ['dragenter', 'dragover'].forEach(evt => {
            dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(evt => {
            dropZone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove('drag-over');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                showFilePreview(files[0]);
            }
        });

        // File input change
        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                showFilePreview(fileInput.files[0]);
            }
        });

        // Remove file
        if (fileRemove) {
            fileRemove.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                fileInput.value = '';
                dropContent.style.display = '';
                dropPreview.style.display = 'none';
            });
        }
    }

    function showFilePreview(file) {
        if (!dropContent || !dropPreview || !fileName || !fileSize) return;

        const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
        const ext = file.name.split('.').pop().toLowerCase();

        if (!validTypes.includes(file.type) && !['pdf', 'docx'].includes(ext)) {
            alert('Please upload a PDF or DOCX file.');
            return;
        }

        if (file.size > 5 * 1024 * 1024) {
            alert('File size exceeds 5MB limit.');
            return;
        }

        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        dropContent.style.display = 'none';
        dropPreview.style.display = '';
    }

    function formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    // ─── Word Count for JD Textarea ─────────────────────────────────────────
    const jdInput = document.getElementById('jd-input');
    const jdWordCount = document.getElementById('jd-word-count');

    if (jdInput && jdWordCount) {
        jdInput.addEventListener('input', () => {
            const text = jdInput.value.trim();
            const count = text ? text.split(/\s+/).length : 0;
            jdWordCount.textContent = count;
        });
    }

    // ─── Form Submission with Loading State ─────────────────────────────────
    const analyzeForm = document.getElementById('analyze-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = analyzeBtn?.querySelector('.btn-text');
    const btnIcon = analyzeBtn?.querySelector('.btn-icon');
    const btnLoader = document.getElementById('btn-loader');

    if (analyzeForm && analyzeBtn) {
        analyzeForm.addEventListener('submit', (e) => {
            // Validate
            if (!fileInput?.files?.length) {
                e.preventDefault();
                alert('Please upload a resume file.');
                return;
            }

            if (!jdInput?.value?.trim()) {
                e.preventDefault();
                alert('Please paste a job description.');
                return;
            }

            // Show loading
            if (btnText) btnText.style.display = 'none';
            if (btnIcon) btnIcon.style.display = 'none';
            if (btnLoader) btnLoader.style.display = 'flex';
            analyzeBtn.disabled = true;
            analyzeBtn.style.opacity = '0.85';
            analyzeBtn.style.cursor = 'wait';
        });
    }

    // ─── Results Page Animations ────────────────────────────────────────────

    // Animated score count-up
    const scoreValues = document.querySelectorAll('[data-target]');
    if (scoreValues.length > 0) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCountUp(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.3 });

        scoreValues.forEach(el => observer.observe(el));
    }

    function animateCountUp(el) {
        const target = parseFloat(el.dataset.target);
        const duration = 1800;
        const start = performance.now();

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease out cubic
            const ease = 1 - Math.pow(1 - progress, 3);
            const current = target * ease;

            el.textContent = Math.round(current);

            if (progress < 1) {
                requestAnimationFrame(update);
            } else {
                // Show decimal for final value if needed
                el.textContent = target % 1 === 0 ? target : target.toFixed(1);
            }
        }

        requestAnimationFrame(update);
    }

    // Animated score ring
    const ringFill = document.querySelector('.ring-fill');
    if (ringFill) {
        const score = parseFloat(ringFill.dataset.score) || 0;
        const circumference = 2 * Math.PI * 52; // r=52
        const offset = circumference - (score / 100) * circumference;

        // Trigger animation after a short delay
        setTimeout(() => {
            ringFill.style.strokeDashoffset = offset;

            // Color the ring based on score
            if (score >= 70) {
                ringFill.style.stroke = '#10b981';
            } else if (score >= 40) {
                ringFill.style.stroke = '#f59e0b';
            } else {
                ringFill.style.stroke = '#ef4444';
            }
        }, 300);
    }

    // Animate section bars
    const sectionBars = document.querySelectorAll('.section-bar-fill');
    if (sectionBars.length > 0) {
        const barObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animated');
                    barObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.2 });

        sectionBars.forEach(bar => barObserver.observe(bar));
    }

    // Fade-in panels on scroll
    const panels = document.querySelectorAll('.panel, .scores-overview, .action-bar');
    if (panels.length > 0) {
        const panelObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate-in');
                    panelObserver.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        panels.forEach(panel => panelObserver.observe(panel));
    }

    // ─── Flash Message Auto-dismiss ─────────────────────────────────────────
    const flashMessages = document.querySelectorAll('.flash-message');
    flashMessages.forEach(msg => {
        setTimeout(() => {
            msg.style.animation = 'fadeInUp 0.4s ease reverse forwards';
            setTimeout(() => msg.remove(), 400);
        }, 6000);
    });

    // ─── Smooth Scroll for Anchor Links ─────────────────────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener('click', (e) => {
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

});
