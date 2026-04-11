document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const clearSearch = document.getElementById('clear-search');
    const searchBtn = document.getElementById('search-btn');
    const limitSelect = document.getElementById('limit-select');
    const resultsSection = document.getElementById('results-section');
    const resultsGrid = document.getElementById('results-grid');
    const loader = document.getElementById('loader');
    const syncBtn = document.getElementById('sync-btn');
    const statusBar = document.getElementById('status-bar');
    const toast = document.getElementById('toast');

    let selectedFile = null;

    // --- Upload Logic ---

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, () => dropZone.classList.remove('drag-over'));
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            showToast('Please select an image file', 'error');
            return;
        }
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            document.querySelector('.upload-content').classList.add('hidden');
            searchBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    clearSearch.addEventListener('click', (e) => {
        e.stopPropagation();
        resetSearch();
    });

    function resetSearch() {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        document.querySelector('.upload-content').classList.remove('hidden');
        searchBtn.disabled = true;
        resultsSection.classList.add('hidden');
        resultsGrid.innerHTML = '';
        statusBar.innerHTML = '';
    }

    // --- Search Logic ---

    searchBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State
        loader.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        searchBtn.disabled = true;
        resultsGrid.innerHTML = '';

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('limit', limitSelect.value);

        try {
            const response = await fetch('/api/v1/search/', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Search failed');

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            console.error(error);
            showToast('Search failed. Please try again.', 'error');
        } finally {
            loader.classList.add('hidden');
            searchBtn.disabled = false;
        }
    });

    function renderResults(data) {
        const { results, strategy, color_weight, texture_weight, semantic_weight } = data;
        const strategyDashboard = document.getElementById('strategy-dashboard');
        const strategyName = document.getElementById('strategy-name');
        const strategyWeights = document.getElementById('strategy-weights');

        if (results.length === 0) {
            statusBar.innerHTML = 'No matches found.';
            strategyDashboard.style.display = 'none';
            resultsSection.classList.remove('hidden');
            return;
        }

        // --- Update Strategy Dashboard ---
        strategyName.textContent = strategy;
        strategyWeights.innerHTML = `
            <div class="strategy-tag">AI: ${(semantic_weight * 100).toFixed(0)}%</div>
            <div class="strategy-tag">Color: ${(color_weight * 100).toFixed(0)}%</div>
            <div class="strategy-tag">Pattern: ${(texture_weight * 100).toFixed(0)}%</div>
        `;
        strategyDashboard.style.display = 'block';

        statusBar.innerHTML = `Matched ${results.length} items using Hybrid Ranking`;
        resultsGrid.innerHTML = '';
        
        results.forEach((item, index) => {
            const card = document.createElement('div');
            
            // Score formatting
            const totalScoreRaw = item.total_similarity * 100;
            const totalScore = totalScoreRaw.toFixed(1);
            const semanticScore = (item.semantic_score * 100).toFixed(0);
            const colorScore = (item.color_dist_score * 100).toFixed(0);
            const textureScore = (item.texture_score * 100).toFixed(0);

            // Confidence logic (Matches Streamlit thresholds)
            const isHighConfidence = index === 0 && totalScoreRaw >= 82 && (item.semantic_score * 100) >= 81;
            
            card.className = `result-card glass-card ${isHighConfidence ? 'high-confidence' : ''}`;
            
            card.innerHTML = `
                ${isHighConfidence ? '<div class="confidence-badge">🏆 HIGH CONFIDENCE MATCH</div>' : ''}
                <div class="card-img-wrapper">
                    <img src="/images/${item.filename}" alt="${item.filename}" loading="lazy">
                    <div class="match-badge">${totalScore}% Match</div>
                </div>
                <div class="card-content">
                    <div class="score-row">
                        <span class="score-main">${totalScore}%</span>
                        <span class="file-name" title="${item.filename}">${item.filename.substring(0, 15)}...</span>
                    </div>
                    <div class="score-breakdown">
                        <div class="score-item">
                            <span class="score-label">AI</span>
                            <div class="score-bar-bg"><div class="score-bar-fill ai-fill" style="width: ${semanticScore}%"></div></div>
                            <span class="score-value">${semanticScore}%</span>
                        </div>
                        <div class="score-item">
                            <span class="score-label">Color</span>
                            <div class="score-bar-bg"><div class="score-bar-fill color-fill" style="width: ${colorScore}%"></div></div>
                            <span class="score-value">${colorScore}%</span>
                        </div>
                        <div class="score-item">
                            <span class="score-label">Pattern</span>
                            <div class="score-bar-bg"><div class="score-bar-fill texture-fill" style="width: ${textureScore}%"></div></div>
                            <span class="score-value">${textureScore}%</span>
                        </div>
                    </div>
                </div>
            `;
            resultsGrid.appendChild(card);
        });

        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // --- Sync Logic ---

    syncBtn.addEventListener('click', async () => {
        const originalText = syncBtn.innerHTML;
        syncBtn.disabled = true;
        syncBtn.innerHTML = '<span class="spinner-small"></span> Syncing...';

        try {
            const response = await fetch('/api/v1/index/sync', { method: 'POST' });
            const data = await response.json();
            
            showToast(`Sync ${data.status}! Processed ${data.processed} new images.`, 'success');
        } catch (error) {
            showToast('Sync failed', 'error');
        } finally {
            syncBtn.disabled = false;
            syncBtn.innerHTML = originalText;
        }
    });

    // --- Utils ---

    function showToast(message, type = '') {
        toast.textContent = message;
        toast.className = `toast ${type}`;
        toast.classList.remove('hidden');
        setTimeout(() => toast.classList.add('hidden'), 5000);
    }
});
