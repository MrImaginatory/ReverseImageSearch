document.addEventListener('DOMContentLoaded', () => {
    // --- Selectors ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const clearSearch = document.getElementById('clear-search');
    const searchBtn = document.getElementById('search-btn');
    const syncBtn = document.getElementById('sync-btn');
    
    const resultsSection = document.getElementById('results-section');
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsGrid = document.getElementById('results-grid');
    const loader = document.getElementById('loader');
    
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
        resultsPlaceholder.classList.remove('hidden');
        resultsGrid.innerHTML = '';
    }

    // --- Search Logic ---
    searchBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State
        resultsPlaceholder.classList.add('hidden');
        resultsSection.classList.add('hidden');
        loader.classList.remove('hidden');
        searchBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('limit', 6);

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
            resultsPlaceholder.classList.remove('hidden');
        } finally {
            loader.classList.add('hidden');
            searchBtn.disabled = false;
        }
    });

    function renderResults(data) {
        const { status, highconfidence, silimar } = data;

        if (!highconfidence && (!silimar || silimar.length === 0)) {
            showToast('No matches found in the collection.', 'error');
            resultsPlaceholder.classList.remove('hidden');
            return;
        }

        resultsGrid.innerHTML = '';

        // Helper to create card matching new CSS card structure
        const createCard = (item, isHighConf = false) => {
            const card = document.createElement('a');
            card.href = `/images/${item.filename}`;
            card.target = '_blank';
            card.className = `result-card fade-in ${isHighConf ? 'high-confidence' : ''}`;
            card.style.textDecoration = 'none';
            card.style.color = 'inherit';
            
            const totalScore = (item.total_similarity * 100).toFixed(1);

            card.innerHTML = `
                <div class="match-badge">${totalScore}% Match</div>
                <div class="card-img-wrapper">
                    <img src="/images/${item.filename}" alt="${item.filename}" loading="lazy">
                </div>
                <div class="card-content">
                    <div class="card-header">
                        <span class="card-filename" title="${item.filename}">${item.filename}</span>
                        <span class="confidence-marker">${item.confidence_label || (isHighConf ? '🎯 Exact' : 'Similar')}</span>
                    </div>
                </div>
            `;
            return card;
        };

        // Render High Confidence Match
        if (highconfidence) {
            const groupTitle = document.createElement('h3');
            groupTitle.className = 'grid-group-title';
            groupTitle.textContent = 'Founded Match';
            resultsGrid.appendChild(groupTitle);
            
            resultsGrid.appendChild(createCard(highconfidence, true));
        }

        // Render Similar Items
        if (silimar && silimar.length > 0) {
            const groupTitle = document.createElement('h3');
            groupTitle.className = 'grid-group-title';
            groupTitle.textContent = 'Similar Collection Items';
            groupTitle.style.marginTop = highconfidence ? '2rem' : '0';
            resultsGrid.appendChild(groupTitle);

            silimar.forEach(item => {
                resultsGrid.appendChild(createCard(item, false));
            });
        }

        resultsSection.classList.remove('hidden');
        if (window.lucide) {
            lucide.createIcons();
        }
    }

    // --- Sync Logic ---
    syncBtn.addEventListener('click', async () => {
        const originalText = syncBtn.innerHTML;
        syncBtn.disabled = true;
        syncBtn.innerHTML = '<span><div class="spinner-small"></div></span> Syncing...';

        try {
            const response = await fetch('/api/v1/index/sync', { method: 'POST' });
            if (!response.ok) throw new Error('Sync failed');
            const data = await response.json();
            
            showToast(`Sync ${data.status}! Processed ${data.processed} new images.`, 'success');
        } catch (error) {
            console.error(error);
            showToast('Index synchronization failed', 'error');
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
