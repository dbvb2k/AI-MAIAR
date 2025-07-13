document.addEventListener('DOMContentLoaded', () => {
    const healthStatus = document.getElementById('health-status');
    const statusEmbedder = document.getElementById('status-embedder');
    const statusVector = document.getElementById('status-vector');
    const statusClassifier = document.getElementById('status-classifier');
    const statusMessage = document.getElementById('status-message');
    const form = document.getElementById('search-form');
    const queryInput = document.getElementById('query');
    const topNInput = document.getElementById('top_n');
    const resultsDiv = document.getElementById('results');
    const loadingDiv = document.getElementById('loading');
    const clearBtn = document.getElementById('clear-btn');
    const consoleContent = document.getElementById('console-content');
    let lastHealth = null;

    async function checkHealth() {
        try {
            const resp = await fetch('/health');
            const data = await resp.json();
            lastHealth = data;
            setStatusDot(statusEmbedder, data.embedding_model);
            setStatusDot(statusVector, data.vector_store);
            setStatusDot(statusClassifier, data.classifier);
            statusMessage.textContent = data.message;
            logConsole('Health: ' + JSON.stringify(data));
        } catch (e) {
            setStatusDot(statusEmbedder, false);
            setStatusDot(statusVector, false);
            setStatusDot(statusClassifier, false);
            statusMessage.textContent = 'API unreachable';
            logConsole('Health check failed: ' + e);
        }
    }
    function setStatusDot(dot, ok) {
        dot.classList.remove('ok', 'err', 'unknown');
        if (ok === true) dot.classList.add('ok');
        else if (ok === false) dot.classList.add('err');
        else dot.classList.add('unknown');
    }
    function logConsole(msg) {
        if (!consoleContent) return;
        const now = new Date().toLocaleTimeString();
        consoleContent.innerHTML += `<div>[${now}] ${msg}</div>`;
        consoleContent.scrollTop = consoleContent.scrollHeight;
    }
    function toggleConsole() {
        if (!consoleContent) return;
        if (consoleContent.style.display === 'none') {
            consoleContent.style.display = 'block';
        } else {
            consoleContent.style.display = 'none';
        }
    }
    window.toggleConsole = toggleConsole;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        resultsDiv.innerHTML = '';
        loadingDiv.style.display = 'block';
        const query = queryInput.value.trim();
        const top_n = parseInt(topNInput.value) || 3;
        if (!query) {
            loadingDiv.style.display = 'none';
            resultsDiv.innerHTML = '<div class="result-card">Please enter a query.</div>';
            return;
        }
        logConsole('Submitting query: ' + query);
        // Vector search
        let vectorResults = [];
        let classifierResult = null;
        try {
            const vresp = await fetch('/vector_search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_n })
            });
            if (!vresp.ok) throw new Error('Vector search failed');
            vectorResults = await vresp.json();
            logConsole('Vector search results: ' + JSON.stringify(vectorResults));
        } catch (e) {
            logConsole('Vector search error: ' + e);
            resultsDiv.innerHTML += `<div class="result-card">Vector search error: ${e}</div>`;
        }
        // Classifier
        try {
            const cresp = await fetch('/classifier', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });
            if (!cresp.ok) throw new Error('Classifier failed');
            classifierResult = await cresp.json();
            logConsole('Classifier result: ' + JSON.stringify(classifierResult));
        } catch (e) {
            logConsole('Classifier error: ' + e);
            resultsDiv.innerHTML += `<div class="result-card">Classifier error: ${e}</div>`;
        }
        loadingDiv.style.display = 'none';
        // Render results
        if (vectorResults && vectorResults.length > 0) {
            resultsDiv.innerHTML += `<h3>Top ${vectorResults.length} Similar Tickets</h3>`;
            vectorResults.forEach((r, i) => {
                resultsDiv.innerHTML += `
                <div class="result-card">
                    <div class="meta">#${i+1} | Ticket ID: <b>${r.ticketid || '-'}</b> | App: <span class="application">${r.application || '-'}</span> <span class="score">${r.similarity.toFixed(3)}</span></div>
                    <div class="summary">${r.summary || '<i>No summary</i>'}</div>
                    <details><summary>Metadata</summary><pre>${JSON.stringify(r.meta, null, 2)}</pre></details>
                </div>`;
            });
        } else {
            resultsDiv.innerHTML += '<div class="result-card">No similar tickets found.</div>';
        }
        if (classifierResult && classifierResult.prediction) {
            resultsDiv.innerHTML += `<h3>Classifier Prediction</h3>
                <div class="result-card">
                    <b>Predicted Application:</b> <span class="application">${classifierResult.prediction}</span><br>
                    ${classifierResult.probabilities ? `<details><summary>Probabilities</summary><pre>${JSON.stringify(classifierResult.probabilities, null, 2)}</pre></details>` : ''}
                </div>`;
        }
    });
    clearBtn.addEventListener('click', (e) => {
        e.preventDefault();
        queryInput.value = '';
        resultsDiv.innerHTML = '';
        loadingDiv.style.display = 'none';
        logConsole('Cleared form/results');
    });
    // Initial health check
    checkHealth();
}); 