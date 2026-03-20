(function () {
  'use strict';

  const form = document.getElementById('search-form');
  const queryInput = document.getElementById('query');
  const searchBtn = document.getElementById('search-btn');
  const searchHint = document.getElementById('search-hint');
  const loading = document.getElementById('loading');
  const errorEl = document.getElementById('error');
  const resultsSection = document.getElementById('results-section');
  const videoCounter = document.getElementById('video-counter');
  const exactMatch = document.getElementById('exact-match');
  const exactIframe = document.getElementById('exact-iframe');
  const exactTime = document.getElementById('exact-time');
  const mainIframe = document.getElementById('main-iframe');
  const transcript = document.getElementById('transcript');
  const metaStart = document.getElementById('meta-start');
  const metaEnd = document.getElementById('meta-end');
  const metaDuration = document.getElementById('meta-duration');
  const scoreAboveVideo = document.getElementById('score-above-video');
  const prevBtn = document.getElementById('prev-btn');
  const nextBtn = document.getElementById('next-btn');
  const navCounter = document.getElementById('nav-counter');
  const allResults = document.getElementById('all-results');
  const allResultsList = document.getElementById('all-results-list');

  let state = {
    results: [],
    exactResults: [],
    index: 0
  };

  function showLoading(show) {
    loading.hidden = !show;
    if (show) searchBtn.disabled = true;
    else searchBtn.disabled = false;
  }

  function showError(msg) {
    errorEl.textContent = msg || '';
    errorEl.hidden = !msg;
  }

  function secondsToHhmmss(seconds) {
    const s = Math.floor(Number(seconds));
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    return [h, m, sec].map(function (n) { return String(n).padStart(2, '0'); }).join(':');
  }

  function buildEmbedUrl(url, start, end, autoplay) {
    let videoId = url;
    if (url.indexOf('v=') !== -1) videoId = url.split('v=')[1].split('&')[0];
    else if (url.indexOf('/') !== -1) videoId = url.split('/').pop();
    const params = new URLSearchParams({ start: String(Math.floor(start)), end: String(Math.floor(end)) });
    if (autoplay) params.set('autoplay', '1');
    return 'https://www.youtube.com/embed/' + videoId + '?' + params.toString();
  }

  function renderCurrent() {
    const idx = state.index;
    const results = state.results;
    const exactResults = state.exactResults;
    if (!results.length) return;

    const current = results[idx];
    videoCounter.textContent = 'Video ' + (idx + 1) + ' of ' + results.length;

    if (exactResults[idx]) {
      const ex = exactResults[idx];
      exactMatch.hidden = false;
      exactIframe.src = buildEmbedUrl(ex.url, ex.start, ex.end, true);
      exactTime.textContent = 'Exact match: ' + secondsToHhmmss(ex.start) + ' → ' + secondsToHhmmss(ex.end) +
        (typeof ex.score === 'number' ? ' · similarity ' + (Math.round(ex.score * 1000) / 1000) : '');
    } else {
      exactMatch.hidden = true;
    }

    if (typeof current.score === 'number') {
      scoreAboveVideo.textContent = 'Cosine similarity: ' + (Math.round(current.score * 1000) / 1000);
      scoreAboveVideo.hidden = false;
      scoreAboveVideo.removeAttribute('aria-hidden');
    } else {
      scoreAboveVideo.textContent = '';
      scoreAboveVideo.hidden = true;
      scoreAboveVideo.setAttribute('aria-hidden', 'true');
    }

    mainIframe.src = buildEmbedUrl(current.url, current.start, current.end, false);
    transcript.textContent = current.transcript_snippet || 'No transcript available.';
    metaStart.textContent = 'Start: ' + secondsToHhmmss(current.start);
    metaEnd.textContent = 'End: ' + secondsToHhmmss(current.end);
    metaDuration.textContent = 'Duration: ' + (current.end - current.start) + 's';

    prevBtn.disabled = idx === 0;
    nextBtn.disabled = idx >= results.length - 1;
    navCounter.textContent = 'Video ' + (idx + 1) + ' / ' + results.length;

    allResultsList.innerHTML = results.map(function (v, i) {
      var label = (i + 1) + '. ' + v.url + ' (' + v.start + 's – ' + v.end + 's)';
      if (typeof v.score === 'number') label += ' · similarity ' + (Math.round(v.score * 1000) / 1000);
      if (i === idx) label += ' ▶ Currently playing';
      return '<li class="' + (i === idx ? 'current' : '') + '">' + escapeHtml(label) + '</li>';
    }).join('');
  }

  function escapeHtml(s) {
    var div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
  }

  function runSearch() {
    var query = (queryInput.value || '').trim();
    if (!query) {
      showError('Please enter a question first.');
      return;
    }

    showError('');
    showLoading(true);
    resultsSection.hidden = true;
    resultsSection.setAttribute('aria-hidden', 'true');

    fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    })
      .then(function (res) {
        if (!res.ok) return res.json().then(function (body) { throw new Error(body.detail || res.statusText); });
        return res.json();
      })
      .then(function (data) {
        state.results = data.results || [];
        state.exactResults = data.exact_results || [];
        state.index = 0;
        showLoading(false);
        if (state.results.length === 0) {
          searchHint.textContent = 'No results found for your question.';
          searchHint.hidden = false;
          return;
        }
        searchHint.hidden = true;
        resultsSection.hidden = false;
        resultsSection.removeAttribute('aria-hidden');
        allResults.open = false;
        renderCurrent();
      })
      .catch(function (err) {
        showLoading(false);
        showError(err.message || 'Search failed.');
      });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    runSearch();
  });

  prevBtn.addEventListener('click', function () {
    if (state.index > 0) {
      state.index--;
      renderCurrent();
    }
  });

  nextBtn.addEventListener('click', function () {
    if (state.index < state.results.length - 1) {
      state.index++;
      renderCurrent();
    }
  });
})();
