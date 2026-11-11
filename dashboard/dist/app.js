/* Buildless, dependency-free rendering of the saved Python analysis snapshot. */
(() => {
  'use strict';

  const data = window.PORTFOLIO_DATA || {status: 'unavailable'};
  const ready = data.status === 'ready';
  const $ = (selector) => document.querySelector(selector);
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const finite = (value) => typeof value === 'number' && Number.isFinite(value);
  const number = (value, digits = 1) => finite(value) ? (Math.abs(value) < 1e-8 ? 0 : value).toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits}) : '—';
  const pct = (value, digits = 2) => finite(value) ? `${number(value * 100, digits)}%` : '—';
  const pp = (value, digits = 2) => finite(value) ? `${value > 0 ? '+' : ''}${number(value * 100, digits)} pp` : '—';
  const date = (value, options = {month: 'short', day: 'numeric', year: 'numeric'}) => {
    if (!value) return '—';
    const parsed = new Date(String(value).slice(0, 10) + 'T12:00:00Z');
    return Number.isNaN(parsed.valueOf()) ? String(value) : parsed.toLocaleDateString('en-US', {...options, timeZone: 'UTC'});
  };
  const strategies = {
    no_rebalancing: {label: 'No rebalancing', color: '#9db3b2'},
    rules_based: {label: 'Rules based', color: '#aa91bf'},
    optimized: {label: 'Optimized', color: '#167865'},
    benchmark: {label: 'SPY benchmark', color: '#d7826e'},
  };
  const strategyOrder = Object.keys(strategies);
  const summary = data.walk_forward?.summary || [];
  const base = summary.find((row) => row.strategy === 'no_rebalancing') || {};
  const optimized = summary.find((row) => row.strategy === 'optimized') || {};
  const meta = data.meta || {};
  const config = data.config || {};

  function dot(color, className = '') {
    const node = el('i', `legend-dot ${className}`);
    node.style.setProperty('--color', color);
    node.setAttribute('aria-hidden', 'true');
    return node;
  }

  function renderMetadata() {
    $('#unavailable-message').hidden = ready;
    if (!ready) {
      $('#data-badge').lastChild.textContent = ' Awaiting analysis';
      $('#export-button').disabled = true;
    }
    if (meta.name) $('#sidebar-name').textContent = meta.name;
    if (config.tickers?.length) {
      $('#sidebar-tickers').replaceChildren(...config.tickers.map((ticker) => el('span', '', ticker)));
    }
    if (meta.price_start && meta.price_end) $('#sample-dates').textContent = `${date(meta.price_start, {year: 'numeric'})} – ${date(meta.price_end, {year: 'numeric'})} · Saved ETF history`;
    if (meta.oos_start && meta.oos_end) $('#wealth-dates').textContent = `${date(meta.oos_start)} – ${date(meta.oos_end)} · Net growth of $1`;
    if (meta.generated_at) $('#generated-at').textContent = `Snapshot · ${date(meta.generated_at)}`;
    const provenance = [
      ['Portfolio', meta.name],
      ['Market data', meta.source],
      ['Price history', meta.price_start && `${date(meta.price_start)} – ${date(meta.price_end)}`],
      ['Price observations', finite(meta.observations) ? number(meta.observations, 0) : null],
      ['Out-of-sample test', meta.oos_start && `${date(meta.oos_start)} – ${date(meta.oos_end)}`],
      ['Covariance lookback', finite(config.lookback_days) ? `${config.lookback_days} trading days` : null],
      ['Rebalance interval', finite(config.rebalance_every_days) ? `${config.rebalance_every_days} trading days` : null],
      ['Generated', meta.generated_at],
    ];
    $('#provenance-list').replaceChildren(...provenance.map(([label, value]) => {
      const row = el('div');
      row.append(el('dt', '', label), el('dd', '', value || 'Not available'));
      return row;
    }));
  }

  function renderConclusions() {
    const conclusions = data.conclusions || [];
    if (conclusions[0]) {
      $('#conclusion-title').textContent = conclusions[0].title;
      $('#conclusion-body').textContent = conclusions[0].detail;
    }
    const visible = conclusions.length ? conclusions.slice(0, 3) : [
      {title: 'A saved result is required', detail: 'Run the analysis to inspect observed risk and return.'},
      {title: 'Every metric needs its context', detail: 'The final-date optimization and the walk-forward test answer different questions.'},
      {title: 'Assumptions define the conclusion', detail: 'Sensitivity analysis will show which findings depend on model settings.'},
    ];
    $('#conclusion-list').replaceChildren(...visible.map((finding, index) => {
      const item = el('div', `evidence-item ${finding.tone || ''}`);
      const copy = el('div');
      copy.append(el('h3', '', finding.title), el('p', '', finding.detail));
      item.append(el('span', 'evidence-number', `0${index + 1}`), copy);
      return item;
    }));
  }

  function renderMetrics() {
    const metrics = [
      {label: 'Realized tracking error', value: optimized.realized_tracking_error, reference: base.realized_tracking_error, better: 'lower', symbol: '↗', digits: 2},
      {label: 'Cumulative net return', value: optimized.cumulative_net_return, reference: base.cumulative_net_return, better: 'higher', symbol: '↗', digits: 2},
      {label: 'Worst historical stress exposure', value: optimized.worst_decision_stress_loss, reference: base.worst_decision_stress_loss, better: 'lower', symbol: '◇', digits: 2},
      {label: 'Total estimated trading cost', value: optimized.total_transaction_cost, reference: base.total_transaction_cost, better: 'lower', symbol: '⇄', digits: 3},
    ];
    $('#metric-grid').replaceChildren(...metrics.map((metric) => {
      const card = el('article', 'metric-card');
      const top = el('div', 'metric-top');
      top.append(el('span', 'metric-label', metric.label), el('span', 'metric-symbol', metric.symbol));
      const comparison = el('div', 'metric-comparison');
      const delta = metric.value - metric.reference;
      const good = metric.better === 'lower' ? delta < 0 : delta > 0;
      const tone = Math.abs(delta) < 1e-7 || !finite(delta) ? 'neutral' : good ? 'positive' : 'caution';
      comparison.append(el('span', `delta ${tone}`, pp(delta)), el('span', '', 'vs. no rebalancing'));
      card.append(top, el('div', 'metric-value', pct(metric.value, metric.digits)), comparison);
      return card;
    }));
  }

  function renderComparison() {
    const body = $('#comparison-body');
    if (!summary.length) {
      const row = el('tr', 'no-results');
      const cell = el('td', '', 'No saved strategy results.'); cell.colSpan = 6; row.append(cell); body.replaceChildren(row); return;
    }
    body.replaceChildren(...strategyOrder.filter((key) => summary.some((row) => row.strategy === key)).map((key) => {
      const result = summary.find((row) => row.strategy === key);
      const row = el('tr', key === 'optimized' ? 'optimized-row' : '');
      const labelCell = el('td'); const label = el('span', 'strategy-label');
      label.append(dot(strategies[key].color), el('span', '', strategies[key].label));
      if (key === 'optimized') label.append(el('span', 'row-tag', 'MODEL'));
      labelCell.append(label); row.append(labelCell);
      [result.cumulative_net_return, result.realized_tracking_error, result.mean_decision_stress_loss, result.total_one_way_turnover, result.total_transaction_cost].forEach((value, index) => row.append(el('td', '', pct(value, index === 4 ? 3 : 2))));
      return row;
    }));
  }

  const visibleStrategies = new Set(strategyOrder);
  let chartPeriod = 'all';
  const namespace = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attributes = {}, text) {
    const node = document.createElementNS(namespace, tag);
    Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function renderWealth() {
    const svg = $('#wealth-svg');
    const all = (data.walk_forward?.wealth || []).filter((row) => row.date);
    if (!all.length) {
      svg.replaceChildren(svgEl('text', {x: 450, y: 165, 'text-anchor': 'middle'}, 'No observed wealth series available.'));
      return;
    }
    let rows = all;
    if (chartPeriod === '1y') {
      const cutoff = new Date(all[all.length - 1].date);
      cutoff.setUTCFullYear(cutoff.getUTCFullYear() - 1);
      rows = all.filter((row) => new Date(row.date) >= cutoff);
    }
    const keys = strategyOrder.filter((key) => visibleStrategies.has(key));
    const values = rows.flatMap((row) => keys.map((key) => row[key]).filter(finite));
    const W = 900, H = 340, left = 48, right = 16, top = 18, bottom = 31;
    const width = W - left - right, height = H - top - bottom;
    let min = Math.min(...values), max = Math.max(...values);
    const span = max - min || .1;
    min -= span * .1; max += span * .1;
    const x = (index) => left + index / Math.max(1, rows.length - 1) * width;
    const y = (value) => top + (max - value) / (max - min) * height;
    svg.replaceChildren(svgEl('title', {id: 'wealth-title'}, 'Observed out-of-sample net wealth'), svgEl('desc', {id: 'wealth-description'}, `Growth of $1 since ${date(all[0].date)}. Displayed dates ${date(rows[0].date)} through ${date(rows[rows.length - 1].date)}. All returns are net of modeled transaction costs.`));
    for (let tick = 0; tick <= 4; tick += 1) {
      const value = min + (max - min) * tick / 4;
      svg.append(svgEl('line', {x1: left, x2: W - right, y1: y(value), y2: y(value), class: 'chart-grid'}), svgEl('text', {x: left - 11, y: y(value) + 3, 'text-anchor': 'end'}, `$${value.toFixed(2)}`));
    }
    for (let tick = 0; tick <= 4; tick += 1) {
      const index = Math.round(tick / 4 * (rows.length - 1));
      svg.append(svgEl('text', {x: x(index), y: H - 5, 'text-anchor': tick === 0 ? 'start' : tick === 4 ? 'end' : 'middle'}, date(rows[index].date, {month: 'short', year: '2-digit'})));
    }
    keys.forEach((key) => {
      const points = rows.map((row, index) => finite(row[key]) ? `${index === 0 ? 'M' : 'L'}${x(index).toFixed(2)} ${y(row[key]).toFixed(2)}` : '').join(' ');
      svg.append(svgEl('path', {d: points, stroke: strategies[key].color, class: 'wealth-line', 'stroke-width': key === 'optimized' ? '3' : '2', 'stroke-dasharray': key === 'benchmark' ? '5 4' : 'none'}));
    });
    const last = all[all.length - 1];
    $('#chart-summary').textContent = `Final growth of $1: ${strategyOrder.filter((key) => finite(last[key])).map((key) => `${strategies[key].label.toLowerCase()} $${number(last[key], 2)}`).join(' · ')}.`;
    const guide = svgEl('line', {x1: left, x2: left, y1: top, y2: H - bottom, stroke: '#d7e1d3', 'stroke-dasharray': '3 3', visibility: 'hidden'});
    svg.append(guide);
    const circles = keys.map((key) => {
      const circle = svgEl('circle', {r: 4, fill: strategies[key].color, stroke: 'white', 'stroke-width': 2, visibility: 'hidden'});
      svg.append(circle); return {key, circle};
    });
    const overlay = svgEl('rect', {x: left, y: top, width, height, fill: 'transparent'});
    svg.append(overlay);
    const tooltip = $('#chart-tooltip');
    overlay.addEventListener('pointermove', (event) => {
      const bounds = svg.getBoundingClientRect();
      const local = (event.clientX - bounds.left) / bounds.width * W;
      const index = Math.max(0, Math.min(rows.length - 1, Math.round((local - left) / width * (rows.length - 1))));
      const row = rows[index];
      guide.setAttribute('x1', x(index)); guide.setAttribute('x2', x(index)); guide.setAttribute('visibility', 'visible');
      circles.forEach(({key, circle}) => {
        circle.setAttribute('cx', x(index)); circle.setAttribute('cy', y(row[key])); circle.setAttribute('visibility', 'visible');
      });
      tooltip.replaceChildren(el('div', 'tooltip-date', date(row.date)), ...keys.map((key) => {
        const line = el('div', 'tooltip-row');
        line.append(dot(strategies[key].color), el('span', '', strategies[key].label), el('strong', '', `$${number(row[key], 3)}`)); return line;
      }));
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(105, Math.min(bounds.width - 105, x(index) / W * bounds.width))}px`;
      tooltip.style.top = '18px';
    });
    overlay.addEventListener('pointerleave', () => {
      guide.setAttribute('visibility', 'hidden'); circles.forEach(({circle}) => circle.setAttribute('visibility', 'hidden')); tooltip.hidden = true;
    });
  }

  function setupChart() {
    $('#strategy-toggles').replaceChildren(...strategyOrder.map((key) => {
      const button = el('button', 'strategy-toggle'); button.type = 'button'; button.setAttribute('aria-pressed', 'true');
      button.append(dot(strategies[key].color), el('span', '', strategies[key].label));
      button.addEventListener('click', () => {
        if (visibleStrategies.has(key)) {
          if (visibleStrategies.size === 1) return;
          visibleStrategies.delete(key);
        } else visibleStrategies.add(key);
        button.setAttribute('aria-pressed', String(visibleStrategies.has(key))); renderWealth();
      }); return button;
    }));
    document.querySelectorAll('[data-period]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.period === chartPeriod));
      button.addEventListener('click', () => {
        chartPeriod = button.dataset.period;
        document.querySelectorAll('[data-period]').forEach((candidate) => {
          candidate.classList.toggle('selected', candidate === button); candidate.setAttribute('aria-pressed', String(candidate === button));
        }); renderWealth();
      });
    });
    renderWealth();
  }

  function renderAllocation() {
    const rows = data.terminal?.weights || [];
    const names = {SPY: 'US equity', QQQ: 'Growth tilt', IWM: 'Small caps', IEF: 'Treasuries', GLD: 'Gold'};
    $('#allocation-chart').replaceChildren(...rows.map((row) => {
      const item = el('div', 'allocation-row');
      const label = el('div', 'allocation-ticker', row.ticker); label.append(el('div', 'allocation-name', names[row.ticker] || 'ETF'));
      const bars = el('div', 'allocation-bars');
      ['no_rebalancing', 'optimized'].forEach((key) => {
        const track = el('div', 'allocation-track'); const bar = el('div', `allocation-bar ${key === 'optimized' ? 'after' : ''}`);
        bar.style.width = `${Math.max(0, Math.min(100, (row[key] || 0) * 100))}%`; track.append(bar); bars.append(track);
      });
      bars.setAttribute('aria-label', `${row.ticker}: before ${pct(row.no_rebalancing)}, optimized ${pct(row.optimized)}`);
      const values = el('div', 'allocation-values'); values.append(el('small', '', pct(row.no_rebalancing, 1)), el('span', '', pct(row.optimized, 1)));
      item.append(label, bars, values); return item;
    }));
    const constraints = [[config.max_asset_weight, 'asset cap'], [config.max_tracking_error, 'predicted TE cap'], [config.max_turnover, 'turnover cap']];
    $('#constraints').replaceChildren(...constraints.map(([value, label]) => {
      const node = el('span', 'constraint-item'); node.append(el('strong', '', pct(value, 0)), document.createTextNode(label)); return node;
    }));
    const comparison = data.terminal?.comparison || [];
    const before = comparison.find((row) => row.strategy === 'no_rebalancing') || {};
    const after = comparison.find((row) => row.strategy === 'optimized') || {};
    const terminal = [
      ['Predicted tracking error', before.predicted_tracking_error, after.predicted_tracking_error],
      ['Worst historical stress loss', before.worst_historical_stress_loss, after.worst_historical_stress_loss],
      ['One-way turnover', null, after.one_way_turnover],
    ];
    $('#terminal-summary').replaceChildren(...terminal.map(([label, original, target]) => {
      const node = el('div', 'terminal-stat'); const values = el('strong');
      if (finite(original)) values.append(el('small', '', `${pct(original)} →`)); values.append(document.createTextNode(pct(target)));
      node.append(el('span', '', label), values); return node;
    }));
  }

  function renderStress() {
    const rows = data.terminal?.stress || [];
    const keys = ['no_rebalancing', 'optimized', 'benchmark'];
    const max = Math.max(.01, ...rows.flatMap((row) => keys.map((key) => Math.abs(row[key] || 0))));
    $('#stress-chart').replaceChildren(...rows.map((row) => {
      const group = el('div', 'stress-group'); group.append(el('div', 'stress-name', row.scenario));
      keys.forEach((key) => {
        const line = el('div', 'stress-bar-row'); const track = el('div', 'stress-track'); const bar = el('div', 'stress-bar');
        bar.style.setProperty('--color', strategies[key].color); bar.style.width = `${Math.abs(row[key] || 0) / max * 100}%`; track.append(bar);
        line.setAttribute('aria-label', `${strategies[key].label}: ${pct(row[key])}`);
        line.append(track, el('span', 'stress-value', pct(row[key], 1))); group.append(line);
      }); return group;
    }));
  }

  function renderRobustness() {
    const robustness = data.robustness || {};
    const findings = robustness.findings || [];
    $('#robustness-findings').replaceChildren(...findings.map((finding) => {
      const card = el('article', `finding-card ${finding.tone || 'neutral'}`);
      card.append(el('span', 'finding-marker'), el('h3', '', finding.title), el('p', '', finding.detail)); return card;
    }));
    const rows = robustness.sensitivity || [];
    const completed = rows.filter((row) => row.status === 'completed' || row.status === 'optimal').length;
    $('#robustness-count').textContent = rows.length ? `${completed} / ${rows.length} runs completed` : 'Awaiting sensitivity runs';
    renderSensitivity();
    $('#sensitivity-filter').addEventListener('change', renderSensitivity);
    $('#sensitivity-sort').addEventListener('change', renderSensitivity);
    const bootstrap = robustness.bootstrap || {};
    if (bootstrap.caveat) $('#bootstrap-caveat').textContent = bootstrap.caveat;
    const intervals = [
      ['Annual return difference', bootstrap.annual_return_difference, 'Optimized minus no rebalancing'],
      ['Tracking-error improvement', bootstrap.tracking_error_improvement, 'Positive means lower optimized TE'],
    ];
    const statNodes = intervals.map(([label, interval, description]) => {
      const node = el('div', 'interval-row'); const copy = el('div', 'interval-label', label);
      copy.append(el('small', '', description));
      const values = el('div', 'interval-value', interval ? `${number(interval.lower * 100, 2)} to ${number(interval.upper * 100, 2)}` : '—');
      values.append(el('small', '', ' pp'));
      node.append(copy, values); return node;
    });
    statNodes.push(el('p', 'bootstrap-method', bootstrap.replicates ? `95% intervals · ${number(bootstrap.replicates, 0)} paired resamples · ${bootstrap.block_days}-day blocks · ${number(bootstrap.observations, 0)} observed days` : 'Bootstrap estimates are not available.'));
    $('#bootstrap-stats').replaceChildren(...statNodes);
  }

  function renderSensitivity() {
    const filter = $('#sensitivity-filter').value;
    const sort = $('#sensitivity-sort').value;
    let rows = [...(data.robustness?.sensitivity || [])];
    const success = (row) => ['completed', 'optimal'].includes(row.status);
    if (filter === 'feasible') rows = rows.filter(success);
    if (filter === 'failed') rows = rows.filter((row) => !success(row));
    const sortKey = {te: 'realized_tracking_error', return: 'cumulative_net_return', stress: 'mean_decision_stress_loss'}[sort];
    if (sortKey) rows.sort((a, b) => {
      if (!finite(a[sortKey])) return 1;
      if (!finite(b[sortKey])) return -1;
      return (sort === 'return' ? -1 : 1) * (a[sortKey] - b[sortKey]);
    });
    const body = $('#sensitivity-body');
    if (!rows.length) {
      const row = el('tr', 'no-results'); const cell = el('td', '', 'No matching sensitivity runs.'); cell.colSpan = 6; row.append(cell); body.replaceChildren(row); return;
    }
    body.replaceChildren(...rows.map((result) => {
      const row = el('tr'); const labelCell = el('td'); labelCell.append(el('span', 'variant-name', result.label || result.variant || 'Model variant'));
      if (result.error) labelCell.append(el('span', 'variant-error', result.error)); row.append(labelCell);
      [result.cumulative_net_return, result.realized_tracking_error, result.mean_decision_stress_loss, result.total_one_way_turnover].forEach((value) => row.append(el('td', '', pct(value))));
      const status = el('td'); status.append(el('span', `run-status ${success(result) ? '' : 'failed'}`, success(result) ? 'Completed' : result.status === 'infeasible' ? 'Infeasible' : 'Failed')); row.append(status); return row;
    }));
  }

  function renderLimitations() {
    const fallback = ['Historical comparisons depend on the ETF universe, benchmark, stress windows, and test period.', 'Covariance estimates and trading-cost assumptions are uncertain. No result guarantees future improvement.'];
    $('#limitations-list').replaceChildren(...(data.limitations?.length ? data.limitations : fallback).map((item) => el('li', '', item)));
  }

  function setupActions() {
    const dialog = $('#provenance-dialog');
    $('#provenance-button').addEventListener('click', () => dialog.showModal());
    $('#close-dialog').addEventListener('click', () => dialog.close());
    dialog.addEventListener('click', (event) => {if (event.target === dialog) {const box = dialog.getBoundingClientRect(); if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();}});
    $('#export-button').addEventListener('click', () => {
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
      const anchor = el('a'); anchor.href = url; anchor.download = 'portfolio-risk-snapshot.json'; document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    const links = [...document.querySelectorAll('.main-nav a')];
    links.forEach((link) => link.addEventListener('click', () => {links.forEach((candidate) => candidate.classList.toggle('active', candidate === link));}));
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver((entries) => {
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible[0]) links.forEach((link) => link.classList.toggle('active', link.getAttribute('href') === `#${visible[0].target.id}`));
      }, {rootMargin: '-12% 0px -55% 0px', threshold: [0, .1, .25, .5]});
      document.querySelectorAll('main>section').forEach((section) => observer.observe(section));
    }
  }

  renderMetadata(); renderConclusions(); renderMetrics(); renderComparison(); setupChart(); renderAllocation(); renderStress(); renderRobustness(); renderLimitations(); setupActions();
})();
