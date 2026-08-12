/**
 * Shared Chart.js theming for Edge-Bench.
 *
 * Colours are read from the CSS custom properties in style.css, so charts
 * follow the light/dark theme and use the same data-series palette as the
 * rest of the dashboard instead of a second, hardcoded one.
 *
 * Requires vendor/chart.umd.min.js to be loaded first.
 */
(function (global) {
    'use strict';

    function cssVar(name, fallback) {
        var value = getComputedStyle(document.body).getPropertyValue(name);
        return (value || '').trim() || fallback;
    }

    /** The categorical palette, in the order series should be assigned. */
    function seriesColors() {
        return [
            cssVar('--series-1', '#58a6ff'),
            cssVar('--series-2', '#3fb950'),
            cssVar('--series-3', '#d9a441'),
            cssVar('--series-4', '#a78bfa'),
            cssVar('--series-5', '#f778ba'),
            cssVar('--series-6', '#56d4dd')
        ];
    }

    /** Colour for series index i, cycling through the palette. */
    function seriesColor(i) {
        var palette = seriesColors();
        return palette[i % palette.length];
    }

    /**
     * Base Chart.js options shared by every chart in the app.
     * @param {string} yLabel axis title, e.g. 'ms' or 'FPS'
     */
    function baseOptions(yLabel) {
        var text = cssVar('--text', '#e3e9f2');
        var muted = cssVar('--text-muted', '#8b98ac');
        var border = cssVar('--border', '#232c39');
        var surface = cssVar('--bg-elevated', '#18202b');
        var grid = cssVar('--border', '#232c39');

        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 220 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: surface,
                    titleColor: text,
                    bodyColor: text,
                    borderColor: border,
                    borderWidth: 1,
                    padding: 10,
                    cornerRadius: 8,
                    displayColors: false,
                    titleFont: { size: 12, weight: '600' },
                    bodyFont: { family: 'ui-monospace, monospace', size: 12 }
                }
            },
            scales: {
                x: {
                    grid: { display: false, drawBorder: false },
                    ticks: {
                        color: muted,
                        maxRotation: 40,
                        font: { size: 11 },
                        autoSkip: true
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: grid, drawBorder: false },
                    border: { display: false },
                    ticks: {
                        color: muted,
                        font: { family: 'ui-monospace, monospace', size: 11 }
                    },
                    title: yLabel
                        ? { display: true, text: yLabel, color: muted, font: { size: 11 } }
                        : { display: false }
                }
            }
        };
    }

    /** Standard bar dataset styling for a comparison chart. */
    function barDataset(values) {
        return {
            data: values,
            backgroundColor: values.map(function (_, i) { return seriesColor(i); }),
            borderRadius: 5,
            borderSkipped: false,
            maxBarThickness: 56
        };
    }

    global.EdgeBenchCharts = {
        cssVar: cssVar,
        seriesColors: seriesColors,
        seriesColor: seriesColor,
        baseOptions: baseOptions,
        barDataset: barDataset
    };
})(window);
