from shiny import ui

from ..theme import load_obsidian_css


_THEME_SYNC_JS = """
(function () {
  var DARK_BG = '#1e1e1e';
  var DARK_FG = '#dcddde';
  var LIGHT_BG = '#f8f8f800';

  // --- Rewrite bslib's style.css rule in place -------------------------
  // The inline-!important strategy below should win the cascade, but some
  // browsers still display the style.css source rule in DevTools. Walking
  // the CSSOM and patching the --bslib-sidebar-main-bg value directly
  // makes the change visible at the source as well.
  function patchBslibStylesheet() {
    for (var i = 0; i < document.styleSheets.length; i++) {
      var sheet = document.styleSheets[i];
      var rules;
      try { rules = sheet.cssRules || sheet.rules; } catch (e) { continue; }
      if (!rules) continue;
      for (var j = 0; j < rules.length; j++) {
        var rule = rules[j];
        if (!rule || !rule.style) continue;
        try {
          if (rule.style.getPropertyValue('--bslib-sidebar-main-bg')) {
            rule.style.setProperty('--bslib-sidebar-main-bg', LIGHT_BG, 'important');
          }
        } catch (e) { /* cross-origin or read-only */ }
      }
    }
  }
  patchBslibStylesheet();
  [50, 200, 600, 1500, 3000].forEach(function (ms) { setTimeout(patchBslibStylesheet, ms); });

  // --- Append an override <style> at the end of <head> so it wins source
  // order against bslib's bundled stylesheet.
  var override = document.createElement('style');
  override.setAttribute('data-tt-override', 'bslib-sidebar-main-bg');
  override.textContent =
    ':root, .bslib-sidebar-layout, html .bslib-sidebar-layout, html body .bslib-sidebar-layout {' +
    '  --bslib-sidebar-main-bg: ' + LIGHT_BG + ' !important;' +
    '}';
  (document.head || document.documentElement).appendChild(override);

  var SELECTORS = [
    'html',
    'body',
    'main.bslib-page-main',
    'div.main',
    '.bslib-sidebar-layout',
    '.bslib-sidebar-layout > .main',
    '.bslib-page-fill',
    '.bslib-page-sidebar',
    '.tab-content'
  ];

  function applyTheme() {
    var isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
    SELECTORS.forEach(function (sel) {
      try {
        document.querySelectorAll(sel).forEach(function (el) {
          el.style.setProperty('background-color', isDark ? DARK_BG : '', 'important');
          el.style.setProperty('color', isDark ? DARK_FG : '', 'important');
        });
      } catch (e) { /* ignore bad selectors */ }
    });
    // Tab panes are coloured via theme-scoped CSS rules (see OBSIDIAN_CSS),
    // not inline styles — clear any stale inline bg left by older builds so
    // a dark→light toggle never leaks dark blocks into the light layout.
    document.querySelectorAll('.tab-pane').forEach(function (el) {
      el.style.removeProperty('background-color');
      el.style.removeProperty('color');
    });
    // bslib reads --_main-bg / --bslib-sidebar-main-bg off .bslib-sidebar-layout
    // to colour the .main container. Force them inline so nothing can override.
    // In light mode, use #f8f8f800 (fully transparent) instead of clearing —
    // clearing falls back to bslib's style.css default of #f8f8f8 (opaque).
    var LIGHT_TRANSPARENT = '#f8f8f800';
    document.querySelectorAll('.bslib-sidebar-layout').forEach(function (el) {
      el.style.setProperty('--_main-bg', isDark ? DARK_BG : LIGHT_TRANSPARENT, 'important');
      el.style.setProperty('--bslib-sidebar-main-bg', isDark ? DARK_BG : LIGHT_TRANSPARENT, 'important');
      el.style.setProperty('--_main-fg', isDark ? DARK_FG : '', 'important');
    });
  }

  new MutationObserver(applyTheme).observe(
    document.documentElement,
    { attributes: true, attributeFilter: ['data-bs-theme'] }
  );
  // Run immediately, plus on DOM ready, plus on a few post-load ticks to
  // catch async-injected bslib containers.
  applyTheme();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyTheme);
  }
  window.addEventListener('load', applyTheme);
  [50, 200, 600, 1500, 3000].forEach(function (ms) { setTimeout(applyTheme, ms); });
})();
"""

_TOOLTIP_QUICKSTART_JS = """
(function () {
  // Stationary tooltip: appears only after 2s of mouse staying still on the element.
  var DELAY_MS = 2000;
  var activeTip = null;
  var activeTimer = null;

  function clearActive() {
    if (activeTimer) { clearTimeout(activeTimer); activeTimer = null; }
    if (activeTip) { activeTip.remove(); activeTip = null; }
  }

  function scheduleTip(el, e) {
    clearActive();
    var text = el.getAttribute('data-tt-help');
    if (!text) return;
    var pageX = e.pageX, pageY = e.pageY;
    activeTimer = setTimeout(function () {
      var tip = document.createElement('div');
      tip.className = 'tt-stationary-tooltip';
      tip.textContent = text;
      tip.style.left = (pageX + 14) + 'px';
      tip.style.top = (pageY + 14) + 'px';
      document.body.appendChild(tip);
      activeTip = tip;
    }, DELAY_MS);
  }

  document.addEventListener('mousemove', function (e) {
    var el = e.target.closest && e.target.closest('[data-tt-help]');
    if (!el) { clearActive(); return; }
    scheduleTip(el, e);
  }, true);
  document.addEventListener('mouseleave', clearActive, true);
  window.addEventListener('blur', clearActive);
})();

(function () {
  // Quick-start floating panel: clickable header toggles open/close.
  // Event delegation on document.documentElement in CAPTURE phase
  // so that stopPropagation() in Shiny/Bootstrap handlers cannot block it.
  document.documentElement.addEventListener('click', function (e) {
    var header = e.target.closest && e.target.closest('.qs-header');
    if (!header) return;
    var qs = header.closest && header.closest('#tt-quickstart');
    if (qs) qs.classList.toggle('qs-open');
  }, true);
})();
"""


def head_content():
    return ui.head_content(
        # Empty inline favicon so the browser stops requesting /favicon.ico.
        ui.tags.link(rel="icon", href="data:,"),
        ui.tags.style(load_obsidian_css()),
        ui.tags.script(_THEME_SYNC_JS),
        ui.tags.script(_TOOLTIP_QUICKSTART_JS),
    )
