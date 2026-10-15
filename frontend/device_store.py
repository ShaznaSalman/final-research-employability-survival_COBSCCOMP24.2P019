"""Browser helpers as one small Streamlit component. It runs in the user's browser and does five things:

- saved profile: writes or removes the answers in localStorage on a command from the app, and reports what is stored;
- tab handoff: keeps the answers in sessionStorage (this tab only, cleared when it closes) so a page reload, such as
  the one the theme switch needs, does not lose a profile that is half filled in;
- theme: reports the theme the browser is really showing, and sets it as data-js-theme for the colour tokens;
- top bar: adds the light/dark switch and the profile avatar next to Streamlit's own toolbar buttons;
- phone menu: closes the side menu after a page is chosen.

Nothing is kept on the server. All markup the script creates is fixed text written here, never user input.
"""
from __future__ import annotations

import streamlit as st

KEY = "jobspark.profile.v1"
SESSION_KEY = "jobspark.session.v1"

JS = """
export default function (component) {
  const { data, setStateValue } = component;
  const KEY = "%(key)s";
  const SESSION_KEY = "%(session)s";
  const w = window;
  w.__jobsparkSetState = setStateValue;
  w.__jobsparkData = data || {};

  // --- saved profile on this device (localStorage) ---------------------------------------------------------------
  const cmd = data && data.command;
  if (cmd && cmd.n !== w.__jobsparkLastCommand) {          // run each command once
    w.__jobsparkLastCommand = cmd.n;
    try {
      if (cmd.op === "save") w.localStorage.setItem(KEY, JSON.stringify(cmd.answers));
      if (cmd.op === "clear") w.localStorage.removeItem(KEY);
    } catch (e) { /* storage blocked (private window): the app still works without it */ }
  }
  let stored = null;
  try {
    const raw = w.localStorage.getItem(KEY);
    stored = raw ? JSON.parse(raw) : null;
  } catch (e) { stored = null; }
  const sig = JSON.stringify(stored);
  if (sig !== w.__jobsparkStoredSig) {                      // report only when it changed
    w.__jobsparkStoredSig = sig;
    setStateValue("stored", stored);
  }

  // --- tab handoff (sessionStorage): report once at start, then keep it current ----------------------------------
  if (!w.__jobsparkSessionSent) {
    w.__jobsparkSessionSent = true;
    let sess = null;
    try {
      const raw = w.sessionStorage.getItem(SESSION_KEY);
      sess = raw ? JSON.parse(raw) : null;
    } catch (e) { sess = null; }
    setStateValue("session", sess);
  }
  if (data && data.session) {
    try { w.sessionStorage.setItem(SESSION_KEY, JSON.stringify(data.session)); } catch (e) { /* ignore */ }
  }

  // --- phone: close the side menu after a page is chosen ------------------------------------------------------------
  if (!w.__jobsparkNavClose) {
    w.__jobsparkNavClose = true;
    document.addEventListener("click", (ev) => {
      const link = ev.target.closest && ev.target.closest('[data-testid="stSidebarNavLink"]');
      if (!link || w.innerWidth >= 768) return;
      setTimeout(() => {
        const btn = document.querySelector('[data-testid="stSidebarCollapseButton"] button')
                 || document.querySelector('[data-testid="stSidebarCollapseButton"]');
        if (btn) btn.click();
      }, 250);
    }, true);
  }

  // --- the theme the browser is really showing -----------------------------------------------------------------------
  const detectTheme = () => {
    const el = document.querySelector(".stApp") || document.body;
    const m = (getComputedStyle(el).backgroundColor || "").match(/[0-9.]+/g);
    if (!m || m.length < 3) return null;
    const [r, g, b] = m.slice(0, 3).map(Number);
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.5 ? "dark" : "light";
  };

  // --- top bar: light/dark switch and profile avatar ---------------------------------------------------------------
  const ICONS = {
    moon: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 13A9 9 0 1 1 11 3a7 7 0 0 0 10 10z"/></svg>',
    sun: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    person: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/></svg>',
  };
  const switchTheme = () => {
    // Streamlit keeps its theme choice as "Light" or "Dark" under a key that includes the page path.
    const next = w.__jobsparkTheme === "dark" ? "Light" : "Dark";
    const paths = new Set(["/", "/predict", "/dashboard", "/compare", "/improve", "/ask", "/market", "/research",
                           w.location.pathname]);
    try { paths.forEach((p) => w.localStorage.setItem("stActiveTheme-" + p + "-v2", JSON.stringify(next))); } catch (e) { /* ignore */ }
    w.location.reload();
  };
  const openProfile = () => {
    const link = document.querySelector('[data-testid="stSidebarNavLink"][href$="/predict"]');
    if (link) link.click(); else w.location.assign("/predict");
  };
  const ensureTopbar = () => {
    const host = document.querySelector('[data-testid="stToolbar"]');
    if (!host) return;
    let bar = document.getElementById("js-topbar");
    if (!bar) {
      bar = document.createElement("div");
      bar.id = "js-topbar";
      bar.className = "js-topbar";
      const theme = document.createElement("button");
      theme.id = "js-theme";
      theme.type = "button";
      theme.className = "js-iconbtn";
      theme.addEventListener("click", switchTheme);
      const avatar = document.createElement("button");
      avatar.id = "js-avatar";
      avatar.type = "button";
      avatar.className = "js-avatar";
      avatar.addEventListener("click", openProfile);
      bar.appendChild(theme);
      bar.appendChild(avatar);
    }
    const anchor = host.querySelector('[data-testid="stToolbarActions"]') || host.querySelector('[data-testid="stMainMenu"]');
    if (anchor && anchor.parentNode && anchor.previousElementSibling !== bar) {
      anchor.parentNode.insertBefore(bar, anchor);                // right side, just before Deploy and the menu
    } else if (!bar.parentNode) {
      host.prepend(bar);
    }
    const dark = w.__jobsparkTheme === "dark";
    const t = document.getElementById("js-theme");
    t.innerHTML = dark ? ICONS.sun : ICONS.moon;
    t.title = dark ? "Switch to light mode" : "Switch to dark mode";
    t.setAttribute("aria-label", t.title);
    const ready = !!(w.__jobsparkData && w.__jobsparkData.profile_done);
    const a = document.getElementById("js-avatar");
    a.innerHTML = ICONS.person;
    a.classList.toggle("ready", ready);
    a.title = ready ? "My profile (complete)" : "Build my profile";
    a.setAttribute("aria-label", a.title);
  };

  const applyTheme = () => {
    const t = detectTheme();
    if (t && t !== w.__jobsparkTheme) {
      w.__jobsparkTheme = t;
      document.documentElement.setAttribute("data-js-theme", t);
      if (w.__jobsparkSetState) w.__jobsparkSetState("theme", t);
    }
    ensureTopbar();
  };
  applyTheme();
  if (!w.__jobsparkThemeTimer) w.__jobsparkThemeTimer = setInterval(applyTheme, 1200);   // re-adds the bar if Streamlit redraws the header
}
""" % {"key": KEY, "session": SESSION_KEY}

_component = st.components.v2.component("jobspark_device_store", js=JS)


def mount(session: dict | None = None, profile_done: bool = False):
    """Mount the component (call once per run, at the very top of the app).

    session: the answers to keep for this tab (or None); profile_done: whether the profile is complete (the avatar shows it).
    The result has .stored (answers saved in this browser, or None), .theme ('light' or 'dark' as displayed, or None
    until the browser has reported) and .session (what this tab had kept before a reload, or None)."""
    return _component(
        key="jobspark_device_store",
        data={"command": st.session_state.get("_store_cmd"), "session": session, "profile_done": profile_done},
        on_stored_change=lambda: None, on_theme_change=lambda: None, on_session_change=lambda: None)
