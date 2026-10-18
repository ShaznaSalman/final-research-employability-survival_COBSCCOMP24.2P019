"""Browser helpers as one small Streamlit component. It runs in the user's browser and does five things:

- saved profile: writes or removes the answers in localStorage on a command from the app, and reports what is stored;
- tab handoff: keeps the answers in sessionStorage (this tab only, cleared when it closes) so a page reload, such as
  the one the theme switch needs, does not lose a profile that is half filled in;
- theme: reports the theme the browser is really showing, and sets it as data-js-theme for the colour tokens;
- top bar: adds the light/dark switch and profile avatar; the avatar menu sends selected identity changes back to Python;
- phone menu: closes the side menu after a page is chosen.

Profile answers and tab handoff remain browser-side; account identity is persisted separately through the profile service.
All menu markup is fixed text, and user-provided names are assigned through input values rather than HTML.
"""
from __future__ import annotations

import streamlit as st

KEY = "jobspark.profile.v1"
SESSION_KEY = "jobspark.session.v1"
AUTH_KEY = "jobspark.profile.auth.v1"

JS = """
export default function (component) {
  const { data, setStateValue } = component;
  const KEY = "%(key)s";
  const SESSION_KEY = "%(session)s";
  const AUTH_KEY = "%(auth)s";
  const w = window;
  w.__jobsparkSetState = setStateValue;
  w.__jobsparkSetTriggerValue = component.setTriggerValue;
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
  if (!w.__jobsparkAuthSent) {
    w.__jobsparkAuthSent = true;
    let token = null;
    try { token = w.sessionStorage.getItem(AUTH_KEY); } catch (e) { /* storage can be blocked */ }
    setStateValue("auth_token", token);
  }
  const authSync = data && data.auth_sync;
  if (authSync && authSync.n !== w.__jobsparkLastAuthSync) {
    w.__jobsparkLastAuthSync = authSync.n;
    setStateValue("auth_token", authSync.token || null);
    try {
      if (authSync.token) w.sessionStorage.setItem(AUTH_KEY, authSync.token);
      else w.sessionStorage.removeItem(AUTH_KEY);
    } catch (e) { /* sign-in remains available for this Streamlit session */ }
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
    const paths = new Set(["/", "/predict", "/dashboard", "/compare", "/career_path", "/account", "/improve", "/ask", "/market", "/research",
                           w.location.pathname]);
    try { paths.forEach((p) => w.localStorage.setItem("stActiveTheme-" + p + "-v2", JSON.stringify(next))); } catch (e) { /* ignore */ }
    w.location.reload();
  };
  const closeProfileMenu = () => {
    const menu = document.getElementById("js-profile-menu");
    if (menu) menu.remove();
  };
  const addProfileMenuStyles = () => {
    if (document.getElementById("js-profile-menu-styles")) return;
    const style = document.createElement("style");
    style.id = "js-profile-menu-styles";
    style.textContent = `
      #js-profile-menu { position:fixed; z-index:99999; top:68px; right:18px; width:min(320px,calc(100vw - 28px));
        padding:18px; border:1px solid rgba(130,150,190,.35); border-radius:16px;
        background:var(--panel,#fff); color:var(--ink,#1d2940); box-shadow:0 16px 42px rgba(15,23,42,.28);
        font:14px/1.45 system-ui,sans-serif; }
      #js-profile-menu h2 { margin:0 0 14px; font-size:18px; }
      #js-profile-menu label,#js-profile-menu .js-profile-label { display:block; margin:10px 0 6px; font-weight:600; }
      #js-profile-menu input { box-sizing:border-box; width:100%%; padding:9px 10px; border:1px solid #9aa8be;
        border-radius:8px; background:var(--panel,#fff); color:inherit; font:inherit; }
      #js-profile-menu .js-profile-avatars { display:grid; grid-template-columns:repeat(6,1fr); gap:7px; }
      #js-profile-menu .js-profile-avatars button { min-height:38px; border:1px solid rgba(130,150,190,.4);
        border-radius:9px; background:transparent; font-size:21px; cursor:pointer; }
      #js-profile-menu .js-profile-avatars button.selected { outline:2px solid #6c8fff; background:rgba(108,143,255,.16); }
      #js-profile-menu .js-profile-note { min-height:20px; margin:10px 0; color:inherit; opacity:.8; }
      #js-profile-menu .js-profile-save { width:100%%; padding:10px 12px; border:0; border-radius:9px;
        background:var(--brand,#4263eb); color:var(--on-brand,#fff); font:600 14px system-ui,sans-serif; cursor:pointer; }
      #js-profile-menu .js-profile-save:hover { filter:brightness(1.08); }
    `;
    document.head.appendChild(style);
  };
  const openProfileMenu = () => {
    const existing = document.getElementById("js-profile-menu");
    if (existing) {
      closeProfileMenu();
      return;
    }
    const menu = document.createElement("section");
    addProfileMenuStyles();
    menu.id = "js-profile-menu";
    menu.setAttribute("role", "dialog");
    menu.setAttribute("aria-label", "Profile settings");
    menu.innerHTML = `
      <h2>Profile settings</h2>
      <label for="js-profile-name">Name or username</label>
      <input id="js-profile-name" maxlength="40" autocomplete="nickname">
      <p class="js-profile-label">Choose an avatar</p>
      <div class="js-profile-avatars" role="group" aria-label="Choose an avatar"></div>
      <p class="js-profile-note"></p>
      <button class="js-profile-save" type="button"></button>
    `;
    const nameInput = menu.querySelector("#js-profile-name");
    nameInput.value = (w.__jobsparkData && w.__jobsparkData.display_name) || "";
    const selectedEmoji = {value: (w.__jobsparkData && w.__jobsparkData.avatar_emoji) || "🙂"};
    const avatars = ["🙂", "😀", "😎", "🤓", "🧑‍💻", "👩‍💻", "👨‍💻", "🚀", "🌱", "⭐", "🐼", "🦊"];
    const avatarList = menu.querySelector(".js-profile-avatars");
    avatars.forEach((emoji) => {
      const option = document.createElement("button");
      option.type = "button";
      option.textContent = emoji;
      option.setAttribute("aria-label", `Choose ${emoji}`);
      option.setAttribute("aria-pressed", String(emoji === selectedEmoji.value));
      option.className = emoji === selectedEmoji.value ? "selected" : "";
      option.addEventListener("click", () => {
        selectedEmoji.value = emoji;
        avatarList.querySelectorAll("button").forEach((item) => {
          const selected = item.textContent === emoji;
          item.classList.toggle("selected", selected);
          item.setAttribute("aria-pressed", String(selected));
        });
      });
      avatarList.appendChild(option);
    });
    const save = menu.querySelector(".js-profile-save");
    const signedIn = !!(w.__jobsparkData && w.__jobsparkData.signed_in);
    const note = menu.querySelector(".js-profile-note");
    if (signedIn) {
      save.textContent = "Save profile appearance";
      save.addEventListener("click", () => {
        w.__jobsparkSetTriggerValue("identity_update", {
          display_name: nameInput.value.trim(),
          avatar_emoji: selectedEmoji.value,
          nonce: Date.now(),
        });
        closeProfileMenu();
      });
    } else {
      note.textContent = "Sign in to save profile appearance across devices.";
      save.textContent = "Sign in to save";
      save.addEventListener("click", () => {
        const link = document.querySelector('[data-testid="stSidebarNavLink"][href$="/account"]');
        if (link) link.click(); else w.location.assign("/account");
      });
    }
    document.body.appendChild(menu);
    nameInput.focus();
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
      avatar.setAttribute("aria-haspopup", "dialog");
      avatar.setAttribute("aria-expanded", "false");
      avatar.addEventListener("click", openProfileMenu);
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
    a.textContent = (w.__jobsparkData && w.__jobsparkData.avatar_emoji) || "🙂";
    a.classList.toggle("ready", ready);
    const name = w.__jobsparkData && w.__jobsparkData.display_name;
    a.title = name ? `Profile settings for ${name}` : "Profile settings";
    a.setAttribute("aria-label", a.title);
    a.setAttribute("aria-expanded", String(!!document.getElementById("js-profile-menu")));
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
  if (!w.__jobsparkProfileMenuEvents) {
    w.__jobsparkProfileMenuEvents = true;
    document.addEventListener("click", (event) => {
      const menu = document.getElementById("js-profile-menu");
      if (menu && !menu.contains(event.target) && event.target.id !== "js-avatar") menu.remove();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeProfileMenu();
    });
  }
  if (!w.__jobsparkThemeTimer) w.__jobsparkThemeTimer = setInterval(applyTheme, 1200);   // re-adds the bar if Streamlit redraws the header
}
""" % {"key": KEY, "session": SESSION_KEY, "auth": AUTH_KEY}

_component = st.components.v2.component("jobspark_device_store", js=JS)


def mount(session: dict | None = None, profile_done: bool = False, display_name: str = "",
          avatar_emoji: str = "🙂", signed_in: bool = False, auth_sync: dict | None = None):
    """Mount the component (call once per run, at the very top of the app).

    session: the answers to keep for this tab (or None); profile_done: whether the profile is complete (the avatar shows it).
    The result has .stored (answers saved in this browser, or None), .theme ('light' or 'dark' as displayed, or None
    until the browser has reported) and .session (what this tab had kept before a reload, or None)."""
    return _component(
        key="jobspark_device_store",
        data={"command": st.session_state.get("_store_cmd"), "session": session, "profile_done": profile_done,
              "display_name": display_name, "avatar_emoji": avatar_emoji, "signed_in": signed_in,
              "auth_sync": auth_sync},
        on_stored_change=lambda: None, on_theme_change=lambda: None, on_session_change=lambda: None,
        on_identity_update_change=lambda: None, on_auth_token_change=lambda: None)


def sync_auth_token(token: str | None) -> None:
    """Queue a browser-tab auth storage update after sign-in, sign-out, or session expiry."""
    nonce = st.session_state.get("_auth_sync_nonce", 0) + 1
    st.session_state["_auth_sync_nonce"] = nonce
    st.session_state["_auth_sync"] = {"n": nonce, "token": token or ""}
