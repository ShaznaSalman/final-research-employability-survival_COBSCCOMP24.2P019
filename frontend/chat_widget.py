"""Spark, the floating chat assistant: a small robot fixed in the bottom corner of every page (except the Ask page).

Clicking the robot opens a chat panel. The answers come from the same grounded assistant as the Ask page (POST /chat):
no language model, and every number comes from the system's own results. Nothing typed is stored beyond this session.

How it is built: a small Streamlit component draws the robot and the panel in the browser (fixed position, so it
floats above every page), and sends each question to Python as a trigger value. The Python side runs inside a
fragment, so sending a question re-runs only this widget, not the page behind it.
"""
from __future__ import annotations

import streamlit as st

from core.chat import PROFILE_FREE_INTENTS, classify
from frontend import profile as prof

STARTERS = ["Why is my estimate like this?", "What if I do a 6-month internship?",
            "Which skills should I learn?", "Can I trust this estimate?"]
BASIC_STARTERS = ["What can I ask?", "How does the model work?", "Do you store my data?", "Is GPA important?"]
NOT_READY = ("I can answer general questions now. To explain your estimate or test changes for your situation, "
             "complete the four short steps under **My profile**.")
UNREACHABLE = "Sorry, I could not reach the results service just now. Please try again in a moment."
MAX_CHARS = 300

# The script is a plain string (no % formatting, no f-string) because the CSS and JS use % and braces freely.
JS = r"""
export default function (component) {
  const { data, setTriggerValue } = component;
  const w = window, d = document;
  const S = (w.__jsChat = w.__jsChat || { open: false, pending: false, seen: 0, optimistic: null, tipDone: false });
  w.__jsChatTrigger = setTriggerValue;
  const info = data || {};
  if (info.open_request && info.open_request !== S.openRequest) {
    S.openRequest = info.open_request;
    S.open = true;
    S.tipDone = true;
  }

  const CSS = `
  #js-chat { position: fixed; right: 24px; bottom: 20px; z-index: 99990; font-family: 'Inter','Segoe UI',sans-serif; }
  #js-chat[hidden] { display: none !important; }
  .jc-fab { position: relative; width: 78px; height: 92px; border: 0; background: none; padding: 0; cursor: pointer; display: block; margin-left: auto; }
  .jc-bob { display: block; animation: jcbob 3.2s ease-in-out infinite; filter: drop-shadow(0 10px 14px rgba(29,82,207,.32)); }
  .jc-fab:hover .jc-bob { animation-duration: 1.6s; }
  .jc-fab:focus-visible { outline: 3px solid var(--brand, #3b66e0); outline-offset: 4px; border-radius: 18px; }
  .jc-shadow { position: absolute; left: 50%; bottom: 2px; width: 46px; height: 8px; margin-left: -23px; border-radius: 50%;
    background: rgba(21,32,58,.28); filter: blur(2px); animation: jcshadow 3.2s ease-in-out infinite; }
  .jc-eye { transform-box: fill-box; transform-origin: center; animation: jcblink 5s infinite; }
  .jc-ant { transform-box: fill-box; transform-origin: 50% 100%; animation: jcsway 3.2s ease-in-out infinite; }
  .jc-dot { position: absolute; right: 6px; top: 6px; width: 14px; height: 14px; border-radius: 50%; background: #19b394;
    border: 2px solid var(--wash, #f5f7fb); }
  @keyframes jcbob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-12px); } }
  @keyframes jcshadow { 0%,100% { transform: scale(1); opacity: .9; } 50% { transform: scale(.7); opacity: .5; } }
  @keyframes jcblink { 0%,92%,100% { transform: scaleY(1); } 95% { transform: scaleY(.1); } }
  @keyframes jcsway { 0%,100% { transform: rotate(-6deg); } 50% { transform: rotate(6deg); } }
  .jc-tip { position: absolute; right: 70px; bottom: 44px; white-space: nowrap; background: var(--panel, #fff); color: var(--ink, #15203a);
    border: 1px solid var(--line, #e3e8f1); border-radius: 14px 14px 4px 14px; padding: .5rem .8rem; font-size: .85rem;
    box-shadow: 0 10px 26px rgba(21,32,58,.18); animation: jcpop .35s ease-out; }
  .jc-tip[hidden] { display: none; }
  @keyframes jcpop { from { opacity: 0; transform: translateY(6px) scale(.96); } to { opacity: 1; transform: none; } }
  .jc-panel { position: absolute; right: 0; bottom: 104px; width: min(380px, calc(100vw - 32px)); max-height: min(580px, calc(100vh - 150px));
    display: flex; flex-direction: column; background: var(--panel, #fff); color: var(--ink, #15203a); border: 1px solid var(--line, #e3e8f1);
    border-radius: 20px; box-shadow: 0 26px 64px rgba(21,32,58,.32); overflow: hidden; animation: jcpop .22s ease-out; }
  .jc-panel[hidden] { display: none; }
  .jc-head { display: flex; align-items: center; gap: .6rem; padding: .8rem .9rem; color: #fff;
    background: linear-gradient(135deg, var(--hero-a, #1d52cf), var(--hero-b, #3069e3)); }
  .jc-head .jc-title { flex: 1; line-height: 1.2; min-width: 0; }
  .jc-head .jc-title b { display: block; font-family: 'Plus Jakarta Sans','Inter',sans-serif; font-size: 1.02rem; }
  .jc-head .jc-title span { font-size: .78rem; opacity: .9; }
  .jc-head button { background: rgba(255,255,255,.16); border: 0; color: #fff; border-radius: 10px; cursor: pointer; font-size: .8rem; padding: .3rem .6rem; min-height: 32px; }
  .jc-head button:hover { background: rgba(255,255,255,.28); }
  .jc-head .jc-x { font-size: 1.25rem; line-height: 1; width: 34px; padding: 0; }
  .jc-msgs { flex: 1; overflow-y: auto; padding: .9rem; display: flex; flex-direction: column; gap: .55rem; min-height: 160px; }
  .jc-b { max-width: 88%; padding: .55rem .8rem; border-radius: 14px; font-size: .9rem; line-height: 1.45; word-wrap: break-word; }
  .jc-b p { margin: 0 0 .35rem; } .jc-b p:last-child { margin-bottom: 0; }
  .jc-b ul { margin: .2rem 0 .2rem 1.1rem; padding: 0; } .jc-b li { margin: .1rem 0; }
  .jc-b.bot { align-self: flex-start; background: var(--blue-soft, #e8effe); color: var(--ink, #15203a); border-bottom-left-radius: 4px; }
  .jc-b.me { align-self: flex-end; background: var(--brand, #3b66e0); color: var(--on-brand, #fff); border-bottom-right-radius: 4px; }
  .jc-dots span { display: inline-block; width: 7px; height: 7px; margin: 0 2px; border-radius: 50%; background: var(--muted, #5a6682); animation: jcdot 1.1s infinite; }
  .jc-dots span:nth-child(2) { animation-delay: .18s; } .jc-dots span:nth-child(3) { animation-delay: .36s; }
  @keyframes jcdot { 0%,80%,100% { transform: translateY(0); opacity: .4; } 40% { transform: translateY(-4px); opacity: 1; } }
  .jc-chips { display: flex; flex-wrap: wrap; gap: .4rem; padding: 0 .9rem .6rem; }
  .jc-chips button, .jc-cta { border: 1px solid var(--line, #e3e8f1); background: var(--panel, #fff); color: var(--ink, #15203a); border-radius: 999px;
    padding: .38rem .75rem; font-size: .82rem; cursor: pointer; text-align: left; }
  .jc-chips button:hover, .jc-cta:hover { background: var(--blue-soft, #e8effe); }
  .jc-cta { background: var(--brand, #3b66e0); color: var(--on-brand, #fff); border: 0; font-weight: 600; align-self: flex-start; }
  .jc-cta:hover { filter: brightness(1.08); background: var(--brand, #3b66e0); }
  .jc-form { display: flex; gap: .5rem; padding: .6rem .9rem; border-top: 1px solid var(--line, #e3e8f1); }
  .jc-form input { flex: 1; min-width: 0; border: 1px solid var(--line, #e3e8f1); border-radius: 12px; padding: .55rem .75rem; font-size: .9rem;
    background: var(--wash, #f5f7fb); color: var(--ink, #15203a); }
  .jc-form input:focus-visible { outline: 3px solid var(--brand, #3b66e0); outline-offset: 1px; }
  .jc-form button { width: 42px; border: 0; border-radius: 12px; cursor: pointer; background: var(--brand, #3b66e0); color: var(--on-brand, #fff); display: grid; place-items: center; }
  .jc-form button:disabled { opacity: .5; cursor: default; }
  .jc-foot { padding: 0 .9rem .7rem; font-size: .72rem; color: var(--muted, #5a6682); }
  @media (max-width: 640px) { #js-chat { right: 10px; bottom: 10px; } .jc-fab { width: 64px; height: 76px; } .jc-fab svg { width: 64px; height: 70px; }
    .jc-panel { position: fixed; left: 10px; right: 10px; bottom: 92px; width: auto; } .jc-tip { display: none; } }
  @media (prefers-reduced-motion: reduce) { .jc-bob, .jc-shadow, .jc-eye, .jc-ant, .jc-dots span, .jc-tip, .jc-panel { animation: none !important; } }
  `;
  if (!d.getElementById("js-chat-style")) {
    const st = d.createElement("style");
    st.id = "js-chat-style";
    st.textContent = CSS;
    d.head.appendChild(st);
  }

  const ROBOT = `<svg viewBox="0 0 78 84" width="78" height="84" aria-hidden="true">
    <defs><linearGradient id="jc-body" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2f66e8"/><stop offset="1" stop-color="#7c5cff"/></linearGradient>
    <linearGradient id="jc-head" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#dfe8ff"/></linearGradient></defs>
    <g class="jc-ant"><line x1="39" y1="6" x2="39" y2="17" stroke="#9fb4ee" stroke-width="3" stroke-linecap="round"/><circle cx="39" cy="6" r="5" fill="#ffa62b"/><circle cx="37.6" cy="4.6" r="1.6" fill="#fff0cc"/></g>
    <rect x="9" y="30" width="7" height="16" rx="3.5" fill="#7c5cff"/><rect x="62" y="30" width="7" height="16" rx="3.5" fill="#7c5cff"/>
    <rect x="14" y="16" width="50" height="40" rx="16" fill="url(#jc-head)" stroke="#c4d3fb" stroke-width="1.5"/>
    <rect x="20" y="23" width="38" height="26" rx="11" fill="#13235a"/>
    <ellipse class="jc-eye" cx="31" cy="35" rx="4.2" ry="5" fill="#3ee0c0"/><ellipse class="jc-eye" cx="47" cy="35" rx="4.2" ry="5" fill="#3ee0c0"/>
    <path d="M32 43 q7 5 14 0" fill="none" stroke="#3ee0c0" stroke-width="2.4" stroke-linecap="round"/>
    <circle cx="25" cy="41" r="2.2" fill="#ff8a8a" opacity=".55"/><circle cx="53" cy="41" r="2.2" fill="#ff8a8a" opacity=".55"/>
    <rect x="22" y="58" width="34" height="18" rx="9" fill="url(#jc-body)"/>
    <circle cx="39" cy="67" r="3.2" fill="#ffc86b"/><rect x="14" y="60" width="9" height="6" rx="3" fill="#2f66e8"/><rect x="55" y="60" width="9" height="6" rx="3" fill="#2f66e8"/>
  </svg>`;
  const FACE = `<svg viewBox="0 0 40 40" width="34" height="34" aria-hidden="true"><rect x="3" y="5" width="34" height="30" rx="12" fill="#fff"/>
    <rect x="8" y="10" width="24" height="19" rx="8" fill="#13235a"/><circle cx="15.5" cy="18.5" r="2.8" fill="#3ee0c0"/><circle cx="24.5" cy="18.5" r="2.8" fill="#3ee0c0"/>
    <path d="M16 24 q4 3.4 8 0" fill="none" stroke="#3ee0c0" stroke-width="1.8" stroke-linecap="round"/></svg>`;
  const SEND = `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4z"/></svg>`;

  const esc = (t) => String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const bold = (t) => t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  const fmt = (text) => {                                   // minimal Markdown: **bold**, "- " bullets, paragraphs (escaped first)
    let out = "", list = false;
    for (const line of esc(text).split("\n")) {
      const m = line.match(/^\s*[-*]\s+(.*)$/);
      if (m) { if (!list) { out += "<ul>"; list = true; } out += "<li>" + bold(m[1]) + "</li>"; }
      else { if (list) { out += "</ul>"; list = false; } if (line.trim()) out += "<p>" + bold(line) + "</p>"; }
    }
    return list ? out + "</ul>" : out;
  };

  const send = (text) => {
    text = String(text || "").trim().slice(0, 300);
    if (!text || S.pending) return;
    S.pending = true;
    S.optimistic = text;
    S.seen = ((w.__jsChatData || {}).history || []).length;
    w.__jsChatTrigger("message", { text: text, n: Date.now() });
    paint();
  };
  const goProfile = () => {
    const link = d.querySelector('[data-testid="stSidebarNavLink"][href$="/predict"]');
    if (link) link.click();
    S.open = false;
    paint();
  };

  const build = () => {
    if (d.getElementById("js-chat")) return;
    const root = d.createElement("div");
    root.id = "js-chat";
    root.innerHTML = `
      <div class="jc-tip" id="jc-tip" hidden>Hi, I'm Spark! Ask me about your result.</div>
      <div class="jc-panel" id="jc-panel" role="dialog" aria-label="Chat with Spark" hidden>
        <div class="jc-head">${FACE}<div class="jc-title"><b>Spark</b><span>Your results assistant</span></div>
          <button type="button" id="jc-clear">Clear</button><button type="button" class="jc-x" id="jc-close" aria-label="Close chat">&times;</button></div>
        <div class="jc-msgs" id="jc-msgs" aria-live="polite"></div>
        <div class="jc-chips" id="jc-chips"></div>
        <form class="jc-form" id="jc-form"><input id="jc-input" type="text" maxlength="300" autocomplete="off" placeholder="Ask about your estimate..." aria-label="Your question">
          <button type="submit" id="jc-send" aria-label="Send">${SEND}</button></form>
        <div class="jc-foot">Answers come only from this system's own results, not a general chatbot. Nothing you type is stored.</div>
      </div>
      <button class="jc-fab" id="jc-fab" type="button" aria-label="Open chat assistant" aria-expanded="false">
        <span class="jc-bob">${ROBOT}</span><span class="jc-shadow"></span><span class="jc-dot" id="jc-dot" hidden></span></button>`;
    d.body.appendChild(root);
    const $ = (id) => d.getElementById(id);
    $("jc-fab").addEventListener("click", () => { S.open = !S.open; S.tipDone = true; paint(); if (S.open) setTimeout(() => $("jc-input").focus(), 60); });
    $("jc-close").addEventListener("click", () => { S.open = false; paint(); $("jc-fab").focus(); });
    $("jc-clear").addEventListener("click", () => { S.pending = false; S.optimistic = null; w.__jsChatTrigger("message", { op: "clear", n: Date.now() }); });
    $("jc-form").addEventListener("submit", (e) => { e.preventDefault(); const i = $("jc-input"); send(i.value); i.value = ""; });
    root.addEventListener("keydown", (e) => { if (e.key === "Escape" && S.open) { S.open = false; paint(); $("jc-fab").focus(); } });
    $("jc-msgs").addEventListener("click", (e) => { if (e.target.closest && e.target.closest(".jc-cta")) goProfile(); });
    $("jc-chips").addEventListener("click", (e) => { const b = e.target.closest && e.target.closest("button"); if (b) send(b.textContent); });
    setTimeout(() => { if (!S.open && !S.tipDone) { const t = $("jc-tip"); if (t) { t.hidden = false; setTimeout(() => { t.hidden = true; S.tipDone = true; }, 9000); } } }, 2500);
  };

  function paint() {
    build();
    const cur = w.__jsChatData || {};
    const root = d.getElementById("js-chat");
    root.hidden = !!cur.hidden;
    const $ = (id) => d.getElementById(id);
    $("jc-panel").hidden = !S.open;
    $("jc-fab").setAttribute("aria-expanded", S.open ? "true" : "false");
    if (S.open) $("jc-tip").hidden = true;
    $("jc-dot").hidden = !cur.ready;

    const history = cur.history || [];
    if (S.pending && history.length > S.seen) { S.pending = false; S.optimistic = null; }   // the answer has arrived
    let html = `<div class="jc-b bot"><p>Hi, I'm <strong>Spark</strong>.</p>` +
      (cur.ready ? `<p>Ask me about your estimate, what is helping or holding you back, or what small changes might do.</p></div>`
                 : `<p>Ask general questions about the model, privacy, or factors that matter across graduates. Complete your profile for answers about your own estimate.</p></div><button type="button" class="jc-cta">Build my profile</button>`);
    for (const [role, text] of history) {
      html += role === "user" ? `<div class="jc-b me">${esc(text)}</div>` : `<div class="jc-b bot">${fmt(text)}</div>`;
    }
    if (S.pending && S.optimistic) html += `<div class="jc-b me">${esc(S.optimistic)}</div>`;
    if (S.pending) html += `<div class="jc-b bot"><span class="jc-dots"><span></span><span></span><span></span></span></div>`;
    const msgs = $("jc-msgs");
    msgs.innerHTML = html;
    msgs.scrollTop = msgs.scrollHeight;

    const chips = history.length ? (cur.followups || []) : (cur.starters || []);
    $("jc-chips").innerHTML = S.pending ? "" : chips.slice(0, 4).map((c) => `<button type="button">${esc(c)}</button>`).join("");
    $("jc-send").disabled = !!S.pending;
  }

  w.__jsChatData = info;
  paint();
  if (info.open_request && info.open_request === S.openRequest) {
    setTimeout(() => {
      const input = d.getElementById("jc-input");
      if (S.open && input) input.focus();
    }, 60);
  }
}
"""

_component = st.components.v2.component("jobspark_chat", js=JS)


def reply_for(text: str, ready: bool, call) -> tuple[str, list[str]]:
    """The assistant's answer and its suggested follow-up questions. `call` is frontend.profile.call (path, payload)
    -> (result, error). Without a finished profile the answer is a fixed prompt to build one."""
    if not ready:
        if classify(text) not in PROFILE_FREE_INTENTS:
            return NOT_READY, []
        profile = None
    else:
        profile = prof.build_profile()
    answer, err = call("/chat", {"profile": profile, "message": text})
    if err or not answer:
        return UNREACHABLE, []
    return answer["reply"], list(answer.get("follow_ups") or [])


@st.fragment
def _widget(hidden: bool) -> None:
    S = st.session_state
    history = S.setdefault("chat_history", [])
    ready = bool(S.get("profile_done"))
    result = _component(
        key="jobspark_chat",
        data={"history": [list(h) for h in history], "hidden": hidden, "ready": ready,
              "starters": STARTERS if ready else BASIC_STARTERS,
              "followups": S.get("chat_followups", []), "open_request": S.get("_chat_open_request")},
        on_message_change=lambda: None)
    msg = result.message
    if msg and msg.get("n") != S.get("_chat_n"):
        S["_chat_n"] = msg.get("n")
        if msg.get("op") == "clear":
            history.clear()
            S["chat_followups"] = []
        else:
            text = str(msg.get("text", "")).strip()[:MAX_CHARS]
            if text:
                reply, follow = reply_for(text, ready, prof.call)
                history += [("user", text), ("assistant", reply)]
                S["chat_followups"] = follow
        st.rerun(scope="fragment")


def render(hidden: bool = False) -> None:
    """Show the floating assistant. Call once per run, from the app entry point."""
    _widget(hidden)


def request_open() -> None:
    """Open Spark on the next render, for navigation tiles that target the chat assistant."""
    st.session_state["_chat_open_request"] = st.session_state.get("_chat_open_request", 0) + 1
