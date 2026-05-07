from __future__ import annotations

import json
import os
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import streamlit as st

from bwa_backend import app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> bytes | None:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: dict[str, Any]) -> Iterator[tuple[str, Any]]:
    last_state = None
    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
            last_state = step
        if last_state is not None:
            yield ("final", last_state)
        return
    except Exception as e:
        raise e


def extract_latest_state(current_state: dict[str, Any], step_payload: Any) -> dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state


# ─────────────────────────────────────────────────────────────────────────────
# Local image renderer
# ─────────────────────────────────────────────────────────────────────────────

_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    return Path(src.strip().lstrip("./")).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: list[tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        parts.append(("img", f"{(m.group('alt') or '').strip()}|||{(m.group('src') or '').strip()}"))
        last = m.end()
    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue
        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    parts[i + 1] = ("md", "\n".join(nxt.splitlines()[1:]))
        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or alt or None, use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or alt or None, use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1


# ─────────────────────────────────────────────────────────────────────────────
# Past-blog helpers
# ─────────────────────────────────────────────────────────────────────────────

def list_past_blogs() -> list[Path]:
    files = [p for p in Path(".").glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline animation
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_STAGES = [
    ("router",                    "Route",   "1"),
    ("research",                  "Research","2"),
    ("orchestrator",              "Plan",    "3"),
    ("worker",                    "Write",   "4"),
    ("merge_content",             "Merge",   "5"),
    ("decide_images",             "Images",  "6"),
    ("generate_and_place_images", "Finish",  "7"),
]

# Keys each node is responsible for — used to show focused output instead of the full state.
NODE_OUTPUT_KEYS: dict[str, list[str]] = {
    "router":                    ["mode", "needs_research", "queries", "recency_days"],
    "research":                  ["evidence", "research_attempted"],
    "orchestrator":              ["plan"],
    "worker":                    ["sections"],
    "sequential_writer":         ["sections"],
    "merge_content":             ["merged_md"],
    "decide_images":             ["md_with_placeholders", "image_specs"],
    "generate_and_place_images": ["final"],
}


def _infer_node(cur: dict[str, Any], prev: dict[str, Any]) -> str | None:
    if cur.get("final") and not prev.get("final"):
        return "generate_and_place_images"
    if cur.get("md_with_placeholders") and not prev.get("md_with_placeholders"):
        return "decide_images"
    if cur.get("merged_md") and not prev.get("merged_md"):
        return "merge_content"
    # Both parallel workers and sequential_writer produce sections — map both to "worker" stage.
    if len(cur.get("sections") or []) > len(prev.get("sections") or []):
        return "worker"
    if cur.get("plan") and not prev.get("plan"):
        return "orchestrator"
    if cur.get("research_attempted") and not prev.get("research_attempted"):
        return "research"
    if (cur.get("evidence") or []) and not (prev.get("evidence") or []):
        return "research"
    if cur.get("mode") and not prev.get("mode"):
        return "router"
    return None


def _pipeline_html(completed: list[str], active: str | None,
                   worker_info: tuple[int, int] | None = None,
                   error: bool = False) -> str:
    items = ""
    for i, (key, label, num) in enumerate(PIPELINE_STAGES):
        if key == "worker" and worker_info:
            done, total = worker_info
            extra = f"<div class='p-sub'>{done}/{total}</div>"
        else:
            extra = ""

        if key in completed:
            cls, badge = "done", "✓"
        elif key == active and error:
            cls, badge = "error", "✕"
        elif key == active:
            cls = "active"
            badge = f"<span class='dot-spin'>{num}</span>"
        else:
            cls, badge = "pending", num

        fill = "filled" if key in completed else ""
        connector = f'<div class="p-line {fill}"></div>' if i < len(PIPELINE_STAGES) - 1 else ""

        items += f"""
        <div class="p-step {cls}">
            <div class="p-dot">{badge}</div>
            <div class="p-name">{label}{extra}</div>
        </div>{connector}"""

    banner = '<div class="pipe-error-banner">Pipeline stopped — see error below</div>' if error else ""
    return f"""
    <div class="pipeline-shell">
        <div class="pipeline-inner">{items}</div>
        {banner}
    </div>"""


def _stats_html(state: dict[str, Any]) -> str:
    mode = state.get("mode") or ""
    evidence_n = len(state.get("evidence") or [])
    sections_n = len(state.get("sections") or [])
    plan = state.get("plan")
    tasks_n: int | str = "—"
    if isinstance(plan, dict):
        tasks_n = len(plan.get("tasks") or [])
    elif hasattr(plan, "tasks"):
        tasks_n = len(plan.tasks or [])

    mode_color = {
        "open_book": "#f472b6",
        "hybrid":    "#fb923c",
        "closed_book": "#4ade80",
    }.get(mode, "#818cf8")
    mode_label = mode.replace("_", " ").title() or "—"

    return f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-val" style="color:{mode_color};font-size:0.85rem;padding-top:6px">{mode_label}</div>
            <div class="stat-lbl">Mode</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{evidence_n}</div>
            <div class="stat-lbl">Evidence</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{tasks_n}</div>
            <div class="stat-lbl">Planned</div>
        </div>
        <div class="stat-card">
            <div class="stat-val">{sections_n}</div>
            <div class="stat-lbl">Written</div>
        </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color-scheme: light !important;
}

/* ── Page background ── */
.stApp {
    background: linear-gradient(160deg, #f0f4ff 0%, #f8f7ff 50%, #fdf4ff 100%) !important;
}

/* ── Hero ── */
.hero { text-align:center; padding:2.2rem 1rem 1.2rem; }
.hero-title {
    font-size: 2.6rem; font-weight: 800; margin:0 0 .4rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    letter-spacing: -0.04em; line-height:1.1;
}
.hero-sub { color:#64748b; font-size:.95rem; font-weight:400; margin:0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#f5f3ff 0%,#faf5ff 100%) !important;
    border-right: 1px solid rgba(99,102,241,.15) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color:#3730a3 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color:#374151 !important; }

/* ── Primary button ── */
.stButton > button[kind="primary"],
.stButton > button[data-testid*="primary"] {
    width:100%; background:linear-gradient(135deg,#4f46e5,#7c3aed) !important;
    border:none !important; border-radius:10px !important; font-weight:700 !important;
    font-size:.9rem !important; letter-spacing:.02em !important;
    box-shadow:0 4px 18px rgba(79,70,229,.3) !important;
    transition:all .25s cubic-bezier(.4,0,.2,1) !important; color:#fff !important;
}
.stButton > button[kind="primary"]:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 26px rgba(79,70,229,.45) !important;
}
.stButton > button[kind="primary"]:active { transform:translateY(0) !important; }

/* ── Secondary buttons ── */
.stButton > button:not([kind="primary"]) {
    background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    border-radius:8px !important; color:#475569 !important;
    font-weight:500 !important; transition:all .2s ease !important;
}
.stButton > button:not([kind="primary"]):hover {
    background:#f5f3ff !important;
    border-color:rgba(99,102,241,.4) !important; color:#4f46e5 !important;
}

/* ── Download buttons ── */
.stDownloadButton > button {
    border-radius:8px !important;
    background:rgba(16,185,129,.08) !important;
    border:1px solid rgba(16,185,129,.35) !important;
    color:#065f46 !important; font-weight:500 !important;
    transition:all .2s ease !important;
}
.stDownloadButton > button:hover {
    background:rgba(16,185,129,.15) !important;
    border-color:rgba(16,185,129,.55) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#f1f5f9 !important; border-radius:12px !important;
    padding:4px !important; border:1px solid #e2e8f0 !important; gap:2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px !important; font-weight:500 !important; font-size:.88rem !important;
    color:#64748b !important; padding:.4rem .8rem !important; transition:all .2s ease !important;
}
.stTabs [aria-selected="true"] {
    background:#fff !important; color:#4f46e5 !important;
    font-weight:600 !important;
    box-shadow:0 1px 4px rgba(0,0,0,.08) !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    border-radius:8px !important; color:#1e293b !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color:rgba(79,70,229,.5) !important;
    box-shadow:0 0 0 3px rgba(79,70,229,.12) !important;
    outline:none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    border-radius:8px !important; color:#1e293b !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background:#ffffff !important;
    border:1px solid #e9ecf5 !important;
    border-radius:12px !important; padding:1rem 1.2rem !important;
    box-shadow:0 1px 4px rgba(0,0,0,.05) !important;
}
[data-testid="stMetricValue"] { font-weight:700 !important; color:#4f46e5 !important; }
[data-testid="stMetricLabel"] { color:#64748b !important; font-size:.8rem !important; }

/* ── DataFrame ── */
[data-testid="stDataFrame"] {
    border-radius:12px !important; overflow:hidden !important;
    border:1px solid #e2e8f0 !important;
    box-shadow:0 1px 4px rgba(0,0,0,.05) !important;
}

/* ── Status ── */
[data-testid="stStatus"] {
    border-radius:12px !important; background:#ffffff !important;
    border:1px solid #e2e8f0 !important;
    box-shadow:0 1px 4px rgba(0,0,0,.05) !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border:1px solid #e2e8f0 !important;
    border-radius:12px !important; background:#ffffff !important;
    box-shadow:0 1px 4px rgba(0,0,0,.04) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius:10px !important; }

/* ── HR ── */
hr { border:none !important; border-top:1px solid #e9ecf5 !important; margin:1.2rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:#f1f5f9; }
::-webkit-scrollbar-thumb { background:rgba(99,102,241,.25); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(99,102,241,.45); }

/* ─────────────────────────────────────────────────────────── */
/*  PIPELINE ANIMATION                                         */
/* ─────────────────────────────────────────────────────────── */
.pipeline-shell {
    padding:1.2rem 1rem;
    background:#ffffff;
    border:1px solid #e2e8f0;
    border-radius:16px; margin-bottom:.8rem;
    overflow-x:auto;
    box-shadow:0 2px 8px rgba(79,70,229,.07);
}
.pipeline-inner {
    display:flex; align-items:center; justify-content:center;
    flex-wrap:nowrap; min-width:500px; gap:0;
}
.p-step {
    display:flex; flex-direction:column; align-items:center;
    gap:5px; min-width:72px; flex-shrink:0;
}
.p-dot {
    width:38px; height:38px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:14px; font-weight:700;
    transition:all .4s cubic-bezier(.4,0,.2,1);
}
.p-name {
    font-size:10px; font-weight:500; text-align:center;
    font-family:'Inter',sans-serif; letter-spacing:.03em;
    white-space:nowrap; transition:color .3s;
}
.p-sub { font-size:8px; color:#d97706; margin-top:2px; }
.p-line {
    flex:1; height:2px; background:#e9ecf5;
    margin:0 3px; margin-bottom:22px; border-radius:2px;
    transition:background .5s ease; min-width:8px;
}
.p-line.filled { background:linear-gradient(90deg,#4f46e5,#7c3aed); }

/* Pending */
.p-step.pending .p-dot {
    background:#f1f5f9; color:#94a3b8;
    border:1.5px solid #e2e8f0;
}
.p-step.pending .p-name { color:#94a3b8; }

/* Done */
.p-step.done .p-dot {
    background:linear-gradient(135deg,#059669,#10b981);
    color:#fff; border:none;
    box-shadow:0 0 12px rgba(16,185,129,.35);
}
.p-step.done .p-name { color:#059669; font-weight:600; }

/* Active */
.p-step.active .p-dot {
    background:linear-gradient(135deg,#d97706,#f59e0b);
    color:#fff; border:none;
    animation:dot-pulse 1.2s ease-in-out infinite;
}
.p-step.active .p-name { color:#d97706; font-weight:700; }
@keyframes dot-pulse {
    0%,100% { box-shadow:0 0 0 0 rgba(245,158,11,.6); transform:scale(1); }
    50%      { box-shadow:0 0 0 9px rgba(245,158,11,0); transform:scale(1.13); }
}

/* Spinning number inside active badge */
.dot-spin { display:inline-block; animation:num-spin 1.4s linear infinite; }
@keyframes num-spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }

/* Error step */
.p-step.error .p-dot {
    background:linear-gradient(135deg,#dc2626,#ef4444);
    color:#fff; border:none;
    box-shadow:0 0 14px rgba(239,68,68,.4);
}
.p-step.error .p-name { color:#ef4444; font-weight:700; }

/* Error banner */
.pipe-error-banner {
    margin-top:10px; padding:8px 14px; border-radius:8px;
    background:rgba(239,68,68,.08); border:1px solid rgba(239,68,68,.3);
    color:#b91c1c; font-size:.82rem; font-weight:600; text-align:center;
}

/* ── Live-stats row ── */
.stat-row {
    display:flex; gap:10px; margin:.6rem 0 1rem; flex-wrap:wrap;
}
.stat-card {
    flex:1; min-width:80px;
    background:#ffffff;
    border:1px solid #e9ecf5;
    border-radius:10px; padding:10px 12px; text-align:center;
    box-shadow:0 1px 4px rgba(0,0,0,.05);
    animation:fadeUp .35s ease forwards;
}
.stat-val {
    font-size:1.5rem; font-weight:800; color:#4f46e5;
    line-height:1; margin-bottom:3px;
    font-family:'Inter',sans-serif;
}
.stat-lbl {
    font-size:.68rem; color:#94a3b8; font-weight:600;
    letter-spacing:.06em; text-transform:uppercase;
}

/* ── Fade-up ── */
@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
.fade-up { animation:fadeUp .45s ease forwards; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# App entry point
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BlogForge AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# Hero header
st.markdown("""
<div class="hero">
    <div class="hero-title">BlogForge AI</div>
    <p class="hero-sub">Intelligent long-form blog generation · Powered by LangGraph + Gemini</p>
</div>
""", unsafe_allow_html=True)

# Always restore from session state so a newly saved key takes effect immediately.
for _env_key in ("GOOGLE_API_KEY", "TAVILY_API_KEY", "GEMINI_MODEL"):
    _stored = st.session_state.get(f"_env_{_env_key}", "")
    if _stored:
        os.environ[_env_key] = _stored

# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    google_key_missing = not os.getenv("GOOGLE_API_KEY")

    with st.expander("🔑 API Keys", expanded=google_key_missing):
        google_key_input = st.text_input(
            "Google API Key",
            value=st.session_state.get("_env_GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", "")),
            type="password", placeholder="AIza...",
            help="Required. Get one free at https://aistudio.google.com/apikey",
        )
        tavily_key_input = st.text_input(
            "Tavily API Key (optional)",
            value=st.session_state.get("_env_TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", "")),
            type="password", placeholder="tvly-...",
            help="Enables live web research.",
        )
        AVAILABLE_MODELS = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ]
        current_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        model_idx = AVAILABLE_MODELS.index(current_model) if current_model in AVAILABLE_MODELS else 0
        model_choice = st.selectbox(
            "Gemini Model", AVAILABLE_MODELS, index=model_idx,
            help="gemini-2.5-flash recommended. Switch to flash-lite on quota errors.",
        )
        st.caption("Keys are applied automatically when you click Generate Blog. Use Save to persist them across page reloads.")
        if st.button("💾 Save Keys"):
            if google_key_input.strip():
                os.environ["GOOGLE_API_KEY"] = google_key_input.strip()
                st.session_state["_env_GOOGLE_API_KEY"] = google_key_input.strip()
                tail = google_key_input.strip()[-4:]
                st.success(f"Keys saved! Active key ends in …{tail}")
            else:
                st.warning("Google API Key is required.")
            if tavily_key_input.strip():
                os.environ["TAVILY_API_KEY"] = tavily_key_input.strip()
                st.session_state["_env_TAVILY_API_KEY"] = tavily_key_input.strip()
            else:
                os.environ.pop("TAVILY_API_KEY", None)
                st.session_state.pop("_env_TAVILY_API_KEY", None)
            os.environ["GEMINI_MODEL"] = model_choice
            st.session_state["_env_GEMINI_MODEL"] = model_choice

        # Show which key is currently active
        active_key = os.getenv("GOOGLE_API_KEY", "")
        if active_key:
            st.caption(f"Active key: …{active_key[-4:]}")

    if google_key_missing:
        st.warning("Add your Google API Key above to get started.")

    st.markdown("---")
    st.markdown("### ✍️ New Blog")
    topic = st.text_area(
        "Topic", height=110,
        placeholder="e.g. How Transformer attention works, or\nWeekly AI news roundup May 2026",
    )
    as_of = st.date_input("As-of date", value=date.today())
    generate_images = st.toggle(
        "Generate images",
        value=False,
        help="Requires a paid Google Cloud billing account. Image generation is NOT available on the free Gemini API tier.",
    )
    if generate_images:
        st.caption("⚠️ Paid feature — ensure billing is enabled on your Google Cloud project.")

    parallel_workers = st.toggle(
        "Parallel section writing",
        value=False,
        help=(
            "OFF (default): sections written one-by-one with a short delay — safe for free-tier Gemini API (10 RPM). "
            "ON: all sections written simultaneously — faster but exhausts free quota quickly. Only enable with a paid API key."
        ),
    )
    if parallel_workers:
        st.caption("⚠️ Parallel mode — ensure you have a paid API key or sufficient quota.")
    else:
        st.caption("Sequential mode — free-tier safe (sections written one at a time).")

    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("### 📂 Past Blogs")
    past_files = list_past_blogs()

    if "confirm_delete" not in st.session_state:
        st.session_state["confirm_delete"] = None

    if not past_files:
        st.caption("No saved blogs yet.")
        selected_md_file = None
    else:
        options: list[str] = []
        file_by_label: dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                md_text = read_md_file(p)
                title = extract_title_from_md(md_text, p.stem)
            except Exception:
                title = p.stem
            label = f"{title}  ·  {p.name}"
            options.append(label)
            file_by_label[label] = p

        selected_label = st.radio(
            "Select a blog", options=options, index=0, label_visibility="collapsed",
        )
        selected_md_file = file_by_label.get(selected_label)

        col_load, col_del = st.columns(2)
        with col_load:
            if st.button("📂 Load", use_container_width=True):
                if selected_md_file:
                    md_text = read_md_file(selected_md_file)
                    st.session_state["last_out"] = {
                        "plan": None, "evidence": [], "image_specs": [], "final": md_text,
                    }
                    st.session_state["topic_prefill"] = extract_title_from_md(
                        md_text, selected_md_file.stem
                    )
                    st.session_state["confirm_delete"] = None

        with col_del:
            if st.button("🗑️ Delete", use_container_width=True):
                st.session_state["confirm_delete"] = str(selected_md_file)

        # Confirmation dialog
        if st.session_state["confirm_delete"] == str(selected_md_file):
            st.warning(f"Delete **{selected_md_file.name}**?")
            yes_col, no_col = st.columns(2)
            with yes_col:
                if st.button("Yes, delete", type="primary", use_container_width=True):
                    try:
                        selected_md_file.unlink()
                        st.session_state["confirm_delete"] = None
                        if st.session_state.get("last_out") and st.session_state["last_out"].get("_source") == str(selected_md_file):
                            st.session_state["last_out"] = None
                        st.success("Deleted.")
                        st.rerun()
                    except Exception as _de:
                        st.error(f"Could not delete: {_de}")
            with no_col:
                if st.button("Cancel", use_container_width=True):
                    st.session_state["confirm_delete"] = None

# ── Session defaults ───────────────────────────────────────────────────────

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None

# ── Tabs ───────────────────────────────────────────────────────────────────

tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🧩 Plan", "🔎 Evidence", "📝 Preview", "🖼️ Images", "🧾 Logs"]
)

logs: list[str] = []


def log(msg: str):
    logs.append(msg)


# ── Generation handler ─────────────────────────────────────────────────────

if run_btn:
    # Auto-apply whatever is currently in the key inputs — no need to click Save Keys first.
    if google_key_input.strip():
        os.environ["GOOGLE_API_KEY"] = google_key_input.strip()
        st.session_state["_env_GOOGLE_API_KEY"] = google_key_input.strip()
    if tavily_key_input.strip():
        os.environ["TAVILY_API_KEY"] = tavily_key_input.strip()
        st.session_state["_env_TAVILY_API_KEY"] = tavily_key_input.strip()
    os.environ["GEMINI_MODEL"] = model_choice
    st.session_state["_env_GEMINI_MODEL"] = model_choice

    if not topic.strip():
        st.warning("Please enter a topic.")
        st.stop()
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API Key is not set. Enter it in the sidebar.")
        st.stop()

    inputs: dict[str, Any] = {
        "topic": topic.strip(), "mode": "", "needs_research": False,
        "queries": [], "evidence": [], "plan": None, "as_of": as_of.isoformat(),
        "recency_days": 7, "sections": [], "generate_images": generate_images,
        "parallel_workers": parallel_workers,
        "merged_md": "", "md_with_placeholders": "", "image_specs": [], "final": "",
        "research_attempted": False,
    }
    st.session_state.pop("last_pipeline_html", None)
    st.session_state["node_outputs"] = {}

    pipeline_ph = st.empty()
    stats_ph    = st.empty()
    status      = st.status("Initializing pipeline…", expanded=True)
    status_msg_ph = status.empty()

    pipeline_ph.markdown(_pipeline_html([], None), unsafe_allow_html=True)

    completed_nodes: list[str] = []
    active_node: str | None    = None
    prev_state: dict[str, Any] = {}
    current_state: dict[str, Any] = {}
    st.session_state["node_outputs"] = {}

    try:
        for kind, payload in try_stream(app, inputs):
            if kind in ("updates", "values"):
                current_state = extract_latest_state(dict(current_state), payload)

                node = _infer_node(current_state, prev_state)
                if node and node != active_node:
                    if active_node and active_node not in completed_nodes:
                        completed_nodes.append(active_node)
                        _keys = NODE_OUTPUT_KEYS.get(active_node, list(prev_state.keys()))
                        st.session_state["node_outputs"][active_node] = {
                            k: prev_state[k] for k in _keys if k in prev_state
                        }
                    active_node = node
                    label = node.replace("_", " ").title()
                    status_msg_ph.markdown(f"**{label}** running…")

                sections = current_state.get("sections") or []
                plan = current_state.get("plan")
                total_tasks = 0
                if isinstance(plan, dict):
                    total_tasks = len(plan.get("tasks") or [])
                elif hasattr(plan, "tasks"):
                    total_tasks = len(plan.tasks or [])
                wp = (len(sections), total_tasks) if active_node == "worker" and total_tasks else None

                pipeline_ph.markdown(
                    _pipeline_html(completed_nodes, active_node, wp),
                    unsafe_allow_html=True,
                )
                stats_ph.markdown(_stats_html(current_state), unsafe_allow_html=True)

                log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")
                prev_state = dict(current_state)

            elif kind == "final":
                if active_node:
                    _keys = NODE_OUTPUT_KEYS.get(active_node, list(payload.keys()))
                    st.session_state["node_outputs"][active_node] = {
                        k: payload[k] for k in _keys if k in payload
                    }
                all_keys = [s[0] for s in PIPELINE_STAGES]
                _final_pipe = _pipeline_html(all_keys, None)
                pipeline_ph.markdown(_final_pipe, unsafe_allow_html=True)
                st.session_state["last_pipeline_html"] = _final_pipe
                stats_ph.empty()
                status_msg_ph.empty()
                st.session_state["last_out"] = payload
                status.update(label="✅ Blog generated successfully!", state="complete", expanded=False)
                log("[final] received final state")

    except Exception as _err:
        pipeline_ph.markdown(
            _pipeline_html(completed_nodes, active_node, error=True),
            unsafe_allow_html=True,
        )
        stats_ph.empty()
        status.update(label="Pipeline failed", state="error", expanded=True)
        cause = _err.__cause__ or _err
        st.error(f"**Graph error** `{type(_err).__name__}`: {_err}")
        if cause is not _err:
            st.error(f"**Underlying cause** `{type(cause).__name__}`: {cause}")
        st.stop()


# ── Results ────────────────────────────────────────────────────────────────

out = st.session_state.get("last_out")

if out:
    # ── Persistent pipeline + node inspector ──────────────────────────────
    # Only render persisted pipeline on reruns — during an active run the live
    # pipeline_ph placeholder is already visible above, so skip to avoid duplication.
    _last_pipe = st.session_state.get("last_pipeline_html")
    if _last_pipe and not run_btn:
        st.markdown(_last_pipe, unsafe_allow_html=True)
        _node_outputs = st.session_state.get("node_outputs", {})
        if _node_outputs:
            _slabels = {k: lbl for k, lbl, _ in PIPELINE_STAGES}
            _nkeys = list(_node_outputs.keys())
            _nlabels = [_slabels.get(k, k.replace("_", " ").title()) for k in _nkeys]
            _nsel = st.selectbox(
                "Select a node to inspect its output:",
                range(len(_nkeys)),
                format_func=lambda i: _nlabels[i],
                key="node_inspect_main",
            )
            if _nsel is not None:
                _snap = json.loads(json.dumps(_node_outputs[_nkeys[_nsel]], default=str))
                with st.expander(f"Output: {_nlabels[_nsel]}", expanded=True):
                    st.json(_snap, expanded=2)
        st.markdown("---")

    # ── Plan tab ──
    with tab_plan:
        plan_obj = out.get("plan")
        if not plan_obj:
            st.info("No plan found in output.")
        else:
            if hasattr(plan_obj, "model_dump"):
                plan_dict = plan_obj.model_dump()
            elif isinstance(plan_obj, dict):
                plan_dict = plan_obj
            else:
                plan_dict = json.loads(json.dumps(plan_obj, default=str))

            st.markdown(f"### {plan_dict.get('blog_title', 'Blog Plan')}")
            c1, c2, c3 = st.columns(3)
            for _col, _lbl, _key in [
                (c1, "Audience", "audience"),
                (c2, "Tone", "tone"),
                (c3, "Kind", "blog_kind"),
            ]:
                _val = plan_dict.get(_key) or "—"
                _col.markdown(
                    f"""<div style="background:#fff;border:1px solid #e9ecf5;border-radius:12px;"""
                    f"""padding:.9rem 1.1rem;box-shadow:0 1px 4px rgba(0,0,0,.05);min-height:72px">"""
                    f"""<div style="font-size:.65rem;color:#94a3b8;font-weight:600;letter-spacing:.06em;"""
                    f"""text-transform:uppercase;margin-bottom:5px">{_lbl}</div>"""
                    f"""<div style="font-weight:700;color:#4f46e5;font-size:.9rem;"""
                    f"""word-break:break-word;line-height:1.4">{_val}</div></div>""",
                    unsafe_allow_html=True,
                )

            tasks = plan_dict.get("tasks", [])
            if tasks:
                st.markdown("#### Sections")
                df = pd.DataFrame([
                    {
                        "#": t.get("id"),
                        "Title": t.get("title"),
                        "Words": t.get("target_words"),
                        "Research": "✓" if t.get("requires_research") else "",
                        "Citations": "✓" if t.get("requires_citations") else "",
                        "Code": "✓" if t.get("requires_code") else "",
                        "Tags": ", ".join(t.get("tags") or []),
                    }
                    for t in tasks
                ]).sort_values("#")
                st.dataframe(df, use_container_width=True, hide_index=True)

                with st.expander("Raw task JSON"):
                    st.json(tasks)

    # ── Evidence tab ──
    with tab_evidence:
        evidence = out.get("evidence") or []
        if not evidence:
            _mode = (out.get("mode") or "").replace("_", " ").title()
            if out.get("research_attempted"):
                st.warning(
                    f"**Research node ran ({_mode} mode) but returned no evidence.** "
                    "Likely causes: Tavily API key is invalid or expired, "
                    "or all search results were filtered out by the recency window. "
                    "Re-enter your Tavily key in the sidebar and try again."
                )
            elif (out.get("mode") or "") in ("open_book", "hybrid"):
                st.warning(
                    "**Research was needed but no Tavily key was found.** "
                    "Add your Tavily API Key in the sidebar to enable web research."
                )
            else:
                st.info("Closed-book mode — no web research needed for this topic.")
        else:
            st.markdown(f"**{len(evidence)} sources collected** — click any item to see full details")
            for idx, e in enumerate(evidence):
                if hasattr(e, "model_dump"):
                    e_dict = e.model_dump()
                elif isinstance(e, dict):
                    e_dict = e
                else:
                    e_dict = json.loads(json.dumps(e, default=str))

                title = e_dict.get("title") or f"Source {idx + 1}"
                source = e_dict.get("source") or ""
                date_str = e_dict.get("published_at") or ""
                label = f"**{idx + 1}.** {title}"
                if source:
                    label += f"  ·  {source}"
                if date_str:
                    label += f"  ·  {date_str}"

                with st.expander(label):
                    url = e_dict.get("url", "")
                    if url:
                        st.markdown(f"[{url}]({url})")
                    st.json(e_dict)

    # ── Preview tab ──
    with tab_preview:
        final_md = out.get("final") or ""
        if not final_md:
            st.warning("No markdown output yet.")
        else:
            render_markdown_with_local_images(final_md)
            st.markdown("---")

            plan_obj = out.get("plan")
            if hasattr(plan_obj, "blog_title"):
                blog_title = plan_obj.blog_title
            elif isinstance(plan_obj, dict):
                blog_title = plan_obj.get("blog_title", "blog")
            else:
                blog_title = extract_title_from_md(final_md, "blog")

            md_filename = f"{safe_slug(blog_title)}.md"
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download Markdown", data=final_md.encode("utf-8"),
                    file_name=md_filename, mime="text/markdown", use_container_width=True,
                )
            with col2:
                bundle = bundle_zip(final_md, md_filename, Path("images"))
                st.download_button(
                    "📦 Download Bundle (MD + images)", data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip", use_container_width=True,
                )

    # ── Images tab ──
    with tab_images:
        specs = out.get("image_specs") or []
        images_dir = Path("images")

        if not specs and not images_dir.exists():
            st.info("No images generated for this blog.")
        else:
            if specs:
                st.markdown(f"**{len(specs)} image(s) planned**")
                with st.expander("Image spec details"):
                    st.json(specs)

            if images_dir.exists():
                files = [p for p in images_dir.iterdir() if p.is_file()]
                if files:
                    cols = st.columns(min(len(files), 2))
                    for idx, p in enumerate(sorted(files)):
                        cols[idx % 2].image(str(p), caption=p.name, use_container_width=True)
                    z = images_zip(images_dir)
                    if z:
                        st.download_button(
                            "⬇️ Download Images (zip)", data=z,
                            file_name="images.zip", mime="application/zip",
                        )
                else:
                    st.warning("images/ directory is empty.")

    # ── Logs tab ──
    with tab_logs:
        if "logs" not in st.session_state:
            st.session_state["logs"] = []
        if logs:
            st.session_state["logs"].extend(logs)
        st.text_area(
            "Event log", value="\n\n".join(st.session_state["logs"][-80:]),
            height=400, label_visibility="collapsed",
        )

else:
    with tab_preview:
        st.markdown("""
<div style="text-align:center;padding:4rem 2rem;">
    <div style="font-size:3rem;margin-bottom:1rem">✍️</div>
    <div style="font-size:1.1rem;color:#94a3b8;font-weight:500">
        Enter a topic in the sidebar and click <strong style="color:#4f46e5">Generate Blog</strong> to get started.
    </div>
</div>""", unsafe_allow_html=True)
