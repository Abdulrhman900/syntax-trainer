import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import os
from datetime import datetime

# ======================================
# 0) Try to import OpenAI (AI Coach)
# ======================================
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ======================================
# 1) Page config
# ======================================
st.set_page_config(
    page_title="Syntax Trainer Pro",
    page_icon="⚡",
    layout="wide",
)

# ======================================
# 2) Global CSS (White + Mint theme)
# ======================================
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top, #ecfeff, #f9fafb);
        color: #0f172a;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .stButton>button {
        border-radius: 999px;
        padding: 0.45rem 1.3rem;
        font-weight: 600;
        border: 1px solid rgba(148,163,184,0.6);
        background: linear-gradient(135deg, #34d399, #22c55e);
        color: white;
        box-shadow: 0 0 14px rgba(52, 211, 153, 0.45);
    }
    .stButton>button:hover {
        border-color: rgba(15,23,42,0.9);
        box-shadow: 0 0 24px rgba(16, 185, 129, 0.8);
        transform: translateY(-1px);
    }
    .card {
        border-radius: 22px;
        padding: 1.2rem 1.4rem;
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 18px 40px rgba(15,23,42,0.08);
        backdrop-filter: blur(12px);
    }
    .subtitle {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        color: rgba(148,163,184,0.95);
        margin-bottom: 0.1rem;
    }
    .title-main {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(90deg, #0f172a, #22c55e);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.4rem;
    }
    .small-label {
        font-size: 12px;
        color: rgba(100,116,139,0.98);
    }
    .pill {
        display: inline-flex;
        align-items: center;
        padding: 0.16rem 0.7rem;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(45,212,191,0.16);
        color: #0f766e;
        border: 1px solid rgba(34,197,94,0.55);
        margin-right: 6px;
    }
    .pill-level {
        background: rgba(251,191,36,0.18);
        color: #92400e;
        border-color: rgba(251,191,36,0.6);
    }
    .pill-mode {
        background: rgba(129,140,248,0.15);
        color: #312e81;
        border-color: rgba(129,140,248,0.6);
    }
    .footer-3d {
        font-family: "Space Grotesk", system-ui;
        font-weight: 700;
        font-size: 12px;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #047857;
        text-shadow:
            0 0 4px rgba(45,212,191,.8),
            0 0 14px rgba(16,185,129,.7);
    }
    .home-card {
        border-radius: 20px;
        padding: 1.1rem 1.2rem;
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(209,213,219,0.8);
        box-shadow: 0 14px 32px rgba(15,23,42,0.06);
        transition: all 0.16s ease-out;
        cursor: pointer;
    }
    .home-card:hover {
        box-shadow: 0 18px 40px rgba(16,185,129,0.2);
        transform: translateY(-2px);
        border-color: rgba(16,185,129,0.7);
    }
    .home-icon {
        font-size: 28px;
        margin-bottom: 0.1rem;
    }
    .home-title {
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 0.1rem;
        color: #0f172a;
    }
    .home-desc {
        font-size: 12px;
        color: #6b7280;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================
# 3) Lottie helper (no extra package)
# ======================================
def render_lottie(url: str, height: int = 180):
    """Embed a Lottie animation by URL using HTML."""
    lottie_html = f"""
    <lottie-player src="{url}" background="transparent" speed="1"
        style="width:100%;height:{height}px;" loop autoplay>
    </lottie-player>
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    """
    components.html(lottie_html, height=height + 10, scrolling=False)

# ======================================
# 4) Google Sheets (read-only via CSV)
# ======================================
SHEET_ID = "1Gly5KDsBf7jjB-x5fwTK3JLOLgggrtWpPM8bYbLqmRk"
BASE_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet="

@st.cache_data(ttl=300, show_spinner=False)
def load_sheet(sheet_name: str) -> pd.DataFrame:
    url = BASE_URL + sheet_name
    df = pd.read_csv(url)
    for c in df.columns:
        df[c] = df[c].astype(str)
    return df

syntax_df = pd.DataFrame()
questions_df = pd.DataFrame()
docs_df = pd.DataFrame()
sheets_ok = True

try:
    syntax_df = load_sheet("Syntax_Practice")
except Exception as e:
    st.error(f"Google Sheet Error in Syntax_Practice: {e}")
    sheets_ok = False

try:
    questions_df = load_sheet("Questions_Practice")
except Exception as e:
    st.error(f"Google Sheet Error in Questions_Practice: {e}")
    sheets_ok = False

try:
    docs_df = load_sheet("Documentation")
except Exception as e:
    st.error(f"Google Sheet Error in Documentation: {e}")
    sheets_ok = False

# ======================================
# 5) OpenAI (AI Coach)
# ======================================
OPENAI_API_KEY = None
client = None
ai_enabled = False

if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
    OPENAI_API_KEY = st.secrets["openai"]["api_key"]
elif "OPENAI_API_KEY" in os.environ:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

if OPENAI_API_KEY and OpenAI is not None:
    client = OpenAI(api_key=OPENAI_API_KEY)
    ai_enabled = True

def ask_ai(prompt: str) -> str:
    if not ai_enabled:
        return "❌ AI Coach is disabled (missing OPENAI_API_KEY in secrets)."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You help students practice SQL, Pandas, NumPy, and scikit-learn. "
                        "Explain briefly but clearly, in English, but you may accept Arabic text in the question. "
                        "Focus on syntax, logic, and missing/extra parts."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI Error: {e}"

# ======================================
# 6) Helpers
# ======================================
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().strip(";")
    s = " ".join(s.split())
    return s.lower()

def pick_random_row(df: pd.DataFrame, lang: str = None, level: str = None):
    if df.empty:
        return None
    tmp = df.copy()
    if lang:
        tmp = tmp[tmp["language"].str.lower() == lang.lower()]
    if level:
        tmp = tmp[tmp["level"].str.lower() == level.lower()]
    if tmp.empty:
        return None
    return tmp.sample(1).iloc[0]

def generate_dataset(dataset_name: str) -> pd.DataFrame | None:
    dataset_name = (dataset_name or "").strip().lower()

    if dataset_name == "employees":
        data = {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["Ali", "Maha", "Sara", "Omar", "Noura", "Khalid"],
            "city": ["Riyadh", "Jeddah", "Riyadh", "Dammam", "Jeddah", "Makkah"],
            "dept": ["IT", "HR", "Sales", "Finance", "IT", "Sales"],
            "salary": [3500, 4800, 5200, 6100, 7000, 2900],
        }
        return pd.DataFrame(data)

    if dataset_name == "products":
        data = {
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["Keyboard", "Mouse", "Monitor", "Laptop", "Headset", "USB"],
            "category": ["Accessories", "Accessories", "Display", "Laptop", "Audio", "Storage"],
            "price": [40, 25, 120, 2500, 80, 15],
            "stock": [50, 120, 15, 7, 35, 200],
        }
        return pd.DataFrame(data)

    if dataset_name == "customers":
        data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["Ahmad", "Laila", "Salem", "Reem", "Yousef"],
            "city": ["Riyadh", "Jeddah", "Riyadh", "Khobar", "Jeddah"],
            "balance": [1200, 3400, 800, 5600, 150],
        }
        return pd.DataFrame(data)

    if dataset_name == "orders":
        data = {
            "id": [1, 2, 3, 4, 5, 6],
            "customer_id": [1, 2, 2, 3, 5, 1],
            "total": [250, 600, 150, 900, 40, 130],
            "status": ["new", "pending", "shipped", "new", "cancelled", "pending"],
        }
        return pd.DataFrame(data)

    if dataset_name == "students":
        data = {
            "id": [1, 2, 3, 4],
            "name": ["Fahd", "Noura", "Lama", "Sultan"],
            "age": [18, 20, 19, 22],
            "grade": [85, 95, 78, 88],
        }
        return pd.DataFrame(data)

    return None

# ======================================
# 7) Session state (user, scores, section)
# ======================================
if "username" not in st.session_state:
    st.session_state["username"] = "Guest"

if "section" not in st.session_state:
    st.session_state["section"] = "home"  # home, practice, qlab, docs, leaderboard, settings

if "syntax_score" not in st.session_state:
    st.session_state["syntax_score"] = 0
if "syntax_attempts" not in st.session_state:
    st.session_state["syntax_attempts"] = 0

if "q_score" not in st.session_state:
    st.session_state["q_score"] = 0
if "q_attempts" not in st.session_state:
    st.session_state["q_attempts"] = 0

if "history" not in st.session_state:
    st.session_state["history"] = []  # for leaderboard + dashboard

def log_event(mode: str, language: str, level: str, correct: bool, delta: int):
    st.session_state["history"].append(
        {
            "time": datetime.utcnow().isoformat(timespec="seconds"),
            "user": st.session_state["username"],
            "mode": mode,
            "language": language,
            "level": level,
            "correct": bool(correct),
            "delta": int(delta),
        }
    )

# ======================================
# 8) Sidebar (user + language + stats)
# ======================================
with st.sidebar:
    st.markdown("### ⚡ Syntax Trainer Pro")
    st.markdown(
        "<p class='small-label'>Interactive syntax lab powered by Google Sheets & AI.</p>",
        unsafe_allow_html=True,
    )

    st.session_state["username"] = st.text_input(
        "Username (for leaderboard):",
        value=st.session_state["username"],
        key="username_input",
    )

    # Languages from sheets
    langs = []
    if not syntax_df.empty:
        langs.extend(list(syntax_df["language"].unique()))
    if not questions_df.empty:
        langs.extend(list(questions_df["language"].unique()))
    if not docs_df.empty:
        langs.extend(list(docs_df["language"].unique()))
    if not langs:
        langs = ["SQL", "Pandas", "NumPy", "sklearn"]
    langs = sorted(set(langs), key=lambda x: x.lower())

    preferred_lang = st.selectbox(
        "Preferred language:",
        options=langs,
        index=0,
        key="preferred_lang_select",
    )

    st.markdown("---")

    st.markdown("**Session stats**")
    st.write(f"- Syntax: {st.session_state['syntax_score']} / {st.session_state['syntax_attempts']}")
    st.write(f"- Q-Lab: {st.session_state['q_score']} / {st.session_state['q_attempts']}")

    if st.button("Reset session stats", key="reset_stats_btn"):
        st.session_state["syntax_score"] = 0
        st.session_state["syntax_attempts"] = 0
        st.session_state["q_score"] = 0
        st.session_state["q_attempts"] = 0
        st.session_state["history"] = []
        st.experimental_rerun()

    st.markdown("---")
    if ai_enabled:
        st.success("AI Coach: Enabled")
    else:
        st.warning("AI Coach: Disabled")

# ======================================
# 9) Header
# ======================================
st.markdown(
    """
    <div class="card">
        <div class="subtitle">Training Environment</div>
        <div class="title-main">Syntax Trainer Pro</div>
        <p class="small-label" style="margin-top:0.1rem; max-width:720px;">
            A practice-first environment for SQL, Pandas, NumPy, and scikit-learn. 
            You type the syntax, the app checks and AI Coach explains. 
            Questions can be in Arabic, UI is fully English.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

# ======================================
# 10) Home section (cards + lotties)
# ======================================
def render_home():
    st.markdown("#### Home · Choose your mode")

    c1, c2, c3 = st.columns(3)
    with c1:
        render_lottie(
            "https://assets5.lottiefiles.com/packages/lf20_w51pcehl.json",
            height=180,
        )
    with c2:
        render_lottie(
            "https://assets2.lottiefiles.com/packages/lf20_gigyrcoy.json",
            height=180,
        )
    with c3:
        render_lottie(
            "https://assets7.lottiefiles.com/packages/lf20_4kx2q32n.json",
            height=180,
        )

    st.write("")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-icon">⭐</div>
                <div class="home-title">Start Practice</div>
                <div class="home-desc">
                    Practice raw syntax line by line. Focus on muscle memory and accuracy.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Practice", key="go_practice"):
            st.session_state["section"] = "practice"
            st.experimental_rerun()

    with col_b:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-icon">🎯</div>
                <div class="home-title">Q-Lab</div>
                <div class="home-desc">
                    Solve real questions with small datasets, then compare with model answers.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Q-Lab", key="go_qlab"):
            st.session_state["section"] = "qlab"
            st.experimental_rerun()

    with col_c:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-icon">📘</div>
                <div class="home-title">Docs Hub</div>
                <div class="home-desc">
                    Browse hand-picked documentation, syntax patterns, and usage examples.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Go to Docs", key="go_docs"):
            st.session_state["section"] = "docs"
            st.experimental_rerun()

    st.write("")
    col_d, col_e = st.columns(2)
    with col_d:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-icon">🏆</div>
                <div class="home-title">Leaderboard</div>
                <div class="home-desc">
                    See your session performance and compare across modes and levels.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View Leaderboard", key="go_leaderboard"):
            st.session_state["section"] = "leaderboard"
            st.experimental_rerun()

    with col_e:
        st.markdown(
            """
            <div class="home-card">
                <div class="home-icon">⚙️</div>
                <div class="home-title">Settings</div>
                <div class="home-desc">
                    Reset progress, review AI status, and manage your preferred language.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Settings", key="go_settings"):
            st.session_state["section"] = "settings"
            st.experimental_rerun()

# ======================================
# 11) Practice section (Syntax_Practice)
# ======================================
def render_practice():
    st.markdown("#### ⭐ Syntax Practice")

    if syntax_df.empty:
        st.warning("Syntax_Practice sheet is empty or not available.")
        return

    col1, col2, col3 = st.columns([1.2, 1.2, 0.8])

    with col1:
        level_options = sorted(syntax_df["level"].unique())
        level = st.selectbox(
            "Level",
            options=level_options,
            index=0,
            key="practice_level_select",
        )
    with col2:
        categories = sorted(syntax_df["category"].unique())
        category = st.selectbox(
            "Category",
            options=["All"] + categories,
            index=0,
            key="practice_category_select",
        )
    with col3:
        st.write("")

    filtered = syntax_df.copy()
    filtered = filtered[filtered["language"].str.lower() == preferred_lang.lower()]
    filtered = filtered[filtered["level"] == level]
    if category != "All":
        filtered = filtered[filtered["category"] == category]

    row = pick_random_row(filtered)
    if row is None:
        st.warning("No matching syntax items for current filters.")
        return

    st.markdown(
        f"""
        <div class="card">
            <div>
                <span class="pill">{row['language']}</span>
                <span class="pill pill-level">{row['level']}</span>
                <span class="pill pill-mode">{row['category']}</span>
            </div>
            <h4 style="margin-top:0.45rem;margin-bottom:0.25rem;">Task</h4>
            <p style="font-size:14px; color:#111827; margin-bottom:0.18rem;">
                {row.get('description_en','').strip()}
            </p>
            <p style="font-size:13px; color:#6b7280; direction:rtl; text-align:right;">
                {row.get('description_ar','').strip()}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    user_answer = st.text_area(
        "Type the exact syntax:",
        height=90,
        key="practice_answer_textarea",
    )

    colb1, colb2, colb3 = st.columns([0.45, 0.3, 0.25])
    with colb1:
        check_btn = st.button("Check", key="practice_check_btn")
    with colb2:
        new_btn = st.button("New item", key="practice_new_btn")
    with colb3:
        show_btn = st.button("Show model answer", key="practice_show_btn")

    expected = row.get("syntax", "")

    if check_btn:
        st.session_state["syntax_attempts"] += 1

        if normalize_text(user_answer) == normalize_text(expected):
            st.success("Correct syntax. Nice work! ✅")
            st.session_state["syntax_score"] += 1
            log_event("Syntax", row["language"], row["level"], True, +1)
        else:
            st.error("Not an exact match. Check spaces, keywords, and order.")
            log_event("Syntax", row["language"], row["level"], False, 0)

        if ai_enabled:
            with st.expander("AI Coach feedback"):
                prompt = f"""
Student is practicing {row['language']} syntax.

Task:
{row.get('description_en','')}

Correct syntax:
{expected}

Student answer:
{user_answer}

1) Is the answer correct / partially correct / wrong?
2) Point out differences.
3) Give a short explanation and one extra example.
"""
                st.markdown(ask_ai(prompt))

    if show_btn:
        st.info("Model answer:")
        st.code(expected, language="sql")

        usage = row.get("usage_example", "").strip()
        if usage:
            with st.expander("Usage example"):
                st.code(usage, language="sql")

    if st.session_state["syntax_attempts"] > 0:
        acc = st.session_state["syntax_score"] / max(st.session_state["syntax_attempts"], 1)
        st.write("")
        st.markdown("**Session progress:**")
        st.progress(acc)

    if new_btn:
        st.experimental_rerun()

# ======================================
# 12) Q-Lab section (Questions_Practice)
# ======================================
def render_qlab():
    st.markdown("#### 🎯 Q-Lab (Questions Practice)")

    if questions_df.empty:
        st.warning("Questions_Practice sheet is empty or not available.")
        return

    col1, col2 = st.columns([1.2, 1.2])
    with col1:
        level_options = sorted(questions_df["level"].unique())
        q_level = st.selectbox(
            "Level",
            options=level_options,
            index=0,
            key="qlab_level_select",
        )
    with col2:
        st.write("")

    filtered = questions_df.copy()
    filtered = filtered[filtered["language"].str.lower() == preferred_lang.lower()]
    filtered = filtered[filtered["level"] == q_level]

    row = pick_random_row(filtered)
    if row is None:
        st.warning("No matching questions for current filters.")
        return

    st.markdown(
        f"""
        <div class="card">
            <div>
                <span class="pill">{row['language']}</span>
                <span class="pill pill-level">{row['level']}</span>
            </div>
            <h4 style="margin-top:0.45rem;margin-bottom:0.25rem;">Question</h4>
            <p style="font-size:15px; color:#111827;">
                {row.get('question','').strip()}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_name = row.get("dataset_name", "").strip()
    df_data = None
    if dataset_name:
        df_data = generate_dataset(dataset_name)

    if df_data is not None:
        with st.expander(f"Dataset preview: {dataset_name}"):
            st.dataframe(df_data, use_container_width=True)
    else:
        preview = row.get("dataset_preview", "").strip()
        if preview:
            with st.expander("Dataset description"):
                st.markdown(preview)

    user_answer_q = st.text_area(
        "Your answer (SQL or code):",
        height=110,
        key="qlab_answer_textarea",
    )

    colq1, colq2, colq3 = st.columns([0.45, 0.3, 0.25])
    with colq1:
        check_q = st.button("Check answer", key="qlab_check_btn")
    with colq2:
        new_q = st.button("New question", key="qlab_new_btn")
    with colq3:
        show_q = st.button("Show model", key="qlab_show_btn")

    correct = row.get("correct_answer", "")

    if check_q:
        st.session_state["q_attempts"] += 1

        if normalize_text(user_answer_q) == normalize_text(correct):
            st.success("Correct answer. Great job! ✅")
            st.session_state["q_score"] += 1
            log_event("Questions", row["language"], row["level"], True, +2)
        else:
            st.error("Not correct or incomplete. Compare with the model answer.")
            log_event("Questions", row["language"], row["level"], False, 0)

        if ai_enabled:
            with st.expander("AI Coach analysis"):
                prompt = f"""
You are an instructor.

Question:
{row.get('question','')}

Correct answer:
{correct}

Student answer:
{user_answer_q}

1) Is the answer correct / partially correct / wrong?
2) Explain the key differences.
3) Provide a corrected version.
"""
                st.markdown(ask_ai(prompt))

    if show_q:
        st.info("Model answer:")
        st.code(correct, language="sql")

    if st.session_state["q_attempts"] > 0:
        acc2 = st.session_state["q_score"] / max(st.session_state["q_attempts"], 1)
        st.write("")
        st.markdown("**Session progress:**")
        st.progress(acc2)

    if new_q:
        st.experimental_rerun()

# ======================================
# 13) Documentation section
# ======================================
def render_docs():
    st.markdown("#### 📘 Docs Hub")

    if docs_df.empty:
        st.warning("Documentation sheet is empty or not available.")
        return

    docs_lang = docs_df[docs_df["language"].str.lower() == preferred_lang.lower()]
    if docs_lang.empty:
        st.warning("No documentation for current language yet.")
        return

    categories = sorted(docs_lang["category"].unique())
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        cat = st.selectbox(
            "Category",
            options=["All"] + categories,
            index=0,
            key="docs_category_select",
        )
    with col2:
        search = st.text_input(
            "Search (title / description / syntax):",
            "",
            key="docs_search_input",
        )

    docs_filtered = docs_lang.copy()
    if cat != "All":
        docs_filtered = docs_filtered[docs_filtered["category"] == cat]

    if search.strip():
        s = search.lower()
        docs_filtered = docs_filtered[
            docs_filtered["title"].str.lower().str.contains(s)
            | docs_filtered["description_en"].str.lower().str.contains(s)
            | docs_filtered["syntax"].str.lower().str.contains(s)
        ]

    if docs_filtered.empty:
        st.info("No documentation matches the current filters.")
        return

    for _, r in docs_filtered.iterrows():
        with st.expander(f"📦 {r.get('title','(no title)')}"):
            st.markdown(
                f"**Category:** `{r.get('category','')}` · **Language:** `{r.get('language','')}`"
            )
            if r.get("description_en", "").strip():
                st.markdown("**Description (EN):**")
                st.markdown(r["description_en"].strip())
            if r.get("description_ar", "").strip():
                st.markdown("**Description (AR):**")
                st.markdown(r["description_ar"].strip())
            if r.get("syntax", "").strip():
                st.markdown("**Syntax:**")
                st.code(r["syntax"].strip(), language="sql")
            if r.get("examples", "").strip():
                st.markdown("**Examples:**")
                st.code(r["examples"].strip(), language="sql")
            if r.get("notes", "").strip():
                st.markdown("**Notes:**")
                st.markdown(r["notes"].strip())

# ======================================
# 14) Leaderboard section (session-based)
# ======================================
def render_leaderboard():
    st.markdown("#### 🏆 Leaderboard (session only)")

    if not st.session_state["history"]:
        st.info("No activity yet. Solve some questions or syntax items first.")
        return

    hdf = pd.DataFrame(st.session_state["history"])

    agg = (
        hdf.groupby("user")
        .agg(
            total_events=("mode", "count"),
            correct=("correct", "sum"),
            score=("delta", "sum"),
        )
        .reset_index()
    )
    agg["accuracy"] = (agg["correct"] / agg["total_events"]).round(2)

    st.markdown("**User ranking (local session):**")
    st.dataframe(
        agg.sort_values(["score", "accuracy"], ascending=[False, False]),
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("**Raw history:**")
    st.dataframe(hdf, use_container_width=True)

# ======================================
# 15) Settings section
# ======================================
def render_settings():
    st.markdown("#### ⚙️ Settings")

    st.markdown(f"**Current username:** `{st.session_state['username']}`")
    st.markdown(f"**Preferred language:** `{preferred_lang}`")

    st.write("")
    st.markdown("**AI Coach status:**")
    if ai_enabled:
        st.success("AI Coach is active and ready.")
    else:
        st.warning("AI Coach is not active (OpenAI API key is missing or invalid).")

    st.write("")
    if st.button("Reset all session progress", key="settings_reset_all_btn"):
        st.session_state["syntax_score"] = 0
        st.session_state["syntax_attempts"] = 0
        st.session_state["q_score"] = 0
        st.session_state["q_attempts"] = 0
        st.session_state["history"] = []
        st.experimental_rerun()

    st.info(
        "This leaderboard and progress are session-based only. "
        "If you want persistent global leaderboard using Google Sheets, "
        "we can later add write-access with a service account."
    )

# ======================================
# 16) Router (Home + sections)
# ======================================
section = st.session_state["section"]

if section == "home":
    render_home()
elif section == "practice":
    render_practice()
elif section == "qlab":
    render_qlab()
elif section == "docs":
    render_docs()
elif section == "leaderboard":
    render_leaderboard()
elif section == "settings":
    render_settings()
else:
    render_home()

# ======================================
# 17) Footer
# ======================================
st.write("")
st.markdown(
    """
    <div style="text-align:center; padding:1.3rem 0 0.6rem 0;">
        <div class="footer-3d">
            MADE BY ABDULRHMAN · WHITE & MINT LAB
        </div>
        <div style="font-size:11px; color:#6b7280; margin-top:0.25rem;">
            Session-based only · No data is written back to Google Sheets yet.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
