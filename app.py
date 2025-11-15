import streamlit as st
import pandas as pd
import random
import string
import os
from datetime import datetime

# ======================================
# 0) محاولة استيراد OpenAI (AI Coach)
# ======================================
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ======================================
# 1) إعداد الصفحة العامة
# ======================================
st.set_page_config(
    page_title="Syntax Trainer Pro",
    page_icon="⚡",
    layout="wide",
)

# --------- Theme Toggle in Session ---------
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Neon Dark"

theme_mode = st.session_state["theme_mode"]


# ======================================
# 2) CSS حسب الثيم (Neon / Light)
# ======================================
if theme_mode == "Neon Dark":
    bg_css = """
    .main {
        background: radial-gradient(circle at top, #111827, #020617);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    """
else:
    bg_css = """
    .main {
        background: radial-gradient(circle at top, #e5e7eb, #f9fafb);
        color: #020617;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    """

st.markdown(
    f"""
    <style>
    {bg_css}

    .stButton>button {{
        border-radius: 999px;
        padding: 0.45rem 1.3rem;
        font-weight: 600;
        border: 1px solid rgba(148,163,184,0.6);
        background: linear-gradient(135deg, #22d3ee, #6366f1);
        color: white;
        box-shadow: 0 0 12px rgba(56,189,248,0.45);
    }}
    .stButton>button:hover {{
        border-color: rgba(248,250,252,0.9);
        box-shadow: 0 0 22px rgba(129,140,248,0.8);
        transform: translateY(-1px);
    }}
    .badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 600;
        background: rgba(56,189,248,0.07);
        color: #a5f3fc;
        border: 1px solid rgba(56,189,248,0.7);
        margin-right: 6px;
    }}
    .badge-level {{
        background: rgba(251,191,36,0.08);
        color: #fed7aa;
        border-color: rgba(251,191,36,0.7);
    }}
    .badge-mode {{
        background: rgba(94,234,212,0.08);
        color: #99f6e4;
        border-color: rgba(94,234,212,0.7);
    }}
    .card {{
        border-radius: 20px;
        padding: 1.1rem 1.4rem;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.20);
        backdrop-filter: blur(12px);
    }}
    .card-light {{
        border-radius: 20px;
        padding: 1.1rem 1.4rem;
        background: rgba(255,255,255,0.96);
        border: 1px solid rgba(148,163,184,0.25);
        backdrop-filter: blur(12px);
    }}
    .subtitle {{
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        color: rgba(148,163,184,0.85);
        margin-bottom: 0.1rem;
    }}
    .title-main {{
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(90deg, #e0f2fe, #a855f7, #f97316);
        -webkit-background-clip: text;
        color: transparent;
        margin-bottom: 0.35rem;
    }}
    .small-label {{
        font-size: 12px;
        color: rgba(148,163,184,0.9);
    }}
    .footer-3d {{
        font-family: "Space Grotesk", system-ui;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #a5b4fc;
        text-shadow:
            0 0 4px rgba(129,140,248,.8),
            0 0 18px rgba(56,189,248,.6),
            0 0 28px rgba(236,72,153,.5);
        transform: translateZ(0);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================================
# 3) إعداد Google Sheets (قراءة فقط الآن)
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
# 4) إعداد OpenAI (AI Coach)
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
else:
    ai_enabled = False


def ask_ai(prompt: str) -> str:
    if not ai_enabled:
        return "❌ AI Coach غير مفعّل (تأكد من إعداد OPENAI_API_KEY في secrets.toml)."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that helps students practice SQL, Pandas, NumPy, and scikit-learn."
                        "You explain briefly but clearly, mixing Arabic with English keywords."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI Error: {e}"


# ======================================
# 5) Helpers: Normalize, Random Row, Dataset
# ======================================
def normalize_sql(s: str) -> str:
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
# 6) Session State for Scores / History
# ======================================
if "syntax_score" not in st.session_state:
    st.session_state["syntax_score"] = 0

if "syntax_attempts" not in st.session_state:
    st.session_state["syntax_attempts"] = 0

if "q_score" not in st.session_state:
    st.session_state["q_score"] = 0

if "q_attempts" not in st.session_state:
    st.session_state["q_attempts"] = 0

if "history" not in st.session_state:
    st.session_state["history"] = []

if "username" not in st.session_state:
    st.session_state["username"] = "Guest"


def log_event(mode: str, language: str, level: str, correct: bool, delta: int):
    st.session_state["history"].append(
        {
            "time": datetime.utcnow().isoformat(timespec="seconds"),
            "user": st.session_state["username"],
            "mode": mode,
            "language": language,
            "level": level,
            "correct": correct,
            "delta": delta,
        }
    )


# ======================================
# 7) Sidebar: User + Theme + Stats
# ======================================
with st.sidebar:
    st.markdown("### ⚡ Syntax Trainer Pro")
    st.markdown(
        "<p class='small-label'>Neon training lab for SQL · Pandas · NumPy · sklearn.</p>",
        unsafe_allow_html=True,
    )

    st.session_state["username"] = st.text_input(
        "Your username (for Leaderboard):",
        value=st.session_state["username"],
    )

    theme_mode = st.selectbox(
        "Theme mode:",
        ["Neon Dark", "Clean Light"],
        index=0 if st.session_state["theme_mode"] == "Neon Dark" else 1,
    )
    st.session_state["theme_mode"] = theme_mode

    st.markdown("---")

    # استنتاج اللغات المتوفرة من الشيتات
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
    preferred_lang = st.selectbox("Preferred language:", options=langs, index=0)

    st.markdown("---")
    st.markdown("<span class='small-label'>Session Progress</span>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.write(f"Syntax ✅ {st.session_state['syntax_score']}/{st.session_state['syntax_attempts']}")
    with col_s2:
        st.write(f"Q-Lab ✅ {st.session_state['q_score']}/{st.session_state['q_attempts']}")

    if st.button("Reset session stats"):
        st.session_state["syntax_score"] = 0
        st.session_state["syntax_attempts"] = 0
        st.session_state["q_score"] = 0
        st.session_state["q_attempts"] = 0
        st.session_state["history"] = []
        st.experimental_rerun()

    st.markdown("---")
    if ai_enabled:
        st.success("✅ AI Coach Enabled")
    else:
        st.warning("⚠️ AI Coach Disabled (missing OPENAI_API_KEY)")


# ======================================
# 8) Header with Lottie-like Glow (no external JS)
# ======================================
if theme_mode == "Neon Dark":
    card_class = "card"
else:
    card_class = "card-light"

st.markdown(
    f"""
    <div class="{card_class}">
        <div class='subtitle'>Interactive Neon Lab</div>
        <div class='title-main'>Syntax Trainer Pro</div>
        <p style="font-size:13px; color:rgba(148,163,184,0.95); max-width:720px;">
            درّب نفسك على كتابة الـ syntax بدون حفظ أعمى. جاوب، شوف الـ dataset، 
            وخلي الـ AI Coach يشرح لك الأخطاء بخليط عربي/إنقليزي. 
            الLevels جاهزة: Beginner · Intermediate · Advanced.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ======================================
# 9) Tabs (UI PRO)
# ======================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Syntax Trainer", "Questions Lab", "Docs Hub", "Leaderboard", "Dashboard", "Profile"]
)

# --------------------------------------
# TAB 1: Syntax Trainer
# --------------------------------------
with tab1:
    st.markdown("#### 🧠 Syntax Trainer")

    if syntax_df.empty:
        st.warning("جدول Syntax_Practice فارغ أو غير متوفر.")
    else:
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            level_options = sorted(syntax_df["level"].unique())
            level = st.selectbox("Level", options=level_options, index=0)

        with col2:
            categories = sorted(syntax_df["category"].unique())
            category = st.selectbox("Category", options=["All"] + categories)

        with col3:
            st.write("")

        filtered = syntax_df.copy()
        filtered = filtered[filtered["language"].str.lower() == preferred_lang.lower()]
        if category != "All":
            filtered = filtered[filtered["category"] == category]

        row = pick_random_row(filtered)
        if row is None:
            st.warning("ما فيه سجلات مطابقة للفلترة.")
        else:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div>
                        <span class='badge'>{row['language']}</span>
                        <span class='badge badge-level'>{row['level']}</span>
                        <span class='badge badge-mode'>{row['category']}</span>
                    </div>
                    <h4 style="margin-top:0.5rem; margin-bottom:0.3rem;">Task 🎯</h4>
                    <p style="font-size:14px; color:rgba(226,232,240,0.94); margin-bottom:0.2rem;">
                        {row.get('description_en','').strip()}
                    </p>
                    <p style="font-size:13px; color:rgba(148,163,184,0.95); direction:rtl; text-align:right;">
                        {row.get('description_ar','').strip()}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            user_answer = st.text_area(
                "✍️ اكتب الـ syntax بالضبط:",
                height=90,
                key="syntax_answer_input_tab1",
            )

            colb1, colb2, colb3 = st.columns([0.5, 0.25, 0.25])
            with colb1:
                check_clicked = st.button("Check", type="primary", key="syntax_check_btn")
            with colb2:
                new_q = st.button("🔁 New", key="syntax_new_btn")
            with colb3:
                show_solution = st.button("👀 Show model answer", key="syntax_show_ans_btn")

            if check_clicked:
                st.session_state["syntax_attempts"] += 1
                expected = row.get("syntax", "")
                if normalize_sql(user_answer) == normalize_sql(expected):
                    st.success("✅ Perfect syntax! 👑")
                    st.session_state["syntax_score"] += 1
                    log_event("Syntax", row["language"], row["level"], True, +1)
                else:
                    st.error("❌ الإجابة غير مطابقة بالكامل.")
                    log_event("Syntax", row["language"], row["level"], False, 0)

                if ai_enabled:
                    with st.expander("🔎 AI Coach Feedback"):
                        prompt = f"""
You are helping a student practice {row['language']} syntax.

Task:
{row.get('description_en','')}

Correct syntax:
{expected}

Student answer:
{user_answer}

1) Say if the answer is correct / partially correct / wrong.
2) Highlight missing or extra parts.
3) Provide a short explanation in Arabic mixed with SQL keywords.
4) Provide one extra correct example.
"""
                        st.markdown(ask_ai(prompt))

            if show_solution:
                expected = row.get("syntax", "")
                st.info("الإجابة النموذجية:")
                st.code(expected, language="sql")

                if row.get("usage_example", "").strip():
                    with st.expander("Usage example / مثال استخدام"):
                        st.code(row["usage_example"], language="sql")

            # Progress bar
            if st.session_state["syntax_attempts"] > 0:
                acc = st.session_state["syntax_score"] / max(st.session_state["syntax_attempts"], 1)
                st.write("")
                st.markdown("**Progress:**")
                st.progress(acc)


            if new_q:
                st.experimental_rerun()

# --------------------------------------
# TAB 2: Questions Lab
# --------------------------------------
with tab2:
    st.markdown("#### 🧪 Questions Lab (SQL / others)")

    if questions_df.empty:
        st.warning("جدول Questions_Practice فارغ أو غير متوفر.")
    else:
        col1, col2 = st.columns([1, 1])
        with col1:
            level_options = sorted(questions_df["level"].unique())
            q_level = st.selectbox("Level", options=level_options, index=0)
        with col2:
            st.write("")

        filtered_q = questions_df.copy()
        filtered_q = filtered_q[filtered_q["language"].str.lower() == preferred_lang.lower()]
        filtered_q = filtered_q[filtered_q["level"] == q_level]

        row_q = pick_random_row(filtered_q)
        if row_q is None:
            st.warning("ما فيه أسئلة مطابقة للفلترة.")
        else:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div>
                        <span class='badge'>{row_q['language']}</span>
                        <span class='badge badge-level'>{row_q['level']}</span>
                    </div>
                    <h4 style="margin-top:0.45rem; margin-bottom:0.3rem;">Question 🎯</h4>
                    <p style="font-size:15px; color:rgba(226,232,240,0.98);">
                        {row_q.get('question','').strip()}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Dataset display
            dataset_name = row_q.get("dataset_name", "").strip()
            df_data = None
            if dataset_name:
                df_data = generate_dataset(dataset_name)
            if df_data is not None:
                with st.expander(f"📊 Dataset: {dataset_name}"):
                    st.dataframe(df_data, use_container_width=True)
            else:
                preview = row_q.get("dataset_preview", "").strip()
                if preview:
                    with st.expander("📊 Dataset preview (نصي)"):
                        st.markdown(preview)

            user_answer_q = st.text_area(
                "✍️ اكتب الحل (SQL أو حسب اللغة):",
                height=110,
                key="q_answer_input",
            )

            colq1, colq2, colq3 = st.columns([0.5, 0.25, 0.25])
            with colq1:
                check_q = st.button("Check answer", type="primary", key="q_check_btn")
            with colq2:
                newq2 = st.button("🔁 New question", key="q_new_btn")
            with colq3:
                show_q_model = st.button("👀 Show model", key="q_show_model_btn")

            correct = row_q.get("correct_answer", "")

            if check_q:
                st.session_state["q_attempts"] += 1
                if normalize_sql(user_answer_q) == normalize_sql(correct):
                    st.success("✅ إجابة صحيحة! ممتاز.")
                    st.session_state["q_score"] += 1
                    log_event("Questions", row_q["language"], row_q["level"], True, +2)
                else:
                    st.error("❌ الإجابة غير صحيحة أو ناقصة.")
                    log_event("Questions", row_q["language"], row_q["level"], False, 0)

                if ai_enabled:
                    with st.expander("🔎 AI Coach – تحليل الإجابة"):
                        prompt = f"""
You are an instructor.

Question:
{row_q.get('question','')}

Correct answer:
{correct}

Student answer:
{user_answer_q}

1) Is the answer correct / partially correct / wrong?
2) Explain the differences (filters, joins, grouping, etc.).
3) Suggest a correct version.
4) Explain in Arabic (with English SQL keywords).
"""
                        st.markdown(ask_ai(prompt))

            if show_q_model:
                st.info("الإجابة النموذجية:")
                st.code(correct, language="sql")

            if st.session_state["q_attempts"] > 0:
                acc2 = st.session_state["q_score"] / max(st.session_state["q_attempts"], 1)
                st.write("")
                st.markdown("**Progress:**")
                st.progress(acc2)

            if newq2:
                st.experimental_rerun()

# --------------------------------------
# TAB 3: Docs Hub
# --------------------------------------
with tab3:
    st.markdown("#### 📚 Docs Hub")

    if docs_df.empty:
        st.warning("جدول Documentation فارغ أو غير متوفر.")
    else:
        docs_lang = docs_df[docs_df["language"].str.lower() == preferred_lang.lower()]
        if docs_lang.empty:
            st.warning("ما فيه توثيق للغة المختارة.")
        else:
            categories = sorted(docs_lang["category"].unique())
            col1, col2 = st.columns([1, 2])
            with col1:
                cat = st.selectbox("Category", options=["All"] + categories)
            with col2:
                q = st.text_input("Search (title / description / syntax):", "")

            docs_filtered = docs_lang.copy()
            if cat != "All":
                docs_filtered = docs_filtered[docs_filtered["category"] == cat]

            if q.strip():
                s = q.lower()
                docs_filtered = docs_filtered[
                    docs_filtered["title"].str.lower().str.contains(s)
                    | docs_filtered["description_en"].str.lower().str.contains(s)
                    | docs_filtered["syntax"].str.lower().str.contains(s)
                ]

            if docs_filtered.empty:
                st.info("ما فيه نتائج مطابقة.")
            else:
                for _, r in docs_filtered.iterrows():
                    with st.expander(f"📦 {r.get('title','(no title)')}"):
                        st.markdown(
                            f"**Category:** `{r.get('category','')}` &nbsp; | "
                            f"**Language:** `{r.get('language','')}`"
                        )
                        if r.get("description_en", "").strip():
                            st.markdown("**Description (EN):**")
                            st.markdown(r["description_en"].strip())
                        if r.get("description_ar", "").strip():
                            st.markdown("**الوصف (AR):**")
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

# --------------------------------------
# TAB 4: Leaderboard (جلسة فقط حالياً)
# --------------------------------------
with tab4:
    st.markdown("#### 🏆 Leaderboard (Session-based)")

    if not st.session_state["history"]:
        st.info("ما فيه بيانات بعد. جرّب تحل أسئلة أول.")
    else:
        hist_df = pd.DataFrame(st.session_state["history"])
        # تجميع حسب المستخدم
        agg = (
            hist_df.groupby("user")
            .agg(
                total_events=("mode", "count"),
                correct=("correct", "sum"),
                score=("delta", "sum"),
            )
            .reset_index()
        )

        agg["accuracy"] = (agg["correct"] / agg["total_events"]).round(2)

        st.markdown("**Ranking (local session only):**")
        st.dataframe(
            agg.sort_values(["score", "accuracy"], ascending=[False, False]),
            use_container_width=True,
        )

        st.markdown("**Raw history:**")
        st.dataframe(hist_df, use_container_width=True)

# --------------------------------------
# TAB 5: Dashboard
# --------------------------------------
with tab5:
    st.markdown("#### 📊 Dashboard (Session Analytics)")

    if not st.session_state["history"]:
        st.info("ما فيه بيانات بعد.")
    else:
        hdf = pd.DataFrame(st.session_state["history"])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total attempts", len(hdf))
        with col2:
            st.metric("Correct", int(hdf["correct"].sum()))
        with col3:
            acc = hdf["correct"].mean()
            st.metric("Accuracy", f"{acc*100:.1f}%")

        st.markdown("---")
        st.markdown("**Attempts per mode:**")
        mode_counts = hdf.groupby("mode")["mode"].count().reset_index(name="count")
        st.dataframe(mode_counts, use_container_width=True)

        st.markdown("**Attempts per language:**")
        lang_counts = hdf.groupby("language")["language"].count().reset_index(name="count")
        st.dataframe(lang_counts, use_container_width=True)

        st.markdown("**Attempts per level:**")
        lvl_counts = hdf.groupby("level")["level"].count().reset_index(name="count")
        st.dataframe(lvl_counts, use_container_width=True)

# --------------------------------------
# TAB 6: Profile
# --------------------------------------
with tab6:
    st.markdown("#### 👤 Profile & Levels")

    st.markdown(f"**Username:** `{st.session_state['username']}`")
    st.write("")
    st.markdown("**Your current session stats:**")
    st.write(f"- Syntax Practice: {st.session_state['syntax_score']} / {st.session_state['syntax_attempts']}")
    st.write(f"- Questions Lab: {st.session_state['q_score']} / {st.session_state['q_attempts']}")

    total_correct = st.session_state["syntax_score"] + st.session_state["q_score"]
    if total_correct < 10:
        lvl = "Beginner"
    elif total_correct < 30:
        lvl = "Intermediate"
    else:
        lvl = "Advanced"

    st.write("")
    st.markdown(f"### 🏅 Your Level: **{lvl}**")

    st.markdown(
        """
        - **Beginner:** أقل من 10 إجابات صحيحة.
        - **Intermediate:** بين 10 و 29 إجابة صحيحة.
        - **Advanced:** 30 فأكثر في نفس الجلسة.
        """
    )

    st.info(
        "حاليًا الـ Profile و Leaderboard مبنية على جلسة المتصفح فقط. "
        "لو حاب نخليها محفوظة في Google Sheets لكل المستخدمين، نحتاج نرجع نضيف خدمة كتابة (gspread + service account)."
    )

# ======================================
# Footer 3D
# ======================================
st.write("")
st.markdown(
    """
    <div style="text-align:center; padding:1.3rem 0;">
        <div class="footer-3d">
            MADE BY ABDULRHMAN · NEON SYNTAX LAB
        </div>
        <div style="font-size:11px; color:rgba(148,163,184,0.85); margin-top:0.3rem;">
            Session-based only · No data is written back to Google Sheets yet.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
