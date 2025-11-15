import streamlit as st
import random
import difflib
import json
import os
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials
from openai import OpenAI
import pandas as pd
# ================================
# GOOGLE SHEETS CONNECTION
# ================================

def connect_gsheets():
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.error(f"❌ Google Sheets connection failed: {e}")
        return None

client = connect_gsheets()

if client:
    try:
        SH = client.open("Syntax_Pro_DB")

        def load_sheet(sheet_name):
            ws = SH.worksheet(sheet_name)
            data = ws.get_all_records()
            return pd.DataFrame(data)

        # Load main sheets
        syntax_df        = load_sheet("Syntax_Practice")
        questions_df     = load_sheet("Questions_Practice")
        docs_df          = load_sheet("Documentation")
        users_df         = load_sheet("Users")
        progress_df      = load_sheet("Progress")

        st.success("✅ Connected to Google Sheets Successfully")

    except Exception as e:
        st.error(f"❌ Error loading Sheets: {e}")

# =========================
# APP CONFIG
# =========================

st.set_page_config(
    page_title="Syntax Trainer Pro",
    page_icon="🔍",
    layout="wide"
)

SHEET_NAME = "Syntax_Pro_DB"

# =========================
# GLOBAL STYLE
# =========================

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #111827, #020617);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app-title {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(90deg,#38bdf8,#a855f7,#f97316);
        -webkit-background-clip: text;
        color: transparent;
        letter-spacing: .04em;
        margin-bottom: 0.1rem;
    }
    .app-subtitle {
        font-size: 13px;
        color: #9ca3af;
        margin-bottom: 1rem;
    }
    .card {
        background: rgba(15,23,42,0.96);
        border-radius: 18px;
        padding: 20px 24px;
        box-shadow: 0 18px 45px rgba(0,0,0,0.45);
        border: 1px solid rgba(148,163,184,0.24);
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 600;
        background: linear-gradient(90deg,#4f46e5,#06b6d4);
        color: white;
        margin-bottom: 6px;
    }
    .lvl-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .08em;
        color: #9ca3af;
        margin-bottom: -3px;
    }
    .lvl-value {
        font-size: 15px;
        font-weight: 600;
        color: #e5e7eb;
    }
    .small-hint {
        font-size: 11px;
        color: #9ca3af;
    }
    .doc-section-title {
        font-size: 16px;
        font-weight: 700;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .footer-3d {
        font-size: 11px;
        text-align: center;
        margin-top: 32px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-weight: 700;
        text-shadow:
            1px 1px 0px #0f172a,
            2px 2px 0px #020617,
            3px 3px 6px rgba(0,0,0,0.7);
    }
    .footer-3d span.main {
        background: linear-gradient(90deg,#38bdf8,#a855f7,#f97316);
        -webkit-background-clip: text;
        color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# GOOGLE SHEETS CLIENT
# =========================

@st.cache_resource(show_spinner=False)
def get_sheets_client():
    try:
        service_info = st.secrets["gcp_service_account"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_info(service_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open(SHEET_NAME)
        return {
            "users": sh.worksheet("Users"),
            "progress": sh.worksheet("Progress"),
            "custom": sh.worksheet("Custom Syntax"),
            "logs": sh.worksheet("AI_Agent_Logs")
        }
    except Exception as e:
        st.sidebar.error(f"Google Sheets connection failed: {e}")
        return None


sheets = get_sheets_client()

# =========================
# OPENAI CLIENT
# =========================

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

# =========================
# SYNTAX DATA
# =========================

SYNTAX_ITEMS = [
    # --- SQL Beginner ---
    {"language": "SQL", "level": "Beginner", "text": "SELECT",
     "description": "Choose which columns to return from a table.",
     "usage": "SELECT column1, column2 FROM customers;"},
    {"language": "SQL", "level": "Beginner", "text": "FROM",
     "description": "Specify the table to read data from.",
     "usage": "SELECT * FROM orders;"},
    {"language": "SQL", "level": "Beginner", "text": "WHERE",
     "description": "Filter rows based on a condition.",
     "usage": "SELECT * FROM orders WHERE amount > 100;"},
    {"language": "SQL", "level": "Beginner", "text": "ORDER BY",
     "description": "Sort the result set by one or more columns.",
     "usage": "SELECT * FROM products ORDER BY price DESC;"},
    {"language": "SQL", "level": "Beginner", "text": "GROUP BY",
     "description": "Group rows that share the same value in columns.",
     "usage": "SELECT status, COUNT(*) FROM orders GROUP BY status;"},
    {"language": "SQL", "level": "Beginner", "text": "LIMIT",
     "description": "Limit the number of rows returned.",
     "usage": "SELECT * FROM customers LIMIT 10;"},

    # --- SQL Intermediate ---
    {"language": "SQL", "level": "Intermediate", "text": "INNER JOIN",
     "description": "Return rows when there is a match in both tables.",
     "usage": "SELECT * FROM orders INNER JOIN customers ON orders.customer_id = customers.id;"},
    {"language": "SQL", "level": "Intermediate", "text": "LEFT JOIN",
     "description": "Return all rows from the left table and matched rows from the right.",
     "usage": "SELECT * FROM customers LEFT JOIN orders ON customers.id = orders.customer_id;"},
    {"language": "SQL", "level": "Intermediate", "text": "BETWEEN",
     "description": "Filter values within a range (inclusive).",
     "usage": "SELECT * FROM orders WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31';"},
    {"language": "SQL", "level": "Intermediate", "text": "HAVING",
     "description": "Filter groups after GROUP BY.",
     "usage": "SELECT customer_id, SUM(amount) AS total "
              "FROM orders GROUP BY customer_id HAVING SUM(amount) > 500;"},
    {"language": "SQL", "level": "Intermediate", "text": "DISTINCT",
     "description": "Return only unique values.",
     "usage": "SELECT DISTINCT city FROM customers;"},

    # --- SQL Advanced ---
    {"language": "SQL", "level": "Advanced", "text": "PARTITION BY",
     "description": "Split rows into partitions for window functions.",
     "usage": "ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY order_date) AS rn"},
    {"language": "SQL", "level": "Advanced", "text": "ROW_NUMBER",
     "description": "Assign a unique sequential number to rows.",
     "usage": "ROW_NUMBER() OVER(ORDER BY amount DESC) AS row_num"},
    {"language": "SQL", "level": "Advanced", "text": "DENSE_RANK",
     "description": "Rank rows without gaps when there are ties.",
     "usage": "DENSE_RANK() OVER(ORDER BY score DESC) AS dense_rank"},
    {"language": "SQL", "level": "Advanced", "text": "LAG",
     "description": "Access a value from a previous row.",
     "usage": "LAG(amount, 1) OVER(ORDER BY order_date) AS previous_amount"},
    {"language": "SQL", "level": "Advanced", "text": "WITH",
     "description": "Start a Common Table Expression (CTE).",
     "usage": "WITH top_customers AS (SELECT * FROM customers WHERE total_spent > 1000) "
              "SELECT * FROM top_customers;"},

    # --- Pandas Beginner ---
    {"language": "Pandas", "level": "Beginner", "text": "df.head()",
     "description": "Show the first N rows of the DataFrame.",
     "usage": "df.head(10)"},
    {"language": "Pandas", "level": "Beginner", "text": "df.info()",
     "description": "Print a concise summary of the DataFrame.",
     "usage": "df.info()"},
    {"language": "Pandas", "level": "Beginner", "text": "df.describe()",
     "description": "Descriptive statistics for numeric columns.",
     "usage": "df.describe()"},
    {"language": "Pandas", "level": "Beginner", "text": "df.columns",
     "description": "Return the column labels.",
     "usage": "df.columns"},

    # --- Pandas Intermediate ---
    {"language": "Pandas", "level": "Intermediate", "text": "df.groupby()",
     "description": "Group data by one or more keys.",
     "usage": "df.groupby('city')['sales'].sum()"},
    {"language": "Pandas", "level": "Intermediate", "text": "df.merge()",
     "description": "Join two DataFrames on a key.",
     "usage": "df.merge(df2, on='id', how='left')"},
    {"language": "Pandas", "level": "Intermediate", "text": "df.drop()",
     "description": "Drop rows or columns.",
     "usage": "df.drop(columns=['unnecessary_col'], inplace=True)"},
    {"language": "Pandas", "level": "Intermediate", "text": "df.fillna()",
     "description": "Fill missing values.",
     "usage": "df['age'].fillna(df['age'].median(), inplace=True)"},

    # --- Pandas Advanced ---
    {"language": "Pandas", "level": "Advanced", "text": "df.apply()",
     "description": "Apply a function along an axis.",
     "usage": "df['price_with_tax'] = df['price'].apply(lambda x: x * 1.15)"},
    {"language": "Pandas", "level": "Advanced", "text": "df.query()",
     "description": "Query the DataFrame with an expression.",
     "usage": "df.query('sales > 100 & city == \"Riyadh\"')"},
    {"language": "Pandas", "level": "Advanced", "text": "df.pivot_table()",
     "description": "Create a pivot table.",
     "usage": "df.pivot_table(values='sales', index='city', columns='product', aggfunc='sum')"},

    # --- NumPy Beginner ---
    {"language": "NumPy", "level": "Beginner", "text": "np.array()",
     "description": "Create a NumPy array.",
     "usage": "arr = np.array([1, 2, 3])"},
    {"language": "NumPy", "level": "Beginner", "text": "np.arange()",
     "description": "Range of evenly spaced values.",
     "usage": "arr = np.arange(0, 10, 2)"},
    {"language": "NumPy", "level": "Beginner", "text": "np.linspace()",
     "description": "Evenly spaced numbers over interval.",
     "usage": "arr = np.linspace(0, 1, 5)"},

    # --- NumPy Intermediate ---
    {"language": "NumPy", "level": "Intermediate", "text": "np.mean()",
     "description": "Compute the mean.",
     "usage": "np.mean(arr)"},
    {"language": "NumPy", "level": "Intermediate", "text": "np.std()",
     "description": "Compute the standard deviation.",
     "usage": "np.std(arr)"},
    {"language": "NumPy", "level": "Intermediate", "text": "np.sum()",
     "description": "Sum all elements.",
     "usage": "np.sum(arr)"},

    # --- NumPy Advanced ---
    {"language": "NumPy", "level": "Advanced", "text": "arr.reshape()",
     "description": "Reshape an array.",
     "usage": "arr.reshape(2, 3)"},
    {"language": "NumPy", "level": "Advanced", "text": "np.dot()",
     "description": "Dot product of two arrays.",
     "usage": "np.dot(a, b)"},

    # --- Sklearn Beginner ---
    {"language": "Scikit-learn", "level": "Beginner", "text": "train_test_split()",
     "description": "Split data into train/test sets.",
     "usage": "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"},
    {"language": "Scikit-learn", "level": "Beginner", "text": "model.fit()",
     "description": "Fit model to training data.",
     "usage": "model.fit(X_train, y_train)"},
    {"language": "Scikit-learn", "level": "Beginner", "text": "model.predict()",
     "description": "Predict using trained model.",
     "usage": "y_pred = model.predict(X_test)"},

    # --- Sklearn Intermediate ---
    {"language": "Scikit-learn", "level": "Intermediate", "text": "StandardScaler()",
     "description": "Standardize features.",
     "usage": "scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)"},
    {"language": "Scikit-learn", "level": "Intermediate", "text": "accuracy_score()",
     "description": "Classification accuracy.",
     "usage": "accuracy_score(y_test, y_pred)"},
    {"language": "Scikit-learn", "level": "Intermediate", "text": "mean_squared_error()",
     "description": "Regression MSE.",
     "usage": "mean_squared_error(y_test, y_pred)"},

    # --- Sklearn Advanced ---
    {"language": "Scikit-learn", "level": "Advanced", "text": "Pipeline()",
     "description": "Chain transformers and estimator.",
     "usage": "pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])"},
    {"language": "Scikit-learn", "level": "Advanced", "text": "GridSearchCV()",
     "description": "Exhaustive parameter search.",
     "usage": "gs = GridSearchCV(model, param_grid, cv=5); gs.fit(X_train, y_train)"}
]

# =========================
# STATE HELPERS
# =========================

def init_state():
    if "score" not in st.session_state:
        st.session_state.score = {"Beginner": 0, "Intermediate": 0, "Advanced": 0}
    if "attempts" not in st.session_state:
        st.session_state.attempts = {"Beginner": 0, "Intermediate": 0, "Advanced": 0}
    if "current_item" not in st.session_state:
        st.session_state.current_item = None
    if "custom_items" not in st.session_state:
        st.session_state.custom_items = []
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = ""
    if "preferred_language" not in st.session_state:
        st.session_state.preferred_language = "SQL"


init_state()

def get_items(language: str, level: str):
    base = [i for i in SYNTAX_ITEMS if i["language"] == language and i["level"] == level]
    custom = [i for i in st.session_state.custom_items
              if i["language"] == language and i["level"] == level]
    return base + custom

def pick_new_question(language: str, level: str):
    pool = get_items(language, level)
    st.session_state.current_item = random.choice(pool) if pool else None

# =========================
# GOOGLE SHEETS HELPERS
# =========================

def create_or_load_user():
    if sheets is None:
        return
    if st.session_state.user_id is not None:
        return

    users_ws = sheets["users"]
    rows = users_ws.get_all_values()
    next_id = len(rows)  # header = 1, أول مستخدم يكون id=1

    user_name = st.session_state.user_name.strip() or f"Guest-{next_id}"
    preferred_language = st.session_state.preferred_language
    now = datetime.utcnow().isoformat()

    users_ws.append_row([
        next_id,
        user_name,
        preferred_language,
        now,
        now
    ])
    st.session_state.user_id = next_id

def update_user_last_login():
    if sheets is None or st.session_state.user_id is None:
        return
    users_ws = sheets["users"]
    data = users_ws.get_all_records()
    uid = st.session_state.user_id
    now = datetime.utcnow().isoformat()
    for idx, row in enumerate(data, start=2):
        if int(row["user_id"]) == uid:
            users_ws.update_cell(idx, 5, now)
            break

def update_user_profile():
    if sheets is None or st.session_state.user_id is None:
        return
    users_ws = sheets["users"]
    data = users_ws.get_all_records()
    uid = st.session_state.user_id
    name = st.session_state.user_name.strip() or f"Guest-{uid}"
    pref = st.session_state.preferred_language
    for idx, row in enumerate(data, start=2):
        if int(row["user_id"]) == uid:
            users_ws.update_cell(idx, 2, name)
            users_ws.update_cell(idx, 3, pref)
            break

def save_progress(level: str):
    if sheets is None or st.session_state.user_id is None:
        return
    ws = sheets["progress"]
    uid = st.session_state.user_id
    attempts = st.session_state.attempts[level]
    score = st.session_state.score[level]
    now = datetime.utcnow().isoformat()
    ws.append_row([uid, level, attempts, score, now])

def save_custom_syntax(user_id, language, level, text, description, usage):
    if sheets is None:
        return
    ws = sheets["custom"]
    now = datetime.utcnow().isoformat()
    ws.append_row([user_id, language, level, text, description, usage, now])

def save_ai_log(user_id, input_code, result_correct, explanation, suggestion):
    if sheets is None:
        return
    ws = sheets["logs"]
    now = datetime.utcnow().isoformat()
    ws.append_row([user_id, input_code, result_correct, explanation, suggestion, now])

def get_leaderboard(top_n: int = 10):
    if sheets is None:
        return None
    progress_ws = sheets["progress"]
    users_ws = sheets["users"]

    progress_records = progress_ws.get_all_records()
    if not progress_records:
        return None

    # نحسب أفضل سكور لكل مستخدم (أكبر قيمة score مسجلة)
    best_scores = {}
    for row in progress_records:
        try:
            uid = int(row["user_id"])
            score = int(row.get("score", 0))
        except Exception:
            continue
        if uid not in best_scores or score > best_scores[uid]:
            best_scores[uid] = score

    user_records = users_ws.get_all_records()
    id_to_name = {int(r["user_id"]): r.get("name", f"User {r['user_id']}") for r in user_records}
    id_to_pref = {int(r["user_id"]): r.get("preferred_language", "") for r in user_records}

    rows = []
    for uid, total_score in best_scores.items():
        rows.append({
            "User ID": uid,
            "Name": id_to_name.get(uid, f"User {uid}"),
            "Preferred Lang": id_to_pref.get(uid, ""),
            "Best Score": total_score
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df = df.sort_values("Best Score", ascending=False).head(top_n).reset_index(drop=True)
    df.index = df.index + 1  # ترتيب من 1
    return df

# =========================
# SIDEBAR
# =========================

with st.sidebar:
    st.markdown("### 🔍 Syntax Trainer Pro")

    st.session_state.user_name = st.text_input(
        "Your name (optional):", value=st.session_state.user_name
    )
    st.session_state.preferred_language = st.selectbox(
        "Preferred language:",
        ["SQL", "Pandas", "NumPy", "Scikit-learn"],
        index=["SQL", "Pandas", "NumPy", "Scikit-learn"].index(
            st.session_state.preferred_language
        )
    )

    mode = st.radio(
        "Mode",
        ["Practice", "Documentation", "AI Agent", "Leaderboard"],
        index=0
    )

    st.markdown("---")
    st.markdown('<div class="lvl-label">Session Score</div>', unsafe_allow_html=True)
    for lvl in ["Beginner", "Intermediate", "Advanced"]:
        att = max(st.session_state.attempts[lvl], 1)
        st.markdown(
            f'<div class="lvl-value">{lvl}: {st.session_state.score[lvl]} / {att}</div>',
            unsafe_allow_html=True
        )
    st.markdown(
        '<p class="small-hint">Scores history is stored into Google Sheets.</p>',
        unsafe_allow_html=True
    )

    if st.button("Save profile & preferences"):
        if sheets is not None:
            create_or_load_user()
            update_user_profile()
            update_user_last_login()
            st.success("Profile saved to Google Sheets.")
        else:
            st.warning("Google Sheets is not available.")

# إنشاء المستخدم تلقائياً إن أمكن
if sheets is not None:
    create_or_load_user()
    update_user_last_login()

# =========================
# HEADER
# =========================

st.markdown('<div class="app-title">Syntax Trainer Pro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    'Multi-level syntax trainer with AI coaching, Google Sheets tracking, '
    'and a global leaderboard.'
    '</div>',
    unsafe_allow_html=True
)

# =========================
# MODES
# =========================

# ---------- PRACTICE ----------
if mode == "Practice":
    language = st.selectbox("Language", ["SQL", "Pandas", "NumPy", "Scikit-learn"])
    level_name = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])

    if (st.session_state.current_item is None or
            st.session_state.current_item["language"] != language or
            st.session_state.current_item["level"] != level_name):
        pick_new_question(language, level_name)

    current = st.session_state.current_item

    col_main, col_side = st.columns([2.5, 1.5])

    with col_main:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<span class="badge">{language} · {level_name}</span>', unsafe_allow_html=True)

        if current is None:
            st.warning("No items for this language/level. Add custom syntax from the right panel.")
        else:
            st.write("### 🧩 Task")
            st.write("Read the description and type the exact syntax (spelling matters).")
            st.markdown(f"**Hint / Description:** {current['description']}")
            st.caption("Usage example will appear only after a correct answer.")

            with st.form("practice_form"):
                answer = st.text_input("Type the exact syntax:", placeholder="e.g. PARTITION BY")
                submitted = st.form_submit_button("Check")

            if submitted:
                lvl = level_name
                st.session_state.attempts[lvl] += 1
                target = current["text"]

                if answer.strip() == target:
                    st.session_state.score[lvl] += 1
                    st.success("✅ Correct syntax!")
                    st.markdown("**Official syntax:**")
                    st.code(target, language="sql" if language == "SQL" else "python")
                    st.markdown("**Usage example:**")
                    st.code(current["usage"], language="sql" if language == "SQL" else "python")
                    save_progress(lvl)
                    pick_new_question(language, level_name)
                else:
                    st.error("❌ Incorrect syntax.")
                    st.write("Expected syntax pattern:")
                    st.code(target, language="sql" if language == "SQL" else "python")
                    close = difflib.get_close_matches(answer.strip(), [target], n=1)
                    if close:
                        st.caption(f"Closest match: `{close[0]}`")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ➕ Add your own syntax")
        st.caption("Any syntax you add is also stored in Google Sheets.")

        with st.form("add_form"):
            new_lang = st.selectbox("Language", ["SQL", "Pandas", "NumPy", "Scikit-learn"], key="new_lang")
            new_lvl = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"], key="new_lvl")
            new_text = st.text_input("Syntax text", placeholder="e.g. df.to_csv()")
            new_desc = st.text_area("Description", placeholder="What does this syntax do?")
            new_usage = st.text_area("Usage example", placeholder="Simple example using this syntax.")
            added = st.form_submit_button("Add")

        if added:
            if not new_text.strip():
                st.warning("Syntax text is required.")
            else:
                new_entry = {
                    "language": new_lang,
                    "level": new_lvl,
                    "text": new_text.strip(),
                    "description": new_desc.strip() or "User-defined syntax.",
                    "usage": new_usage.strip() or new_text.strip()
                }
                st.session_state.custom_items.append(new_entry)
                if st.session_state.user_id is not None:
                    save_custom_syntax(
                        st.session_state.user_id,
                        new_lang,
                        new_lvl,
                        new_text.strip(),
                        new_desc.strip() or "User-defined syntax.",
                        new_usage.strip() or new_text.strip()
                    )
                st.success("✅ Custom syntax added (session + Google Sheets).")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------- DOCUMENTATION ----------
elif mode == "Documentation":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="badge">Reference · Documentation</span>', unsafe_allow_html=True)
    st.write("Browse a structured mini-doc for each language and level.")

    docs_language = st.selectbox("Language", ["SQL", "Pandas", "NumPy", "Scikit-learn"], key="docs_lang")

    for lvl in ["Beginner", "Intermediate", "Advanced"]:
        items = get_items(docs_language, lvl)
        if not items:
            continue
        with st.expander(f"{docs_language} · {lvl}"):
            for it in items:
                st.markdown(f'<div class="doc-section-title">{it["text"]}</div>', unsafe_allow_html=True)
                st.markdown(f"- **Description:** {it['description']}")
                st.markdown("**Usage:**")
                st.code(it["usage"], language="sql" if docs_language == "SQL" else "python")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- AI AGENT ----------
elif mode == "AI Agent":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="badge">AI Agent · Syntax Coach</span>', unsafe_allow_html=True)
    st.write("Paste any line of SQL / Pandas / NumPy / Scikit-learn, and the agent will:")
    st.markdown("- Tell you if it is correct.")
    st.markdown("- Explain what is wrong (if any).")
    st.markdown("- Suggest a corrected version.")

    ai_lang = st.selectbox("Language", ["SQL", "Pandas", "NumPy", "Scikit-learn"], key="ai_lang")
    user_code = st.text_area("Your code / line:", height=160, placeholder="Write any statement here...")

    st.caption("Uses OpenAI API. Make sure OPENAI_API_KEY is set in Secrets.")

    if st.button("Ask AI Agent"):
        if not user_code.strip():
            st.warning("Please paste some code first.")
        else:
            client = get_openai_client()
            if client is None:
                st.error("OPENAI_API_KEY is not configured.")
            else:
                system_prompt = """
                You are a strict syntax tutor for SQL and Python (Pandas, NumPy, Scikit-learn).
                The user will send a single statement.
                Your job:
                1. Decide if the syntax is correct.
                2. If incorrect, explain exactly what is wrong (spelling, parentheses, wrong function, wrong structure, etc.).
                3. Propose a corrected version.
                4. Answer in JSON with keys: correct (bool), explanation (string), correction (string).
                """
                user_prompt = f"Language: {ai_lang}\nCode:\n{user_code}"

                try:
                    with st.spinner("Thinking..."):
                        resp = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ]
                        )
                        content = resp.choices[0].message.content
                        result = json.loads(content)

                    st.subheader("Result")
                    st.json(result)

                    if st.session_state.user_id is not None:
                        save_ai_log(
                            st.session_state.user_id,
                            user_code,
                            result.get("correct", False),
                            result.get("explanation", ""),
                            result.get("correction", "")
                        )

                except Exception as e:
                    st.error(f"Error from AI Agent: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- LEADERBOARD ----------
elif mode == "Leaderboard":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<span class="badge">Leaderboard · Top performers</span>', unsafe_allow_html=True)
    st.write("Global ranking of users based on their best recorded score (across all levels).")

    if sheets is None:
        st.warning("Google Sheets is not available.")
    else:
        df_lead = get_leaderboard(top_n=20)
        if df_lead is None or df_lead.empty:
            st.info("No progress data yet. Start practicing to appear on the leaderboard.")
        else:
            st.dataframe(df_lead, use_container_width=True)
            st.caption("Best Score = highest score reached in any session across all levels.")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# FOOTER
# =========================

st.markdown(
    """
    <div class="footer-3d">
        <span class="main">Made by Abdulrhman</span><br/>
        Syntax muscle memory · Data & AI craftsmanship · Built to be shared and improved.
    </div>
    """,
    unsafe_allow_html=True
)
