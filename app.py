import streamlit as st
import pandas as pd
import gspread

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Syntax Trainer Pro",
    page_icon="⚡",
    layout="wide"
)

st.markdown("<h1 style='text-align:center;'>⚡ Syntax Trainer Pro</h1>", unsafe_allow_html=True)

# -----------------------------------
# LOAD PUBLIC GOOGLE SHEET
# -----------------------------------
SHEET_URL = "https://docs.google.com/spreadsheets/d/1Gly5KDsBf7jjB-x5fwTK3JLOLgggrtWpPM8bYbLqmRk/edit?usp=sharing"

def load_public_sheet(sheet_url, sheet_name):
    try:
        gc = gspread.Client(None)  
        sh = gc.open_by_url(sheet_url)
        ws = sh.worksheet(sheet_name)
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Google Sheet Error in {sheet_name}: {e}")
        return pd.DataFrame()

# LOAD ALL TABS
syntax_df       = load_public_sheet(SHEET_URL, "Syntax_Practice")
questions_df    = load_public_sheet(SHEET_URL, "Questions_Practice")
docs_df         = load_public_sheet(SHEET_URL, "Documentation")
users_df        = load_public_sheet(SHEET_URL, "Users")
progress_df     = load_public_sheet(SHEET_URL, "Progress")

st.success("Sheets Loaded Successfully (Public Read-Only Mode)")

# -----------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Syntax Practice", "Questions Practice", "Documentation"]
)

# -----------------------------------
# PAGE 1 — SYNTAX PRACTICE MODE
# -----------------------------------
if page == "Syntax Practice":
    st.header("Syntax Practice Trainer")

    if syntax_df.empty:
        st.error("Syntax_Practice sheet is empty.")
    else:
        languages = syntax_df["language"].unique()
        levels    = syntax_df["level"].unique()

        col1, col2 = st.columns(2)
        with col1:
            selected_lang = st.selectbox("Language", languages)
        with col2:
            selected_level = st.selectbox("Level", levels)

        filtered = syntax_df[
            (syntax_df["language"] == selected_lang) &
            (syntax_df["level"] == selected_level)
        ]

        if len(filtered) == 0:
            st.warning("No tasks available for this level/language.")
        else:
            task = filtered.sample(1).iloc[0]

            st.subheader(f"Category: {task['category']}")
            st.info(task['description_en'])
            st.write("**Arabic Explanation:**")
            st.write(task["description_ar"])

            user_answer = st.text_input("Write the exact syntax:")

            if st.button("Check"):
                if user_answer.strip().lower() == task["syntax"].strip().lower():
                    st.success("Correct ✔")
                    st.code(task["usage_example"])
                else:
                    st.error("Incorrect ❌ — Try again.")
                    st.write("Expected syntax:")
                    st.code(task["syntax"])

# -----------------------------------
# PAGE 2 — QUESTIONS PRACTICE MODE
# -----------------------------------
elif page == "Questions Practice":
    st.header("Questions Practice Mode")

    if questions_df.empty:
        st.warning("Questions_Practice sheet is empty.")
    else:
        langs = questions_df["language"].unique()
        lvls = questions_df["level"].unique()

        col1, col2 = st.columns(2)
        with col1:
            lang = st.selectbox("Language", langs)
        with col2:
            lvl = st.selectbox("Level", lvls)

        qset = questions_df[
            (questions_df["language"] == lang) &
            (questions_df["level"] == lvl)
        ]

        if len(qset) == 0:
            st.warning("No questions for this selection.")
        else:
            q = qset.sample(1).iloc[0]

            st.subheader("📘 Question")
            st.write(q["question_en"])
            st.write("**Arabic:**")
            st.write(q["question_ar"])

            st.write("### Dataset Preview")
            st.code(q["dataset_preview"])

            ans = st.text_area("Your Answer:")

            if st.button("Submit Answer"):
                if ans.strip().lower() == q["correct_answer"].strip().lower():
                    st.success("Correct! ✔")
                    st.write("### Explanation")
                    st.info(q["explanation_en"])
                else:
                    st.error("Incorrect ❌")
                    st.write("Correct answer:")
                    st.code(q["correct_answer"])
                    st.info(q["explanation_en"])

# -----------------------------------
# PAGE 3 — DOCUMENTATION
# -----------------------------------
elif page == "Documentation":
    st.header("📚 Language Documentation")

    if docs_df.empty:
        st.warning("Documentation sheet is empty.")
    else:
        langs = docs_df["language"].unique()
        catgs = docs_df["category"].unique()

        col1, col2 = st.columns(2)
        with col1:
            lang = st.selectbox("Language", langs)
        with col2:
            cat = st.selectbox("Category", catgs)

        selected = docs_df[
            (docs_df["language"] == lang) &
            (docs_df["category"] == cat)
        ]

        if len(selected) == 0:
            st.warning("No documentation found.")
        else:
            for _, row in selected.iterrows():
                st.subheader(row["title"])
                st.write("### English")
                st.info(row["description_en"])
                st.write("### Arabic")
                st.info(row["description_ar"])
                st.write("### Syntax")
                st.code(row["syntax"])
                st.write("### Examples")
                st.code(row["examples"])
                st.write("---")

# -----------------------------------
# FOOTER
# -----------------------------------
st.markdown(
    "<br><center><h4 style='opacity:0.6'>Made by Abdulrhman ⚡</h4></center>",
    unsafe_allow_html=True
)
