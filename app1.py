import streamlit as st
import html
import retrieval3 as retrieval
import sqlite3
import uuid
import json
import base64
from pathlib import Path
from datetime import datetime


# PAGE CONFIG

st.set_page_config(
    page_title="RAG CORE",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CHAT HISTORY DATABAS

DB_PATH = "chat_history.db"


# LOGO

LOGO_PATH = Path("assets/logo.png")


def get_logo_data_uri():
    """Load the RAG CORE logo as a base64 data URI for HTML rendering."""
    try:
        if LOGO_PATH.exists():
            encoded = base64.b64encode(
                LOGO_PATH.read_bytes()
            ).decode("utf-8")

            return f"data:image/png;base64,{encoded}"

    except Exception:
        pass

    return ""


LOGO_DATA_URI = get_logo_data_uri()


# DATABASE FUNCTIONS

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_chat_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def create_conversation(title="New conversation"):

    chat_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    conn = get_db()

    conn.execute("""
        INSERT INTO conversations
        (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    """, (
        chat_id,
        title,
        now,
        now
    ))

    conn.commit()
    conn.close()

    return chat_id


def update_conversation_title(chat_id, title):

    conn = get_db()

    conn.execute("""
        UPDATE conversations
        SET title = ?, updated_at = ?
        WHERE id = ?
    """, (
        title[:60],
        datetime.now().isoformat(),
        chat_id
    ))

    conn.commit()
    conn.close()


def save_message(
    chat_id,
    role,
    content,
    sources=None
):

    conn = get_db()

    conn.execute("""
        INSERT INTO messages
        (
            conversation_id,
            role,
            content,
            sources,
            created_at
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        chat_id,
        role,
        content,
        json.dumps(
            sources or [],
            ensure_ascii=False
        ),
        datetime.now().isoformat()
    ))

    conn.execute("""
        UPDATE conversations
        SET updated_at = ?
        WHERE id = ?
    """, (
        datetime.now().isoformat(),
        chat_id
    ))

    conn.commit()
    conn.close()


def load_conversations():

    conn = get_db()

    rows = conn.execute("""
        SELECT *
        FROM conversations
        ORDER BY updated_at DESC
    """).fetchall()

    conn.close()

    return rows


def load_messages(chat_id):

    conn = get_db()

    rows = conn.execute("""
        SELECT *
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
    """, (
        chat_id,
    )).fetchall()

    conn.close()

    messages = []

    for row in rows:

        sources = []

        if row["sources"]:

            try:

                sources = json.loads(
                    row["sources"]
                )

            except Exception:

                sources = []

        messages.append({
            "role": row["role"],
            "content": row["content"],
            "sources": sources
        })

    return messages


def delete_conversation(chat_id):

    conn = get_db()

    conn.execute("""
        DELETE FROM messages
        WHERE conversation_id = ?
    """, (
        chat_id,
    ))

    conn.execute("""
        DELETE FROM conversations
        WHERE id = ?
    """, (
        chat_id,
    ))

    conn.commit()
    conn.close()


init_chat_db()

# CUSTOM CSS

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');


html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stMain"] {

    font-family: 'Inter', sans-serif !important;
    background: #090b10 !important;
    color: #f5f7fa !important;
}


.stApp {

    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(80, 100, 255, 0.08),
            transparent 30%
        ),
        radial-gradient(
            circle at 85% 20%,
            rgba(0, 200, 180, 0.06),
            transparent 30%
        ),
        #090b10 !important;
}


.block-container {

    padding-top: 1.5rem !important;
    padding-bottom: 5rem !important;
    max-width: 1450px !important;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {

    background: #0d1017 !important;
    background-color: #0d1017 !important;
}


section[data-testid="stSidebar"] {

    border-right:
        1px solid rgba(255,255,255,0.06)
        !important;
}


section[data-testid="stSidebar"] > div {

    padding-top: 1.5rem !important;
}


section[data-testid="stSidebar"] * {

    color: #dfe3ea;
}


/* =========================================================
   LOGO
   ========================================================= */

.logo-container {

    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 5px;
}


.logo-icon {

    width: 48px;
    height: 48px;
    border-radius: 12px;

    display: flex;
    align-items: center;
    justify-content: center;

    overflow: hidden;
    flex-shrink: 0;

    background: transparent;
}


.logo-icon img {

    width: 48px;
    height: 48px;

    object-fit: contain;
    display: block;
}


.logo-text {

    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.5px;

    color: #f5f7fa !important;
}


.logo-sub {

    color: #7d8492 !important;

    font-size: 11px;

    margin-left: 54px;
    margin-top: -5px;
}


/* =========================================================
   HERO
   ========================================================= */

.hero {

    text-align: center;

    padding: 65px 20px 35px 20px;
}


.hero-icon {

    width: 86px;
    height: 86px;

    margin: auto;

    border-radius: 22px;

    display: flex;
    align-items: center;
    justify-content: center;

    overflow: hidden;

    background: transparent;
}


.hero-icon img {

    width: 86px;
    height: 86px;

    object-fit: contain;

    display: block;
}


.hero-title {

    font-size: 40px;

    font-weight: 800;

    margin-top: 18px;

    letter-spacing: -1.5px;

    color: #f5f7fa !important;
}


.hero-subtitle {

    color: #8c93a1 !important;

    font-size: 15px;

    margin-top: 8px;
}


/* =========================================================
   STATUS
   ========================================================= */

.status {

    display: inline-flex;

    align-items: center;

    gap: 7px;

    background:
        rgba(65, 214, 151, 0.08)
        !important;

    border:
        1px solid
        rgba(65, 214, 151, 0.15)
        !important;

    color: #62dca7 !important;

    border-radius: 20px;

    padding: 6px 11px;

    font-size: 11px;
}


.status-dot {

    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: #58d99e !important;
}


/* =========================================================
   INFO CARDS
   ========================================================= */

.info-card {

    background:
        rgba(17, 21, 30, 0.78)
        !important;

    border:
        1px solid
        rgba(255,255,255,0.065)
        !important;

    border-radius: 17px;

    padding: 18px;

    min-height: 100px;
}


.info-label {

    color: #777f8d !important;

    font-size: 11px;

    text-transform: uppercase;

    letter-spacing: 0.8px;
}


.info-value {

    font-size: 21px;

    font-weight: 700;

    margin-top: 6px;

    color: #f5f7fa !important;
}


.info-small {

    color: #656d7a !important;

    font-size: 11px;

    margin-top: 4px;
}


/* =========================================================
   FEATURE CARDS
   ========================================================= */

.feature-card {

    background:
        rgba(17, 21, 30, 0.65)
        !important;

    border:
        1px solid
        rgba(255,255,255,0.055)
        !important;

    border-radius: 16px;

    padding: 20px;

    min-height: 135px;
}


.feature-icon {

    font-size: 22px;

    margin-bottom: 10px;
}


.feature-title {

    font-weight: 700;

    font-size: 14px;

    color: #f5f7fa !important;
}


.feature-desc {

    color: #777f8d !important;

    font-size: 12px;

    line-height: 1.6;

    margin-top: 5px;
}


/* =========================================================
   CHAT
   ========================================================= */

.chat-user-wrapper {

    display: flex;

    justify-content: flex-end;

    margin: 20px 0;
}


.chat-user {

    max-width: 75%;

    background: #1d2330 !important;

    border:
        1px solid
        rgba(255,255,255,0.06)
        !important;

    border-radius:
        18px 18px 5px 18px;

    padding: 14px 18px;

    line-height: 1.7;

    color: #f5f7fa !important;
}


.chat-ai-wrapper {

    display: flex;

    justify-content: flex-start;

    margin: 20px 0;
}


.chat-ai {

    max-width: 85%;

    background:
        rgba(16,20,28,0.9)
        !important;

    border:
        1px solid
        rgba(255,255,255,0.065)
        !important;

    border-radius:
        18px 18px 18px 5px;

    padding: 18px 20px;

    line-height: 1.8;

    color: #e7eaf0 !important;
}


.ai-label {

    color: #8793ff !important;

    font-size: 11px;

    font-weight: 700;

    margin-bottom: 8px;

    text-transform: uppercase;

    letter-spacing: 0.7px;
}


/* =========================================================
   SOURCE CARD
   ========================================================= */

.source-card {

    background: #11151d !important;

    border:
        1px solid
        rgba(255,255,255,0.055)
        !important;

    border-radius: 14px;

    padding: 15px;

    margin: 8px 0;
}


.source-header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 15px;
}


.source-name {

    font-weight: 600;

    font-size: 13px;

    color: #e8ebf0 !important;
}


.source-meta {

    color: #727987 !important;

    font-size: 11px;

    margin-top: 5px;
}


.similarity {

    background:
        rgba(72,196,139,0.12)
        !important;

    color: #63dca4 !important;

    border:
        1px solid
        rgba(72,196,139,0.18)
        !important;

    padding: 5px 9px;

    border-radius: 8px;

    font-size: 11px;

    font-weight: 700;

    white-space: nowrap;
}


/* =========================================================
   SECTION TITLE
   ========================================================= */

.section-title {

    font-size: 12px;

    color: #777f8d !important;

    text-transform: uppercase;

    letter-spacing: 1px;

    margin-bottom: 12px;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {

    width: 100%;

    border-radius: 10px !important;

    border:
        1px solid
        rgba(255,255,255,0.07)
        !important;

    background: #151a23 !important;

    background-color: #151a23 !important;

    color: #dfe3ea !important;

    transition: all 0.2s ease;
}


.stButton > button:hover {

    border-color:
        rgba(120,130,255,0.45)
        !important;

    color: white !important;

    background: #181e28 !important;
}


/* =========================================================
   SLIDERS
   ========================================================= */

section[data-testid="stSidebar"]
[data-testid="stSlider"] {

    color: #dfe3ea !important;
}


section[data-testid="stSidebar"]
[data-testid="stSlider"] label {

    color: #aeb5c2 !important;
}


/* =========================================================
   LANGUAGE SELECTBOX
   ========================================================= */

section[data-testid="stSidebar"]
[data-testid="stSelectbox"] {

    background: transparent !important;

    background-color: transparent !important;

    color: #dfe3ea !important;
}


section[data-testid="stSidebar"]
[data-testid="stSelectbox"] label,

section[data-testid="stSidebar"]
[data-testid="stSelectbox"] label p {

    color: #aeb5c2 !important;
}


section[data-testid="stSidebar"]
[data-testid="stSelectbox"]
[data-baseweb="select"],

section[data-testid="stSidebar"]
[data-testid="stSelectbox"]
[data-baseweb="select"] > div {

    background: #151a23 !important;

    background-color: #151a23 !important;
}


section[data-testid="stSidebar"]
[data-testid="stSelectbox"]
[role="combobox"] {

    background: #151a23 !important;

    background-color: #151a23 !important;

    color: #dfe3ea !important;
}


section[data-testid="stSidebar"]
[data-testid="stSelectbox"]
[data-baseweb="select"] span {

    color: #dfe3ea !important;
}


section[data-testid="stSidebar"]
[data-testid="stSelectbox"]
[data-baseweb="select"] > div {

    border:
        1px solid
        rgba(255,255,255,0.08)
        !important;

    border-radius: 10px !important;

    box-shadow: none !important;
}


section[data-testid="stSidebar"]
[data-testid="stSelectbox"] svg {

    fill: #8d95a5 !important;

    color: #8d95a5 !important;
}


/* =========================================================
   SELECTBOX DROPDOWN
   ========================================================= */

[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div {

    background: #11151d !important;

    background-color: #11151d !important;

    border-color:
        rgba(255,255,255,0.08)
        !important;
}


[data-baseweb="menu"] {

    background: #11151d !important;

    background-color: #11151d !important;

    border:
        1px solid
        rgba(255,255,255,0.06)
        !important;
}


[data-baseweb="menu"] li {

    background: #11151d !important;

    background-color: #11151d !important;

    color: #dfe3ea !important;
}


[data-baseweb="menu"] li span,
[data-baseweb="menu"] li div {

    color: #dfe3ea !important;
}


[data-baseweb="menu"] li:hover {

    background: #1b2130 !important;

    background-color: #1b2130 !important;

    color: #ffffff !important;
}


[data-baseweb="menu"]
li[aria-selected="true"] {

    background: #1b2130 !important;

    color: #ffffff !important;
}


/* =========================================================
   CHAT HISTORY
   ========================================================= */

.chat-history-button button {

    text-align: left !important;
}


/* =========================================================
   CODE BLOCKS
   ========================================================= */

[data-testid="stCodeBlock"] {

    background: #11151d !important;

    background-color: #11151d !important;

    border:
        1px solid
        rgba(255,255,255,0.06)
        !important;

    border-radius: 10px !important;
}


[data-testid="stCodeBlock"] pre {

    background: #11151d !important;

    background-color: #11151d !important;

    color: #b9c0ff !important;
}


[data-testid="stCodeBlock"] code {

    background: transparent !important;

    color: #b9c0ff !important;
}


section[data-testid="stSidebar"] pre {

    background: #11151d !important;

    background-color: #11151d !important;

    color: #b9c0ff !important;

    border:
        1px solid
        rgba(255,255,255,0.06)
        !important;

    border-radius: 10px !important;
}


section[data-testid="stSidebar"] code {

    color: #b9c0ff !important;

    background: transparent !important;
}


/* =========================================================
   CHAT INPUT
   ========================================================= */

[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottomBlockContainer"] {

    background: #090b10 !important;

    background-color: #090b10 !important;

    border: none !important;

    box-shadow: none !important;
}


[data-testid="stBottom"] {

    border-top: none !important;
}


[data-testid="stBottom"] * {

    --background-color: #090b10 !important;
}


[data-testid="stChatInput"],
[data-testid="stChatInputContainer"] {

    background: transparent !important;

    background-color: transparent !important;

    border: none !important;

    box-shadow: none !important;
}


[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] form > div {

    background: #11151d !important;

    background-color: #11151d !important;

    border-radius: 16px !important;
}


[data-testid="stChatInput"] > div {

    border:
        1px solid
        rgba(255,255,255,0.08)
        !important;

    box-shadow:
        0 10px 35px
        rgba(0,0,0,0.20)
        !important;
}


[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {

    background: transparent !important;

    background-color: transparent !important;

    color: #f5f7fa !important;

    caret-color: #8b7cff !important;

    border: none !important;

    outline: none !important;

    box-shadow: none !important;
}


[data-testid="stChatInput"]
textarea::placeholder {

    color: #68717f !important;

    opacity: 1 !important;
}


[data-testid="stChatInput"]
[data-baseweb="textarea"],

[data-testid="stChatInput"]
[data-baseweb="textarea"] > div {

    background: transparent !important;

    background-color: transparent !important;

    border: none !important;
}


[data-testid="stChatInput"] button {

    background: #6875ff !important;

    background-color: #6875ff !important;

    color: white !important;

    border: none !important;

    border-radius: 10px !important;
}


[data-testid="stChatInput"] button:hover {

    background: #7b86ff !important;

    background-color: #7b86ff !important;
}


/* =========================================================
   EXPANDER
   ========================================================= */

[data-testid="stExpander"],
[data-testid="stExpander"] > details,
[data-testid="stExpander"] details {

    background: #10141c !important;

    background-color: #10141c !important;

    border:
        1px solid
        rgba(255,255,255,0.055)
        !important;

    border-radius: 14px !important;
}


[data-testid="stExpander"] summary {

    color: #dfe3ea !important;

    background: transparent !important;
}


/* =========================================================
   STREAMLIT ELEMENT BACKGROUNDS
   ========================================================= */

[data-testid="stVerticalBlock"] {

    background: transparent !important;
}


[data-testid="stHorizontalBlock"] {

    background: transparent !important;
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {

    border-color:
        rgba(255,255,255,0.05)
        !important;
}


/* =========================================================
   HEADER / FOOTER
   ========================================================= */

header {

    background: transparent !important;
}


#MainMenu {

    visibility: hidden;
}


footer {

    visibility: hidden;
}


/* =========================================================
   DARK MODE
   ========================================================= */

section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea,
section[data-testid="stSidebar"] [data-baseweb],
section[data-testid="stSidebar"] [data-baseweb] > div {

    color-scheme: dark !important;
}


/* =========================================================
   REMOVE WHITE SURFACES
   ========================================================= */

section[data-testid="stSidebar"]
div[data-testid="stVerticalBlockBorderWrapper"],

section[data-testid="stSidebar"]
div[data-testid="stElementContainer"] {

    background: transparent !important;
}

</style>
""")


# SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []


if "retrieval_stats" not in st.session_state:

    st.session_state.retrieval_stats = {

        "retrieved": 0,
        "relevant": 0,
        "best_similarity": 0.0
    }


if "current_chat_id" not in st.session_state:

    st.session_state.current_chat_id = (
        create_conversation()
    )


if "chat_title_set" not in st.session_state:

    st.session_state.chat_title_set = False


# SIDEBAR

with st.sidebar:

    # LOGO

    if LOGO_DATA_URI:

        sidebar_logo = f"""
        <div class="logo-container">

            <div class="logo-icon">

                <img
                    src="{LOGO_DATA_URI}"
                    alt="RAG CORE logo"
                >

            </div>

            <div class="logo-text">

                RAG CORE

            </div>

        </div>

        <div class="logo-sub">

            Cardiovascular AI Assistant

        </div>
        """

    else:

        sidebar_logo = """
        <div class="logo-container">

            <div class="logo-icon">

                ✦

            </div>

            <div class="logo-text">

                RAG CORE

            </div>

        </div>

        <div class="logo-sub">

            Cardiovascular AI Assistant

        </div>
        """

    st.html(sidebar_logo)


    st.write("")


    # NEW CONVERSATION

    if st.button(
        "＋  New conversation",
        use_container_width=True
    ):

        new_chat_id = create_conversation()

        st.session_state.current_chat_id = (
            new_chat_id
        )

        st.session_state.messages = []

        st.session_state.chat_title_set = False

        st.session_state.retrieval_stats = {

            "retrieved": 0,
            "relevant": 0,
            "best_similarity": 0.0
        }

        st.rerun()


    st.markdown("---")


    # SYSTEM STATUS

    st.html("""
    <div class="info-label">
        SYSTEM STATUS
    </div>

    <div style="margin-top:8px;">

        <span class="status">

            <span class="status-dot"></span>

            RAG system online

        </span>

    </div>
    """)


    st.write("")


    # RAG CONFIGURATION

    st.html("""
    <div class="info-label">
        RAG CONFIGURATION
    </div>
    """)


    top_k = st.slider(

        "Top K",

        min_value=1,

        max_value=10,

        value=5,

        help="Number of chunks retrieved from ChromaDB."
    )


    threshold = st.slider(

        "Relevance threshold",

        min_value=0,

        max_value=100,

        value=70,

        help="Minimum similarity required for the final chunk."
    )


    st.markdown("---")


    # MODELS

    st.html("""
    <div class="info-label">
        MODELS
    </div>
    """)


    st.caption("Embedding")


    st.code(
        "BAAI/bge-m3",
        language=None
    )


    st.caption("LLM")


    st.code(
        "qwen/qwen3.6-27b",
        language=None
    )


    st.markdown("---")


    # LANGUAGE
    

    st.html("""
    <div
        class="info-label"
        style="margin-bottom:8px;"
    >

        LANGUAGE

    </div>
    """)


    language = st.selectbox(

        "Language",

        [

            "Auto Detect",
            "English",
            "العربية"

        ],

        key="language_selector",

        label_visibility="collapsed"
    )


    st.write("")


    
    # CHAT HISTORY
   

    st.markdown("---")


    st.html("""
    <div
        class="info-label"
        style="margin-bottom:10px;"
    >

        CHAT HISTORY

    </div>
    """)


    conversations = load_conversations()


    if conversations:

        for chat in conversations:

            title = chat["title"]

            if len(title) > 32:

                title = title[:32] + "..."


            is_current = (

                chat["id"]
                ==
                st.session_state.current_chat_id

            )


            button_label = (

                "●  "
                if is_current
                else
                "○  "

            ) + title


            if st.button(

                button_label,

                key=f"chat_history_{chat['id']}",

                use_container_width=True

            ):

                st.session_state.current_chat_id = (
                    chat["id"]
                )

                st.session_state.messages = (
                    load_messages(
                        chat["id"]
                    )
                )

                st.session_state.chat_title_set = True

                st.session_state.retrieval_stats = {

                    "retrieved": 0,
                    "relevant": 0,
                    "best_similarity": 0.0
                }

                st.rerun()

    else:

        st.caption(
            "No saved conversations yet."
        )


    st.write("")


    
    # CLEAR CHAT
    

    if st.button(
        "🗑 Clear chat",
        use_container_width=True
    ):

        delete_conversation(
            st.session_state.current_chat_id
        )

        new_chat_id = create_conversation()

        st.session_state.current_chat_id = (
            new_chat_id
        )

        st.session_state.messages = []

        st.session_state.chat_title_set = False

        st.session_state.retrieval_stats = {

            "retrieved": 0,
            "relevant": 0,
            "best_similarity": 0.0
        }

        st.rerun()


# APPLY UI SETTINGS TO RETRIEVAL

retrieval.TOP_K = top_k

retrieval.RELEVANCE_THRESHOLD = float(
    threshold
)

# MAIN HEADER


st.html("""
<div style="
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:10px;
">

    <div>

        <div style="
            color:#7e87ff;
            font-size:11px;
            font-weight:700;
            letter-spacing:1px;
            text-transform:uppercase;
        ">

            AI KNOWLEDGE SYSTEM

        </div>


        <div style="
            font-size:28px;
            font-weight:800;
            margin-top:4px;
            color:#f5f7fa;
        ">

            Cardiovascular Intelligence

        </div>

    </div>


    <div class="status">

        <span class="status-dot"></span>

        System Ready

    </div>

</div>
""")


# EMPTY STATE


if not st.session_state.messages:

    # CENTER LOGO
   

    if LOGO_DATA_URI:

        hero_logo = f"""
        <img
            src="{LOGO_DATA_URI}"
            alt="RAG CORE logo"
        >
        """

    else:

        hero_logo = "✦"


    st.html(f"""
    <div class="hero">

        <div class="hero-icon">

            {hero_logo}

        </div>


        <div class="hero-title">

            Ask your medical knowledge assistant

        </div>


        <div class="hero-subtitle">

            Retrieval-Augmented Generation powered by your
            cardiovascular knowledge base.

        </div>

    </div>
    """)


    col1, col2, col3 = st.columns(3)


    with col1:

        st.html("""
        <div class="feature-card">

            <div class="feature-icon">
                ⌕
            </div>


            <div class="feature-title">

                Semantic Retrieval

            </div>


            <div class="feature-desc">

                Finds the most relevant knowledge chunks
                from your medical documents.

            </div>

        </div>
        """)


    with col2:

        st.html("""
        <div class="feature-card">

            <div class="feature-icon">
                ✦
            </div>


            <div class="feature-title">

                Grounded Answers

            </div>


            <div class="feature-desc">

                Generates answers using retrieved evidence
                instead of relying only on model memory.

            </div>

        </div>
        """)


    with col3:

        st.html("""
        <div class="feature-card">

            <div class="feature-icon">
                ◈
            </div>


            <div class="feature-title">

                Source Transparency

            </div>


            <div class="feature-desc">

                Shows documents, pages, similarity scores
                and retrieved evidence.

            </div>

        </div>
        """)


    st.write("")
    st.write("")


    st.html("""
    <div style="
        text-align:center;
        color:#5f6673;
        font-size:12px;
        margin-bottom:10px;
    ">

        Try asking:

    </div>
    """)


    q1, q2, q3 = st.columns(3)


    with q1:

        if st.button(
            "What are the risk factors for cardiovascular disease?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What are the risk factors for cardiovascular disease?"
            )

            st.rerun()


    with q2:

        if st.button(
            "What is ischaemic heart disease?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What is ischaemic heart disease?"
            )

            st.rerun()


    with q3:

        if st.button(
            "What causes heart failure?",
            use_container_width=True
        ):

            st.session_state.pending_question = (
                "What causes heart failure?"
            )

            st.rerun()


# DISPLAY CHAT HISTORY

for message in st.session_state.messages:

    if message["role"] == "user":

        user_text = html.escape(
            str(
                message["content"]
            )
        )


        st.html(f"""
        <div class="chat-user-wrapper">

            <div class="chat-user">

                {user_text}

            </div>

        </div>
        """)


    else:

        assistant_text = str(
            message.get(
                "content",
                ""
            )
        )


        assistant_text = html.escape(
            assistant_text
        )


        assistant_text = assistant_text.replace(
            "\n",
            "<br>"
        )


        st.html(f"""
        <div class="chat-ai-wrapper">

            <div class="chat-ai">

                <div class="ai-label">

                    ✦ RAG CORE

                </div>

                {assistant_text}

            </div>

        </div>
        """)


        sources = message.get(
            "sources",
            []
        )


        if sources:

            with st.expander(

                f"◈ Evidence · {len(sources)} source"
                + (
                    "s"
                    if len(sources) != 1
                    else
                    ""
                )

            ):

                for source in sources:

                    source_file = html.escape(
                        str(
                            source.get(
                                "file",
                                "Unknown"
                            )
                        )
                    )


                    source_section = html.escape(
                        str(
                            source.get(
                                "section",
                                "Unknown section"
                            )
                        )
                    )


                    source_page = source.get(
                        "page",
                        "?"
                    )


                    source_chunk_id = html.escape(
                        str(
                            source.get(
                                "chunk_id",
                                "Unknown"
                            )
                        )
                    )


                    similarity = float(
                        source.get(
                            "similarity",
                            0
                        )
                    )


                    source_text = source.get(
                        "text",
                        ""
                    )


                    st.html(f"""
                    <div class="source-card">

                        <div class="source-header">

                            <div>

                                <div class="source-name">

                                    📄 {source_file}

                                </div>


                                <div class="source-meta">

                                    Page {source_page}

                                    &nbsp;•&nbsp;

                                    {source_section}

                                </div>

                            </div>


                            <div class="similarity">

                                {similarity:.2f}%

                            </div>

                        </div>


                        <div
                            class="source-meta"
                            style="margin-top:10px;"
                        >

                            Chunk ID: {source_chunk_id}

                        </div>

                    </div>
                    """)


                    st.caption(
                        source_text
                    )


# CHAT INPUT


prompt = st.chat_input(
    "Ask anything about cardiovascular diseases...",
    key="main_chat_input"
)


# EXAMPLE QUESTION HANDLER


if (
    not prompt
    and
    "pending_question" in st.session_state
):

    prompt = st.session_state.pending_question

    del st.session_state.pending_question



# HANDLE QUESTION


if prompt:

    st.session_state.messages.append({

        "role": "user",

        "content": prompt

    })


    save_message(

        st.session_state.current_chat_id,

        "user",

        prompt

    )


    if not st.session_state.chat_title_set:

        title = prompt.strip()

        if len(title) > 45:

            title = title[:45] + "..."


        update_conversation_title(

            st.session_state.current_chat_id,

            title

        )


        st.session_state.chat_title_set = True


    try:

        with st.spinner(
            "Searching medical knowledge base..."
        ):

            if language == "English":

                user_lang = "en"

            elif language == "العربية":

                user_lang = "ar"

            else:

                user_lang = retrieval.detect_language(
                    prompt
                )


            search_query = (
                retrieval.prepare_search_query(
                    prompt,
                    user_lang
                )
            )


            retrieved_chunks = (
                retrieval.retrieve_chunks(
                    search_query
                )
            )


            relevant_chunks = (
                retrieval.select_best_chunk(
                    retrieved_chunks
                )
            )


            answer = retrieval.generate_answer(

                prompt,

                relevant_chunks,

                user_lang

            )


        sources = []


        for chunk in relevant_chunks:

            metadata = chunk.get(
                "metadata",
                {}
            )


            sources.append({

                "file":
                    retrieval.get_document_name(
                        metadata
                    ),

                "page":
                    retrieval.get_page(
                        metadata
                    ),

                "section":
                    retrieval.get_section(
                        metadata
                    ),

                "chunk_id":
                    retrieval.get_chunk_id(
                        metadata
                    ),

                "similarity":
                    float(
                        chunk.get(
                            "similarity",
                            0
                        )
                    ),

                "text":
                    chunk.get(
                        "text",
                        ""
                    )

            })


        best_similarity = 0.0


        if retrieved_chunks:

            best_similarity = max(

                float(
                    chunk.get(
                        "similarity",
                        0
                    )
                )

                for chunk in retrieved_chunks

            )


        st.session_state.retrieval_stats = {

            "retrieved":
                len(retrieved_chunks),

            "relevant":
                len(relevant_chunks),

            "best_similarity":
                best_similarity

        }


        st.session_state.messages.append({

            "role":
                "assistant",

            "content":
                answer,

            "sources":
                sources

        })


        save_message(

            st.session_state.current_chat_id,

            "assistant",

            answer,

            sources

        )


    except Exception as e:

        error_message = (

            "❌ Retrieval system error:\n\n"

            f"{str(e)}"

        )


        st.session_state.messages.append({

            "role":
                "assistant",

            "content":
                error_message,

            "sources":
                []

        })


        save_message(

            st.session_state.current_chat_id,

            "assistant",

            error_message,

            []

        )


        st.session_state.retrieval_stats = {

            "retrieved":
                0,

            "relevant":
                0,

            "best_similarity":
                0.0

        }


    st.rerun()


# RETRIEVAL STATISTICS


if st.session_state.messages:

    st.markdown("---")


    st.html("""
    <div class="section-title">

        Retrieval Statistics

    </div>
    """)


    stats = st.session_state.retrieval_stats


    c1, c2, c3 = st.columns(3)


    with c1:

        st.html(f"""
        <div class="info-card">

            <div class="info-label">

                Retrieved chunks

            </div>


            <div class="info-value">

                {stats["retrieved"]}

            </div>


            <div class="info-small">

                Top-K semantic retrieval

            </div>

        </div>
        """)


    with c2:

        st.html(f"""
        <div class="info-card">

            <div class="info-label">

                Relevant chunks

            </div>


            <div class="info-value">

                {stats["relevant"]}

            </div>


            <div class="info-small">

                Above relevance threshold

            </div>

        </div>
        """)


    with c3:

        st.html(f"""
        <div class="info-card">

            <div class="info-label">

                Best similarity

            </div>


            <div class="info-value">

                {stats["best_similarity"]:.2f}%

            </div>


            <div class="info-small">

                Highest retrieved similarity

            </div>

        </div>
        """)