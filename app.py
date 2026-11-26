import pickle
from html import escape
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OmniSphere | Premium Discovery",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- GLOBAL DARK THEME & STYLING ---
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #05070a;
            --card: #0f121a;
            --text: #ffffff;
            --accent: #d4af37;
            --sub-text: #94a3b8;
            --border: #1e293b;
        }
        .stApp {
            background-color: #05070a;
            color: #ffffff;
            font-family: 'Outfit', sans-serif;
        }
        .block-container {
            max-width: 1100px;
            padding-top: 60px;
            padding-bottom: 60px;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .omni-header {
            margin-bottom: 40px;
            text-align: center;
        }
        .omni-main-title {
            font-size: 3.5rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -1.5px;
            color: #ffffff;
            font-family: 'Outfit', sans-serif;
        }
        .omni-main-title em {
            font-style: normal;
            color: #d4af37;
            font-weight: 300;
        }
        .omni-subtitle {
            color: #94a3b8;
            letter-spacing: 2px;
            text-transform: uppercase;
            font-size: 0.7rem;
            margin-top: 6px;
            font-family: 'Outfit', sans-serif;
        }
        .field-label {
            font-size: 0.65rem;
            color: #d4af37;
            font-weight: 700;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 6px;
            font-family: 'Outfit', sans-serif;
        }
        div[data-testid="stSelectbox"] > div > div {
            background-color: #05070a !important;
            color: #ffffff !important;
            border: 1px solid #1e293b !important;
            border-radius: 14px !important;
            min-height: 50px;
            font-family: 'Outfit', sans-serif !important;
        }
        .st-key-movie_select div[data-testid="stSelectbox"] > div > div {
            background-color: #0f121a !important;
            min-height: 60px;
            font-size: 1.1rem;
        }
        div[data-testid="stSelectbox"] input,
        div[data-testid="stSelectbox"] div[value] {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        div[data-testid="stSelectbox"] input::placeholder {
            color: #94a3b8 !important;
            -webkit-text-fill-color: #94a3b8 !important;
        }
        div[data-testid="stSelectbox"] svg {
            color: #94a3b8 !important;
        }
        .omni-header h1.omni-main-title {
            color: #ffffff !important;
        }
        div[data-testid="stButton"] > button {
            width: 100%;
            background-color: #d4af37 !important;
            color: #000000 !important;
            border: none !important;
            height: 54px !important;
            border-radius: 14px !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px;
            font-family: 'Outfit', sans-serif !important;
        }
        div[data-testid="stButton"] > button:hover {
            background-color: #e5c14c !important;
            color: #000000 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine(path="ott_recommendation_engine.pkl"):
    with open(path,"rb") as f:
        engine=pickle.load(f)
    return engine["data"],engine["matrix"]


@st.cache_data
def get_labels(_df):
    #most voted titles come first in the dropdown
    order=_df["imdb_votes"].fillna(0).sort_values(ascending=False).index
    labels={}
    for i in order:
        labels[i]=f"{_df['title'][i]} ({_df['release_year'][i]}) • {_df['type'][i].title()}"
    return labels


def recommend(df,matrix,idx,platform,content_type,top_n=6):
    data=df.copy()
    data["similarity"]=(matrix@matrix[idx].T).toarray().ravel()
    data=data.drop(idx)

    if platform!="all":
        data=data[data["available_on"].str.lower().str.contains(platform.lower(),regex=False)]
    if content_type!="both":
        data=data[data["type"].str.lower()==content_type]

    data=data[data["similarity"]>0].nlargest(50,"similarity")
    data["final_score"]=0.8*data["similarity"]+0.2*data["rating"]/10
    return data.nlargest(top_n,"final_score")


def render_results_grid(items):
    cards_html=[]
    for _,m in items.iterrows():
        score="N/A" if pd.isna(m["imdb_score"]) else m["imdb_score"]
        kind="TV Show" if m["type"]=="SHOW" else "Movie"
        cards_html.append(
            f"""
            <div class="movie-card">
                <div class="card-header">
                    <span class="imdb-badge">IMDb {score}</span>
                    <h2 class="title">{escape(m["title"])}</h2>
                    <p class="meta-line">{m["release_year"]} • {kind}</p>
                    <p class="ott-line">{escape(m["available_on"])}</p>
                </div>
                <div class="summary-body">
                    <div class="summary-text">
                        <strong class="summary-heading">SUMMARY</strong>
                        {escape(m["description"])}
                    </div>
                </div>
            </div>
            """
        )

    grid_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: #05070a;
                --card: #0f121a;
                --text: #ffffff;
                --accent: #d4af37;
                --sub-text: #94a3b8;
                --border: #1e293b;
                --scroll-thumb: #1e293b;
            }}
            body {{
                background-color: transparent;
                color: var(--text);
                font-family: 'Outfit', sans-serif;
                margin: 0;
                padding: 8px 0 24px 0;
            }}
            .results-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                gap: 25px;
            }}
            .movie-card {{
                background: var(--card);
                border-radius: 24px;
                border: 1px solid var(--border);
                height: 200px;
                position: relative;
                overflow: hidden;
                transition: all 0.4s cubic-bezier(0.25, 1, 0.5, 1);
                cursor: pointer;
                display: flex;
                flex-direction: column;
                box-sizing: border-box;
            }}
            .movie-card:hover {{
                height: 400px;
                border-color: var(--accent);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
            }}
            .card-header {{
                padding: 25px 25px 15px 25px;
                flex-shrink: 0;
            }}
            .imdb-badge {{
                font-size: 0.7rem;
                font-weight: 800;
                padding: 4px 10px;
                border-radius: 6px;
                border: 1px solid var(--accent);
                color: var(--accent);
                display: inline-block;
            }}
            .title {{
                margin: 15px 0 5px 0;
                font-size: 1.4rem;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                transition: all 0.3s ease;
                color: var(--text);
            }}
            .movie-card:hover .title {{
                white-space: normal;
                overflow: visible;
                text-overflow: clip;
            }}
            .meta-line {{
                color: var(--sub-text);
                font-size: 0.8rem;
                margin: 5px 0;
            }}
            .ott-line {{
                color: var(--accent);
                font-size: 0.75rem;
                font-weight: 700;
                margin: 5px 0 0 0;
            }}
            .summary-body {{
                padding: 0 25px 25px 25px;
                min-height: 0;
                opacity: 0;
                overflow-y: auto;
                transition: opacity 0.3s ease;
                scrollbar-width: thin;
                scrollbar-color: var(--scroll-thumb) transparent;
            }}
            .summary-body::-webkit-scrollbar {{
                width: 4px;
            }}
            .summary-body::-webkit-scrollbar-thumb {{
                background: var(--scroll-thumb);
                border-radius: 10px;
            }}
            .movie-card:hover .summary-body {{
                opacity: 1;
            }}
            .summary-text {{
                color: var(--sub-text);
                line-height: 1.6;
                font-size: 0.9rem;
                margin-top: 10px;
                border-top: 1px solid var(--border);
                padding-top: 15px;
            }}
            .summary-heading {{
                color: var(--text);
                font-size: 0.7rem;
                display: block;
                margin-bottom: 10px;
                letter-spacing: 1px;
            }}
        </style>
    </head>
    <body>
        <div class="results-grid">
            {"".join(cards_html)}
        </div>
    </body>
    </html>
    """

    rows_estimate=max(1,(len(items)+2)//3)
    iframe_height=rows_estimate*435+40
    st.iframe(grid_html,height=iframe_height)


# --- HEADER ---
st.markdown(
    """
    <div class="omni-header">
        <h1 class="omni-main-title">Omni<em>Sphere</em></h1>
        <p class="omni-subtitle">Movie & TVshow Recommendation system</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- LOAD ENGINE ---
try:
    df,matrix=load_engine()
except Exception as e:
    st.error(f"Error loading model engine ('ott_recommendation_engine.pkl'): {e}")
    st.stop()

labels=get_labels(df)

# --- SEARCH: type to filter, then pick from the list ---
selected_idx=st.selectbox(
    "Enter the movie you have recently watched...",
    options=list(labels),
    format_func=labels.get,
    index=None,
    placeholder="Enter the movie you have recently watched...",
    label_visibility="collapsed",
    key="movie_select",
)

# --- SECONDARY FILTERS & RECOMMEND BUTTON ---
col_type,col_ott,col_btn=st.columns([1,1,0.55],vertical_alignment="bottom")

type_options={"Both":"both","Movie":"movie","TV Show":"show"}
ott_options={
    "All Platforms":"all",
    "Netflix":"Netflix",
    "HBO Max":"HBO",
    "Amazon Prime":"Amazon Prime",
    "Disney+":"Disney+",
    "Apple TV":"Apple TV",
    "Paramount+":"Paramount+",
}

with col_type:
    st.markdown('<div class="field-label">TYPE</div>',unsafe_allow_html=True)
    selected_type=st.selectbox("TYPE",options=list(type_options),label_visibility="collapsed")

with col_ott:
    st.markdown('<div class="field-label">PLATFORM</div>',unsafe_allow_html=True)
    selected_ott=st.selectbox("PLATFORM",options=list(ott_options),label_visibility="collapsed")

with col_btn:
    recommend_clicked=st.button("Recommend",key="recommend_btn")

# --- EXECUTION & RESULTS ---
if recommend_clicked:
    if selected_idx is None:
        st.warning("Please select a title from the list first.")
        st.stop()

    with st.spinner("SYNCHRONIZING..."):
        recs=recommend(df,matrix,selected_idx,ott_options[selected_ott],type_options[selected_type])

    if recs.empty:
        st.info("No matching recommendations found for that title and filter selection.")
    else:
        render_results_grid(recs)
