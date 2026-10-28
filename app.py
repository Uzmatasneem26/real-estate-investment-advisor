from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cleaned_housing_data.csv"
MODEL_DIR = ROOT / "models"
CURRENT_YEAR = 2026

PALETTE = ["#6366F1", "#14B8A6", "#F59E0B", "#EC4899", "#3B82F6", "#84CC16", "#F97316", "#8B5CF6"]
SEQ = ["#E0E7FF", "#A5B4FC", "#6366F1", "#4338CA", "#312E81"]

st.set_page_config(page_title="Real Estate Investment Advisor", page_icon="🏡",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1400px;}
#MainMenu, footer {visibility: hidden;}

/* Hero */
.hero {background: linear-gradient(120deg, #312E81 0%, #4F46E5 45%, #0EA5E9 100%);
       border-radius: 22px; padding: 30px 36px; color: #fff; margin-bottom: 22px;
       box-shadow: 0 18px 40px -18px rgba(79,70,229,.55); position: relative; overflow: hidden;}
.hero:after {content:""; position:absolute; right:-60px; top:-60px; width:260px; height:260px;
       background: radial-gradient(circle, rgba(255,255,255,.22), transparent 70%); border-radius:50%;}
.hero h1 {font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; letter-spacing: -.5px;}
.hero p {margin: 0; opacity: .88; font-size: 1rem;}
.badge {display:inline-block; padding: 3px 11px; border-radius: 999px; font-size: .72rem; font-weight: 600;
        background: rgba(255,255,255,.18); margin-right: 6px; margin-top: 12px; backdrop-filter: blur(4px);}

/* Metric cards */
[data-testid="stMetric"] {background: linear-gradient(145deg, rgba(99,102,241,.10), rgba(14,165,233,.04));
       border: 1px solid rgba(99,102,241,.22); padding: 16px 18px; border-radius: 16px;
       transition: transform .18s ease, box-shadow .18s ease;}
[data-testid="stMetric"]:hover {transform: translateY(-3px); box-shadow: 0 12px 24px -14px rgba(99,102,241,.6);}
[data-testid="stMetricLabel"] {font-size: .82rem; opacity: .8; font-weight: 600; text-transform: uppercase; letter-spacing: .4px;}
[data-testid="stMetricValue"] {font-weight: 800; font-size: 1.55rem;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap: 6px; border-bottom: none;}
.stTabs [data-baseweb="tab"] {border-radius: 10px; padding: 8px 18px; background: rgba(99,102,241,.07); font-weight: 600;}
.stTabs [aria-selected="true"] {background: linear-gradient(120deg,#4F46E5,#6366F1); color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* Misc */
.section-note {padding: 13px 16px; border-left: 4px solid #6366F1; background: rgba(99,102,241,.08);
       border-radius: 10px; font-size: .92rem;}
.card {border: 1px solid rgba(148,163,184,.25); border-radius: 16px; padding: 16px 18px; margin-bottom: 10px;
       background: rgba(148,163,184,.05);}
.card h4 {margin: 0 0 4px 0; font-size: 1rem;}
.card .meta {color: #64748b; font-size: .85rem;}
.card .price {font-weight: 800; font-size: 1.2rem; color: #4F46E5;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600;
       background: rgba(20,184,166,.15); color: #0F766E; margin-right: 5px;}
.stButton>button, .stDownloadButton>button {border-radius: 12px; font-weight: 600;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, rgba(79,70,229,.10), rgba(14,165,233,.04));}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- Helpers
@st.cache_data(show_spinner=False)
def load_data():
    if not DATA_PATH.exists():
        return None
    data = pd.read_csv(DATA_PATH, low_memory=False)
    if "Age_of_Property" not in data and "Year_Built" in data:
        data["Age_of_Property"] = CURRENT_YEAR - pd.to_numeric(data["Year_Built"], errors="coerce")
    if "Amenities_Count" not in data and "Amenities" in data:
        data["Amenities_Count"] = data["Amenities"].fillna("").apply(
            lambda x: len([v for v in str(x).split(",") if v.strip()]))
    if "Price_per_SqFt" not in data and {"Price_in_Lakhs", "Size_in_SqFt"} <= set(data.columns):
        data["Price_per_SqFt"] = data["Price_in_Lakhs"] * 100000 / data["Size_in_SqFt"].clip(lower=1)
    return data


@st.cache_resource(show_spinner=False)
def load_model(filename):
    path = MODEL_DIR / filename
    return joblib.load(path) if path.exists() else None


def load_metrics(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def money_lakhs(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if abs(value) >= 100:
        return f"₹{value/100:,.2f} Cr"
    return f"₹{value:,.2f} L"


def chart_theme(fig, height=390):
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=15, r=15, t=60, b=15),
                      legend_title_text="", font=dict(family="Inter, sans-serif"),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      title=dict(font=dict(size=16)), colorway=PALETTE,
                      hoverlabel=dict(font_family="Inter"))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.22)")
    return fig


def hero(title, subtitle, badges=()):
    chips = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p>{chips}</div>', unsafe_allow_html=True)


def build_feature_row(df, state, city, locality, ptype, bhk, size, price, year, furnished,
                      floor, total_floors, transport, schools, hospitals, parking, security,
                      facing, owner, availability, amenities):
    return pd.DataFrame([{
        "State": state, "City": city, "Locality": locality, "Property_Type": ptype,
        "BHK": int(bhk), "Size_in_SqFt": int(size), "Year_Built": int(year),
        "Furnished_Status": furnished, "Floor_No": min(int(floor), int(total_floors)),
        "Total_Floors": int(total_floors), "Age_of_Property": max(0, CURRENT_YEAR - int(year)),
        "Nearby_Schools": int(schools), "Nearby_Hospitals": int(hospitals),
        "Public_Transport_Accessibility": transport, "Parking_Space": parking,
        "Security": security, "Amenities": ", ".join(amenities) if amenities else "None",
        "Facing": facing, "Owner_Type": owner, "Availability_Status": availability,
        "Amenities_Count": len(amenities), "Price_per_SqFt": float(price) * 100000 / max(int(size), 1),
        "Price_in_Lakhs": float(price)
    }])


def gauge(value, title, suffix="%", rng=(0, 100)):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": suffix, "font": {"size": 38}},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": list(rng)}, "bar": {"color": "#4F46E5", "thickness": .28},
               "bgcolor": "rgba(148,163,184,.15)", "borderwidth": 0,
               "steps": [{"range": [rng[0], rng[1] * .4], "color": "rgba(239,68,68,.18)"},
                         {"range": [rng[1] * .4, rng[1] * .7], "color": "rgba(245,158,11,.18)"},
                         {"range": [rng[1] * .7, rng[1]], "color": "rgba(20,184,166,.22)"}]}))
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif"))
    return fig


def emi(principal, annual_rate, years):
    r = annual_rate / 1200
    n = int(years * 12)
    if n <= 0:
        return 0.0
    return principal / n if r == 0 else principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def opts(col):
    return sorted(df[col].dropna().astype(str).unique().tolist()) if col in df else ["N/A"]


df = load_data()

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("## 🏡 Property Advisor")
    st.caption("Real Estate Investment Analytics")
    st.divider()
    page = st.radio("NAVIGATION", ["📊 Overview", "🔎 Property Explorer", "🎯 Investment Prediction",
                                   "📈 Five-Year Forecast", "🧪 Model Performance", "ℹ️ About"])
    st.divider()
    if df is not None:
        st.markdown("**Dataset snapshot**")
        st.caption(f"{len(df):,} listings · {df['City'].nunique() if 'City' in df else 0} cities")
    st.caption("Estimates are for educational analysis only, not financial advice.")

if df is None:
    st.error("Cleaned dataset not found. Put `cleaned_housing_data.csv` in the `data/` folder and run `python scripts/data_cleaning.py`.")
    st.stop()

required = ["State", "City", "Property_Type", "Price_in_Lakhs", "Size_in_SqFt"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Dataset is missing required columns: {', '.join(missing)}")
    st.stop()

page = page.split(" ", 1)[1]

# ================================================================ OVERVIEW
if page == "Overview":
    hero("Real Estate Investment Advisor",
         "Property-market intelligence, investment screening and five-year value scenarios.",
         ["Market analytics", "ML screening", "Scenario planning"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Listings", f"{len(df):,}")
    c2.metric("States / UTs", f"{df['State'].nunique():,}")
    c3.metric("Cities", f"{df['City'].nunique():,}")
    c4.metric("Median price", money_lakhs(df["Price_in_Lakhs"].median()))
    c5.metric("Median ₹/sq ft", f"₹{df['Price_per_SqFt'].median():,.0f}" if "Price_per_SqFt" in df else "—")

    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(["Market map", "Price drivers", "Distributions", "State table"])

    with tab1:
        l, r = st.columns([1.15, 1])
        with l:
            tm = df.groupby(["State", "City"], as_index=False).agg(
                Listings=("Price_in_Lakhs", "size"), Median_Price=("Price_in_Lakhs", "median"))
            fig = px.treemap(tm, path=[px.Constant("India"), "State", "City"], values="Listings",
                             color="Median_Price", color_continuous_scale=SEQ,
                             title="Listing volume (size) vs median price (colour)")
            fig.update_traces(root_color="rgba(0,0,0,0)")
            st.plotly_chart(chart_theme(fig, 470), use_container_width=True)
        with r:
            cs = df.groupby("City", as_index=False).agg(Median_Price=("Price_in_Lakhs", "median"),
                                                        Listings=("Price_in_Lakhs", "size"))
            cs = cs.sort_values("Median_Price", ascending=False).head(12).sort_values("Median_Price")
            fig = px.bar(cs, x="Median_Price", y="City", orientation="h", color="Median_Price",
                         color_continuous_scale=SEQ, text_auto=".1f", hover_data={"Listings": True},
                         title="Top 12 cities by median price (₹ lakhs)", labels={"Median_Price": "", "City": ""})
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(chart_theme(fig, 470), use_container_width=True)

    with tab2:
        l, r = st.columns(2)
        sample = df.sample(min(6000, len(df)), random_state=42)
        with l:
            fig = px.scatter(sample, x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type", opacity=.55,
                             hover_data=["City", "BHK"] if "BHK" in df else ["City"],
                             title="Size vs listed price", trendline=None,
                             labels={"Size_in_SqFt": "Size (sq ft)", "Price_in_Lakhs": "Price (₹ L)"})
            st.plotly_chart(chart_theme(fig, 420), use_container_width=True)
        with r:
            num = df.select_dtypes("number").drop(columns=[c for c in ["ID"] if c in df], errors="ignore")
            if num.shape[1] > 2:
                corr = num.corr().round(2)
                fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                                title="Correlation heatmap (numeric features)")
                st.plotly_chart(chart_theme(fig, 420), use_container_width=True)
        l, r = st.columns(2)
        with l:
            if "BHK" in df:
                b = df.groupby("BHK", as_index=False)["Price_in_Lakhs"].median()
                fig = px.bar(b, x="BHK", y="Price_in_Lakhs", text_auto=".1f", title="Median price by BHK",
                             labels={"Price_in_Lakhs": "Median price (₹ L)"}, color_discrete_sequence=[PALETTE[1]])
                st.plotly_chart(chart_theme(fig, 360), use_container_width=True)
        with r:
            if "Age_of_Property" in df:
                d = df.copy()
                d["Age band"] = pd.cut(d["Age_of_Property"], [-1, 2, 5, 10, 20, 100],
                                       labels=["0–2 yrs", "3–5", "6–10", "11–20", "20+"])
                a = d.groupby("Age band", observed=True, as_index=False)["Price_per_SqFt"].median()
                fig = px.line(a, x="Age band", y="Price_per_SqFt", markers=True,
                              title="Median ₹/sq ft by property age", labels={"Price_per_SqFt": "₹ / sq ft"})
                fig.update_traces(line=dict(width=3, color=PALETTE[0]), marker=dict(size=10))
                st.plotly_chart(chart_theme(fig, 360), use_container_width=True)

    with tab3:
        l, r = st.columns(2)
        with l:
            tmix = df["Property_Type"].value_counts().rename_axis("Property type").reset_index(name="Listings")
            fig = px.pie(tmix, names="Property type", values="Listings", hole=.6,
                         title="Property type mix", color_discrete_sequence=PALETTE)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(chart_theme(fig, 400), use_container_width=True)
        with r:
            pdata = df[df["Price_in_Lakhs"].between(df["Price_in_Lakhs"].quantile(.01), df["Price_in_Lakhs"].quantile(.99))]
            fig = px.histogram(pdata, x="Price_in_Lakhs", nbins=45, marginal="box",
                               color_discrete_sequence=[PALETTE[0]],
                               title="Price distribution (1st–99th percentile)",
                               labels={"Price_in_Lakhs": "Price (₹ lakhs)"})
            st.plotly_chart(chart_theme(fig, 400), use_container_width=True)

    with tab4:
        ss = df.groupby("State", as_index=False).agg(
            Listings=("Price_in_Lakhs", "size"), Median_Price=("Price_in_Lakhs", "median"),
            Average_Size=("Size_in_SqFt", "mean")).sort_values("Listings", ascending=False)
        st.dataframe(ss.round(2), use_container_width=True, hide_index=True,
                     column_config={
                         "Listings": st.column_config.ProgressColumn("Listings", min_value=0,
                                                                     max_value=int(ss["Listings"].max()), format="%d"),
                         "Median_Price": st.column_config.NumberColumn("Median price (₹ L)", format="%.1f"),
                         "Average_Size": st.column_config.NumberColumn("Avg size (sq ft)", format="%.0f")})

    st.markdown('<div class="section-note">These are listing-data patterns. Price differences reflect location, property mix and data coverage; they do not prove one market will outperform another.</div>', unsafe_allow_html=True)

# ================================================================ EXPLORER
elif page == "Property Explorer":
    hero("Property Explorer", "Filter, rank and compare listings to build your shortlist.", ["Smart filters", "City compare", "CSV export"])

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        states = f1.multiselect("State", opts("State"))
        ptypes = f2.multiselect("Property type", opts("Property_Type"))
        bhks = f3.multiselect("BHK", sorted(df["BHK"].dropna().unique().tolist()) if "BHK" in df else [])
        g1, g2 = st.columns(2)
        pmin = float(max(0, np.floor(df["Price_in_Lakhs"].min())))
        pmax = float(np.ceil(df["Price_in_Lakhs"].quantile(.99)))
        if pmax <= pmin:
            pmax = pmin + 1
        prange = g1.slider("Price (₹ lakhs, capped at 99th pct.)", pmin, pmax, (pmin, pmax))
        smin, smax = int(df["Size_in_SqFt"].min()), int(df["Size_in_SqFt"].quantile(.99))
        srange = g2.slider("Size (sq ft)", smin, max(smax, smin + 1), (smin, max(smax, smin + 1)))
        h1, h2, h3 = st.columns(3)
        furn = h1.multiselect("Furnishing", opts("Furnished_Status")) if "Furnished_Status" in df else []
        sort_by = h2.selectbox("Sort by", ["Price: low → high", "Price: high → low", "₹/sq ft: low → high", "Size: large → small"])
        top_n = h3.slider("Cards to show", 3, 12, 6)

    flt = df[df["Price_in_Lakhs"].between(*prange) & df["Size_in_SqFt"].between(*srange)].copy()
    if states: flt = flt[flt["State"].astype(str).isin(states)]
    if ptypes: flt = flt[flt["Property_Type"].astype(str).isin(ptypes)]
    if bhks: flt = flt[flt["BHK"].isin(bhks)]
    if furn: flt = flt[flt["Furnished_Status"].astype(str).isin(furn)]

    sorters = {"Price: low → high": ("Price_in_Lakhs", True), "Price: high → low": ("Price_in_Lakhs", False),
               "₹/sq ft: low → high": ("Price_per_SqFt", True), "Size: large → small": ("Size_in_SqFt", False)}
    scol, asc = sorters[sort_by]
    if scol in flt:
        flt = flt.sort_values(scol, ascending=asc)

    a, b, c, d = st.columns(4)
    a.metric("Matches", f"{len(flt):,}")
    b.metric("Median price", money_lakhs(flt["Price_in_Lakhs"].median()) if len(flt) else "—")
    c.metric("Median size", f"{flt['Size_in_SqFt'].median():,.0f} sq ft" if len(flt) else "—")
    d.metric("Median ₹/sq ft", f"₹{flt['Price_per_SqFt'].median():,.0f}" if len(flt) and "Price_per_SqFt" in flt else "—")

    if flt.empty:
        st.warning("No properties match these filters. Widen the ranges or clear a selection.")
    else:
        t1, t2, t3, t4 = st.tabs(["Shortlist cards", "Charts", "City comparison", "Data table"])
        with t1:
            cols = st.columns(3)
            for i, (_, r) in enumerate(flt.head(top_n).iterrows()):
                bhk_txt = f"{int(r['BHK'])} BHK" if "BHK" in r and pd.notna(r["BHK"]) else ""
                loc = r["Locality"] if "Locality" in r and pd.notna(r.get("Locality")) else ""
                ppsf = f"₹{r['Price_per_SqFt']:,.0f}/sq ft" if "Price_per_SqFt" in r else ""
                with cols[i % 3]:
                    st.markdown(f"""<div class="card">
                    <div class="price">{money_lakhs(r['Price_in_Lakhs'])}</div>
                    <h4>{bhk_txt} {r['Property_Type']}</h4>
                    <div class="meta">{loc}{', ' if loc else ''}{r['City']}, {r['State']}</div>
                    <div style="margin-top:8px"><span class="pill">{int(r['Size_in_SqFt']):,} sq ft</span><span class="pill">{ppsf}</span></div>
                    </div>""", unsafe_allow_html=True)
        with t2:
            l, r = st.columns(2)
            with l:
                fig = px.box(flt.sample(min(12000, len(flt)), random_state=42), x="Property_Type", y="Price_in_Lakhs",
                             color="Property_Type", title="Price spread by property type",
                             labels={"Property_Type": "", "Price_in_Lakhs": "Price (₹ L)"})
                st.plotly_chart(chart_theme(fig, 400), use_container_width=True)
            with r:
                fig = px.scatter(flt.sample(min(5000, len(flt)), random_state=42), x="Size_in_SqFt",
                                 y="Price_per_SqFt" if "Price_per_SqFt" in flt else "Price_in_Lakhs",
                                 color="BHK" if "BHK" in flt else "Property_Type", opacity=.6,
                                 hover_data=["City", "Property_Type"], title="Size vs ₹/sq ft")
                st.plotly_chart(chart_theme(fig, 400), use_container_width=True)
        with t3:
            cities = st.multiselect("Pick 2–6 cities to compare", sorted(flt["City"].astype(str).unique()),
                                    default=flt["City"].value_counts().head(4).index.astype(str).tolist())
            if len(cities) >= 2:
                cmp_df = flt[flt["City"].astype(str).isin(cities)]
                l, r = st.columns(2)
                with l:
                    fig = px.violin(cmp_df, x="City", y="Price_in_Lakhs", color="City", box=True, points=False,
                                    title="Price distribution", labels={"Price_in_Lakhs": "Price (₹ L)", "City": ""})
                    st.plotly_chart(chart_theme(fig, 400), use_container_width=True)
                with r:
                    agg = cmp_df.groupby("City").agg(Median_Price=("Price_in_Lakhs", "median"),
                                                     Median_PSF=("Price_per_SqFt", "median") if "Price_per_SqFt" in cmp_df else ("Price_in_Lakhs", "median"),
                                                     Avg_Size=("Size_in_SqFt", "mean"),
                                                     Listings=("Price_in_Lakhs", "size")).round(1).reset_index()
                    st.dataframe(agg, use_container_width=True, hide_index=True)
            else:
                st.info("Select at least two cities.")
        with t4:
            st.dataframe(flt.head(1500), use_container_width=True, hide_index=True)
        st.download_button("⬇ Download filtered listings (CSV)", flt.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_properties.csv", mime="text/csv")

# ================================================================ PREDICTION
elif page == "Investment Prediction":
    hero("Investment Screening", "Model-based screening of a property profile, with a due-diligence checklist.",
         ["Proxy-label classifier", "Not a profit probability"])
    st.warning("The source data has no verified investment-return label. This classifier uses a proxy label based on relative city price, access, amenities and age. It is not a profit probability.")
    model = load_model("investment_model.pkl")
    if model is None:
        st.info("The investment model has not been created yet. Run `python scripts/train_models.py` from the project root.")
    else:
        with st.form("investment_form"):
            t1, t2, t3 = st.tabs(["📍 Location & type", "🏗 Building details", "🌿 Surroundings & amenities"])
            with t1:
                a, b, c = st.columns(3)
                state = a.selectbox("State", opts("State"), key="inv_state")
                city = b.selectbox("City", opts("City"), key="inv_city")
                locality = c.selectbox("Locality", opts("Locality"), key="inv_locality")
                a, b, c = st.columns(3)
                ptype = a.selectbox("Property type", opts("Property_Type"), key="inv_type")
                bhk = b.number_input("BHK", 1, 15, 3, key="inv_bhk")
                size = c.number_input("Size (sq ft)", 100, 20000, 1500, key="inv_size")
                a, b = st.columns(2)
                price = a.number_input("Current listed price (₹ lakhs)", 1.0, 100000.0, 100.0, 5.0, key="inv_price")
                availability = b.selectbox("Availability", opts("Availability_Status"), key="inv_availability")
            with t2:
                a, b, c = st.columns(3)
                year = a.number_input("Year built", 1900, CURRENT_YEAR, 2018, key="inv_year")
                furnished = b.selectbox("Furnishing", opts("Furnished_Status"), key="inv_furnished")
                owner = c.selectbox("Owner type", opts("Owner_Type"), key="inv_owner")
                a, b, c = st.columns(3)
                floor = a.number_input("Floor number", 0, 100, 2, key="inv_floor")
                total_floors = b.number_input("Total floors", 1, 100, 10, key="inv_total_floors")
                facing = c.selectbox("Facing", opts("Facing"), key="inv_facing")
            with t3:
                a, b, c = st.columns(3)
                transport = a.selectbox("Public transport access", opts("Public_Transport_Accessibility"), key="inv_transport")
                schools = b.number_input("Nearby schools", 0, 100, 5, key="inv_schools")
                hospitals = c.number_input("Nearby hospitals", 0, 100, 5, key="inv_hospitals")
                a, b = st.columns(2)
                parking = a.selectbox("Parking", opts("Parking_Space"), key="inv_parking")
                security = b.selectbox("Security", opts("Security"), key="inv_security")
                amen_opts = sorted({i.strip() for v in df["Amenities"].dropna().astype(str).head(5000)
                                    for i in v.split(",") if i.strip()}) if "Amenities" in df else ["Gym", "Pool", "Garden"]
                amenities = st.multiselect("Amenities", amen_opts, default=amen_opts[:2], key="inv_amenities")
            submitted = st.form_submit_button("Evaluate property", type="primary", use_container_width=True)

        if submitted:
            row = build_feature_row(df, state, city, locality, ptype, bhk, size, price, year, furnished, floor,
                                    total_floors, transport, schools, hospitals, parking, security, facing,
                                    owner, availability, amenities)
            try:
                pred = int(model.predict(row)[0])
                proba = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
                conf = float(proba[pred]) * 100 if proba is not None else None
                pos_score = float(proba[1]) * 100 if proba is not None and len(proba) > 1 else None

                if pred == 1:
                    st.success("Screening result: Potentially suitable for further review")
                else:
                    st.warning("Screening result: Needs closer review")

                l, r = st.columns([1, 1.2])
                with l:
                    if pos_score is not None:
                        st.plotly_chart(gauge(pos_score, "Proxy-positive score"), use_container_width=True)
                with r:
                    a, b = st.columns(2)
                    a.metric("Screening category", "Proxy positive" if pred == 1 else "Proxy negative")
                    b.metric("Model confidence", f"{conf:.1f}%" if conf is not None else "N/A")
                    ppsf = price * 100000 / size
                    city_med = df.loc[df["City"].astype(str) == city, "Price_per_SqFt"].median() if "Price_per_SqFt" in df else np.nan
                    a, b = st.columns(2)
                    a.metric("Your ₹/sq ft", f"₹{ppsf:,.0f}")
                    if pd.notna(city_med):
                        b.metric(f"{city} median ₹/sq ft", f"₹{city_med:,.0f}", delta=f"{(ppsf/city_med-1)*100:+.1f}% vs city",
                                 delta_color="inverse")

                st.markdown("#### Due-diligence checklist")
                items = ["Verify title, RERA registration and approvals", "Inspect structure and building condition",
                         "Compare recent transactions in the same locality", "Estimate rent, maintenance and taxes",
                         "Assess commute, access and resale demand"]
                cc = st.columns(len(items))
                for col, it in zip(cc, items):
                    col.checkbox(it, key=f"chk_{it}")
                st.caption("Model confidence refers to the proxy label, not the likelihood of financial profit.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}. Retrain the model after running `python scripts/data_cleaning.py`.")

# ================================================================ FORECAST
elif page == "Five-Year Forecast":
    hero("Five-Year Scenario Lab", "Stress-test appreciation, financing and rental income assumptions.",
         ["Bear / Base / Bull", "EMI & yield", "Sensitivity"])
    st.info("The dataset contains current listed prices, not observed future prices. Future values are scenarios calculated from your assumptions, not predictions learned by the model.")
    model = load_model("price_prediction_model.pkl")
    if model is None:
        st.warning("The price model has not been created yet. Run `python scripts/train_models.py` from the project root.")
    else:
        with st.form("forecast_form"):
            a, b, c = st.columns(3)
            state = a.selectbox("State", opts("State"), key="fc_state")
            city = b.selectbox("City", opts("City"), key="fc_city")
            ptype = c.selectbox("Property type", opts("Property_Type"), key="fc_type")
            a, b, c = st.columns(3)
            bhk = a.number_input("BHK", 1, 15, 3, key="fc_bhk")
            size = b.number_input("Size (sq ft)", 100, 20000, 1500, key="fc_size")
            year = c.number_input("Year built", 1900, CURRENT_YEAR, 2018, key="fc_year")
            current_price = st.number_input("Current asking price (₹ lakhs)", 1.0, 100000.0, 100.0, 5.0, key="fc_price")
            st.markdown("**Scenario assumptions**")
            a, b, c = st.columns(3)
            bear = a.slider("Bear appreciation (% / yr)", 0.0, 20.0, 2.0, 0.5)
            base = b.slider("Base appreciation (% / yr)", 0.0, 20.0, 6.0, 0.5)
            bull = c.slider("Bull appreciation (% / yr)", 0.0, 20.0, 10.0, 0.5)
            st.markdown("**Financing & rental (optional)**")
            a, b, c, d = st.columns(4)
            down_pct = a.slider("Down payment %", 0, 100, 20)
            loan_rate = b.slider("Loan rate % p.a.", 5.0, 15.0, 8.75, 0.25)
            tenure = c.slider("Tenure (years)", 5, 30, 20)
            rent = d.number_input("Monthly rent (₹)", 0, 1000000, 25000, 1000)
            submitted = st.form_submit_button("Run scenarios", type="primary", use_container_width=True)

        if submitted:
            sample = df.iloc[[0]].copy()
            for k, v in {"State": state, "City": city, "Property_Type": ptype, "BHK": bhk, "Size_in_SqFt": size,
                         "Year_Built": year, "Age_of_Property": max(0, CURRENT_YEAR - year),
                         "Price_in_Lakhs": current_price}.items():
                sample.loc[:, k] = v
            if "Price_per_SqFt" in sample.columns:
                sample.loc[:, "Price_per_SqFt"] = current_price * 100000 / size
            try:
                est = max(0.0, float(model.predict(sample)[0]))
                fv = {n: est * (1 + r / 100) ** 5 for n, r in [("Bear", bear), ("Base", base), ("Bull", bull)]}
                a, b, c, d = st.columns(4)
                a.metric("Model-estimated value", money_lakhs(est),
                         delta=f"{(est/current_price-1)*100:+.1f}% vs asking", delta_color="off")
                b.metric("Bear · 5 yr", money_lakhs(fv["Bear"]), delta=f"{(fv['Bear']/est-1)*100:.1f}%")
                c.metric("Base · 5 yr", money_lakhs(fv["Base"]), delta=f"{(fv['Base']/est-1)*100:.1f}%")
                d.metric("Bull · 5 yr", money_lakhs(fv["Bull"]), delta=f"{(fv['Bull']/est-1)*100:.1f}%")

                t1, t2, t3 = st.tabs(["Value paths", "Sensitivity", "Financing & yield"])
                with t1:
                    yrs = list(range(CURRENT_YEAR, CURRENT_YEAR + 6))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=yrs, y=[est * (1 + bull / 100) ** i for i in range(6)], line=dict(width=0),
                                             showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=yrs, y=[est * (1 + bear / 100) ** i for i in range(6)], fill="tonexty",
                                             fillcolor="rgba(99,102,241,.14)", line=dict(width=0), name="Bear–Bull range",
                                             hoverinfo="skip"))
                    for name, r, col in [("Bear", bear, "#EF4444"), ("Base", base, "#4F46E5"), ("Bull", bull, "#14B8A6")]:
                        fig.add_trace(go.Scatter(x=yrs, y=[est * (1 + r / 100) ** i for i in range(6)], mode="lines+markers",
                                                 name=f"{name} ({r:.1f}%)", line=dict(width=3, color=col)))
                    fig.add_hline(y=current_price, line_dash="dot", line_color="#94A3B8",
                                  annotation_text="Asking price", annotation_position="bottom right")
                    fig.update_layout(title="Scenario value paths (₹ lakhs)", yaxis_title="₹ lakhs")
                    st.plotly_chart(chart_theme(fig, 440), use_container_width=True)
                with t2:
                    rates = list(range(0, 16, 2))
                    yrs_axis = [1, 2, 3, 4, 5]
                    z = [[est * (1 + r / 100) ** y for y in yrs_axis] for r in rates]
                    fig = px.imshow(z, x=[f"Yr {y}" for y in yrs_axis], y=[f"{r}%" for r in rates],
                                    color_continuous_scale=SEQ, text_auto=".0f", aspect="auto",
                                    title="Value (₹ lakhs) by appreciation rate and horizon",
                                    labels=dict(color="₹ L", y="Annual rate"))
                    st.plotly_chart(chart_theme(fig, 420), use_container_width=True)
                with t3:
                    price_rs = current_price * 100000
                    down = price_rs * down_pct / 100
                    loan = price_rs - down
                    m_emi = emi(loan, loan_rate, tenure)
                    gross_yield = rent * 12 / price_rs * 100 if price_rs else 0
                    months = 60
                    paid_emi = m_emi * months
                    # outstanding principal after 5 yrs
                    r_m = loan_rate / 1200
                    n = tenure * 12
                    bal = loan if r_m == 0 else loan * ((1 + r_m) ** n - (1 + r_m) ** months) / ((1 + r_m) ** n - 1)
                    base_fv_rs = fv["Base"] * 100000
                    equity_gain = base_fv_rs - bal - down
                    net_cash = rent * months - paid_emi
                    total_return = equity_gain + rent * months - paid_emi
                    a, b, c, d = st.columns(4)
                    a.metric("Monthly EMI", f"₹{m_emi:,.0f}")
                    b.metric("Gross rental yield", f"{gross_yield:.2f}%")
                    c.metric("Loan balance @ 5 yr", f"₹{bal/100000:,.1f} L")
                    d.metric("Rent − EMI (5 yr)", f"₹{net_cash/100000:,.1f} L")
                    wf = go.Figure(go.Waterfall(
                        x=["Down payment", "Base-case value", "Loan outstanding", "Rent received", "EMIs paid", "Net position"],
                        measure=["relative", "relative", "relative", "relative", "relative", "total"],
                        y=[-down / 1e5, base_fv_rs / 1e5, -bal / 1e5, rent * months / 1e5, -paid_emi / 1e5, 0],
                        increasing=dict(marker=dict(color="#14B8A6")), decreasing=dict(marker=dict(color="#EF4444")),
                        totals=dict(marker=dict(color="#4F46E5"))))
                    wf.update_layout(title="Illustrative 5-year cash waterfall — base case (₹ lakhs)", showlegend=False)
                    st.plotly_chart(chart_theme(wf, 420), use_container_width=True)
                    st.caption("Illustrative only: ignores stamp duty, brokerage, maintenance, property tax, tax benefits, vacancy, and the time value of money.")
                    st.metric("Illustrative net gain (base case)", f"₹{total_return/100000:,.1f} L",
                              delta=f"{total_return/down*100:.1f}% on down payment" if down else None)
            except Exception as exc:
                st.error(f"Forecast failed: {exc}. Retrain the price model and try again.")

# ================================================================ MODEL PERFORMANCE
elif page == "Model Performance":
    hero("Model Performance", "Held-out test metrics, interpretation and validation caveats.", ["80/20 split", "Proxy label"])
    cls = load_metrics("classification_metrics.json")
    reg = load_metrics("regression_metrics.json")
    if cls is None or reg is None:
        st.info("Performance files are not available yet. Run `python scripts/data_cleaning.py` then `python scripts/train_models.py`.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Classifier accuracy", f"{cls['accuracy']:.1%}" if cls.get("accuracy") is not None else "N/A")
        b.metric("Classifier F1", f"{cls['f1']:.3f}" if cls.get("f1") is not None else "N/A")
        c.metric("Price MAE", f"₹{reg['mae_lakhs']:,.2f} L" if reg.get("mae_lakhs") is not None else "N/A")
        d.metric("Price R²", f"{reg['r2']:.3f}" if reg.get("r2") is not None else "N/A")

        t1, t2, t3 = st.tabs(["Classifier", "Regressor", "Validation notes"])
        with t1:
            rows = [{"Metric": "Accuracy", "Value": cls.get("accuracy")}, {"Metric": "F1 score", "Value": cls.get("f1")},
                    {"Metric": "ROC-AUC", "Value": cls.get("roc_auc")}]
            cdf = pd.DataFrame(rows).dropna()
            l, r = st.columns([1.2, 1])
            with l:
                if not cdf.empty:
                    fig = px.bar(cdf, x="Metric", y="Value", text=cdf["Value"].map(lambda x: f"{x:.3f}"), range_y=[0, 1.05],
                                 color="Metric", color_discrete_sequence=PALETTE, title="Classification metrics (higher is better)")
                    fig.update_traces(textposition="outside", showlegend=False)
                    st.plotly_chart(chart_theme(fig, 380), use_container_width=True)
            with r:
                if cls.get("roc_auc") is not None:
                    st.plotly_chart(gauge(cls["roc_auc"] * 100, "ROC-AUC"), use_container_width=True)
                st.write("- **Accuracy:** share classified correctly.\n- **F1:** balances precision and recall.\n- **ROC-AUC:** how well the score ranks positives above negatives.")
        with t2:
            rdf = pd.DataFrame([{"Metric": "MAE (₹ L)", "Value": reg.get("mae_lakhs")},
                                {"Metric": "RMSE (₹ L)", "Value": reg.get("rmse_lakhs")}]).dropna()
            l, r = st.columns([1.2, 1])
            with l:
                if not rdf.empty:
                    fig = px.bar(rdf, x="Metric", y="Value", text=rdf["Value"].map(lambda x: f"₹{x:,.2f} L"),
                                 color="Metric", color_discrete_sequence=["#F59E0B", "#EC4899"],
                                 title="Price error metrics (lower is better)")
                    fig.update_traces(textposition="outside", showlegend=False)
                    st.plotly_chart(chart_theme(fig, 380), use_container_width=True)
            with r:
                if reg.get("r2") is not None:
                    st.plotly_chart(gauge(max(0, reg["r2"]) * 100, "R² (explained variance)"), use_container_width=True)
                mae, med = reg.get("mae_lakhs"), float(df["Price_in_Lakhs"].median())
                if mae is not None and med > 0:
                    st.write(f"MAE ≈ **{float(mae)/med*100:.1f}%** of the median listing price ({money_lakhs(med)}).")
                if reg.get("r2") is not None:
                    if reg["r2"] >= .8:
                        st.success("Strong fit on this split; still verify on other locations or time periods.")
                    elif reg["r2"] >= .5:
                        st.info("Moderate fit. Review feature quality and validate on newer listings.")
                    else:
                        st.warning("Limited fit. Review features, outliers and generalization.")
        with t3:
            st.markdown("""
**Evaluation setup**
- Random 80/20 train-test split with a fixed seed.
- The classifier uses a rule-generated proxy label; real investment outcomes are absent.
- The regressor estimates current listed price; five-year values come from user-chosen appreciation rates.

**Caveats**
- Scores describe a held-out split of the same dataset, not future markets.
- Random splits do not fully test geographic or time-based generalization.
- No metric here validates rental yield, realized resale profit, transaction costs or future appreciation.
- For production, use verified outcomes, a time-based test set, and post-deployment monitoring.
""")
            m1, m2 = st.columns(2)
            with m1: st.json(cls)
            with m2: st.json(reg)

# ================================================================ ABOUT
elif page == "About":
    hero("About the project", "An interactive analytics tool for exploring listings and screening property scenarios.",
         ["Python", "Scikit-learn", "Streamlit", "Plotly"])
    l, r = st.columns([1.3, 1])
    with l:
        st.markdown("""
**Components**
- **Overview:** treemap, correlations, price drivers and state tables.
- **Explorer:** multi-filter shortlist cards, city comparison and CSV export.
- **Investment screening:** Random Forest on a proxy label, with gauge and checklist.
- **Scenario lab:** bear/base/bull paths, sensitivity heatmap, EMI and rental yield.
- **Model performance:** held-out metrics with interpretation.

**Data limitation:** the dataset has property attributes and current listed prices only — no realized returns or future prices. Results are demonstrations for analysis and education, not financial advice.
""")
    with r:
        a, b = st.columns(2)
        a.metric("Rows", f"{len(df):,}")
        b.metric("Columns", f"{df.shape[1]:,}")
        a.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        b.metric("Cities", f"{df['City'].nunique():,}")
    st.markdown("#### Data preview")
    st.dataframe(df.head(15), use_container_width=True, hide_index=True)