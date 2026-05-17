import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Emerging Digital Trends", page_icon="🚀", layout="wide")

BASE = ""

@st.cache_data
def load_data():
    df = pd.read_csv(BASE + "clean_reddit_comments.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data
def load_time_series():
    ts = pd.read_csv(BASE + "topic_time_series.csv")
    ts["week"] = pd.to_datetime(ts["week"])
    return ts.sort_values("week").reset_index(drop=True)

@st.cache_data
def load_labels():
    return pd.read_csv(BASE + "topic_labels.csv")

@st.cache_data
def load_trends():
    return pd.read_csv(BASE + "emerging_trends_results.csv")

@st.cache_resource
def load_model():
    with open(BASE + "lda_model.pkl", "rb") as f:
        lda = pickle.load(f)
    with open(BASE + "vectorizer.pkl", "rb") as f:
        cv = pickle.load(f)
    return lda, cv

@st.cache_data(ttl=3600)
def load_live():
    url = "https://raw.githubusercontent.com/saqibbostan83-ui/hybrid-ml-trend-detection/main/live_predictions.csv"
    try:
        return pd.read_csv(url)
    except:
        return None

with st.spinner("Loading system..."):
    df     = load_data()
    ts     = load_time_series()
    labels = load_labels()
    trends = load_trends()
    lda, cv = load_model()

topic_names = dict(zip(labels["topic_id"], labels["label"]))
topic_words = dict(zip(labels["topic_id"], labels["top_words"]))
topic_cols  = [c for c in ts.columns if c.startswith("topic_")]

# SIDEBAR
st.sidebar.title("🚀 Trend Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Overview",
    "🔍 Topic Explorer",
    "🚀 Emerging Trends",
    "🔴 Live Data"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📅 Date Filter")
min_date   = ts["week"].min().date()
max_date   = ts["week"].max().date()
date_range = st.sidebar.date_input("Select Range", value=(min_date, max_date),
                                    min_value=min_date, max_value=max_date)
st.sidebar.markdown("---")
st.sidebar.info(f"📝 {len(df):,} comments\n\n📅 {len(ts)} weeks\n\n🧠 {lda.n_components} topics\n\n🌐 r/worldnews")

if len(date_range) == 2:
    s, e = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    ts_f = ts[(ts["week"] >= s) & (ts["week"] <= e)].copy()
else:
    ts_f = ts.copy()

def make_wordcloud(words_str, title):
    wc = WordCloud(width=800, height=380, background_color="white",
                   colormap="viridis", max_words=50).generate(words_str.replace(",", " "))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════
if page == "📊 Overview":
    st.title("🌐 Hybrid ML Trend Detection System")
    st.caption("LDA Topic Modelling + Change Point Detection on Reddit Discussions")
    st.markdown("---")

    emerging_count = len(trends[trends["status"] == "🚀 EMERGING"])
    growing_count  = len(trends[trends["status"] == "📈 GROWING"])
    top_col = ts_f[topic_cols].sum().idxmax()
    top_id  = int(top_col.split("_")[1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📝 Total Comments",  f"{len(df):,}")
    c2.metric("📅 Weeks Analysed",  f"{len(ts_f)}")
    c3.metric("🚀 Trending Topics", f"{emerging_count + growing_count}")
    c4.metric("🏆 Top Topic",       topic_names[top_id])
    st.markdown("---")

    # Topic volume bar chart
    st.subheader("📊 Total Discussion Volume per Topic")
    totals = ts_f[topic_cols].sum()
    totals.index = [topic_names[int(c.split("_")[1])] for c in totals.index]
    totals = totals.sort_values(ascending=True)
    fig_bar = go.Figure(go.Bar(
        x=totals.values, y=totals.index, orientation="h",
        marker=dict(color=totals.values, colorscale="Viridis", showscale=True),
        text=totals.values, textposition="outside"
    ))
    fig_bar.update_layout(height=420, xaxis_title="Total Posts", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Pie chart
    st.subheader("🥧 Topic Share Distribution")
    fig_pie = go.Figure(go.Pie(
        labels=totals.index,
        values=totals.values,
        hole=0.4,
        textinfo="label+percent"
    ))
    fig_pie.update_layout(height=420, showlegend=True)
    st.plotly_chart(fig_pie, use_container_width=True)

    # Heatmap
    st.subheader("🗓️ Weekly Activity Heatmap")
    ts_f2 = ts_f.copy()
    ts_f2["week_str"] = ts_f2["week"].dt.strftime("%Y-%m-%d")
    heat  = ts_f2[topic_cols].T
    heat.index   = [topic_names[int(c.split("_")[1])] for c in heat.index]
    heat.columns = ts_f2["week_str"].values
    fig_heat = go.Figure(go.Heatmap(
        z=heat.values, x=heat.columns, y=heat.index,
        colorscale="YlOrRd", colorbar=dict(title="Posts")
    ))
    fig_heat.update_layout(height=420, xaxis=dict(tickangle=45))
    st.plotly_chart(fig_heat, use_container_width=True)

    # All topics line chart
    st.subheader("📈 All Topics Over Time")
    fig_all = go.Figure()
    for col in topic_cols:
        tid = int(col.split("_")[1])
        fig_all.add_trace(go.Scatter(
            x=ts_f["week"], y=ts_f[col],
            mode="lines", name=topic_names[tid], line=dict(width=1.8)
        ))
    fig_all.update_layout(height=420, hovermode="x unified",
                          xaxis_title="Week", yaxis_title="Posts",
                          plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_all, use_container_width=True)

    # Topic decline chart
    st.subheader("📉 Topic Trend Direction")
    first_half = ts_f.iloc[:len(ts_f)//2][topic_cols].mean()
    second_half = ts_f.iloc[len(ts_f)//2:][topic_cols].mean()
    change = ((second_half - first_half) / (first_half + 1) * 100).round(1)
    change.index = [topic_names[int(c.split("_")[1])] for c in change.index]
    change = change.sort_values()
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in change.values]
    fig_dir = go.Figure(go.Bar(
        x=change.values, y=change.index,
        orientation="h", marker_color=colors,
        text=[f"{v:+.1f}%" for v in change.values],
        textposition="outside"
    ))
    fig_dir.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_dir.update_layout(
        title="Growth vs Decline (First Half vs Second Half)",
        xaxis_title="Change (%)", height=420,
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_dir, use_container_width=True)

# ══════════════════════════════════════════
# PAGE 2 — TOPIC EXPLORER
# ══════════════════════════════════════════
elif page == "🔍 Topic Explorer":
    st.title("🔍 Topic Explorer")
    st.markdown("Explore keywords and weekly trends for each topic.")
    st.markdown("---")

    options   = {f"Topic {i} — {topic_names[i]}": i for i in range(lda.n_components)}
    sel_label = st.selectbox("Select a Topic", list(options.keys()))
    sel_id    = options[sel_label]
    sel_col   = f"topic_{sel_id}"

    col_wc, col_kw = st.columns([1.2, 1])
    with col_wc:
        st.subheader("☁️ Word Cloud")
        st.pyplot(make_wordcloud(topic_words[sel_id], topic_names[sel_id]))
    with col_kw:
        st.subheader("📋 Keyword Importance")
        word_list = [w.strip() for w in topic_words[sel_id].split(",")]
        weights   = list(range(len(word_list), 0, -1))
        fig_kw = go.Figure(go.Bar(
            x=weights, y=word_list, orientation="h",
            marker=dict(color=weights, colorscale="Blues", showscale=False)
        ))
        fig_kw.update_layout(height=380, xaxis_title="Relative Importance",
                             plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_kw, use_container_width=True)

    st.markdown("---")

    # Time series + trend line
    st.subheader(f"📈 Weekly Trend — {topic_names[sel_id]}")
    x_num = np.arange(len(ts_f))
    y_val = ts_f[sel_col].values
    z = np.polyfit(x_num, y_val, 1) if len(x_num) > 1 else [0, 0]
    p = np.poly1d(z)
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts_f["week"], y=ts_f[sel_col],
        mode="lines+markers", name="Weekly Posts",
        line=dict(color="#1f77b4", width=2.5),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.1)"
    ))
    fig_ts.add_trace(go.Scatter(
        x=ts_f["week"], y=p(x_num), mode="lines",
        name="Trend", line=dict(color="red", width=2, dash="dash")
    ))
    fig_ts.update_layout(height=400, hovermode="x unified",
                         xaxis_title="Week", yaxis_title="Posts",
                         plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_ts, use_container_width=True)

    # Growth rate
    st.subheader("📊 Week-over-Week Growth Rate (%)")
    growth = ts_f[sel_col].pct_change() * 100
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in growth.fillna(0)]
    fig_gr = go.Figure(go.Bar(x=ts_f["week"], y=growth, marker_color=colors))
    fig_gr.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_gr.update_layout(height=350, xaxis_title="Week", yaxis_title="Growth (%)",
                         hovermode="x unified", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gr, use_container_width=True)

    # Confidence distribution
    st.subheader("🎯 Topic Confidence Distribution")
    cv_matrix = cv.transform(df["clean_text"].fillna(""))
    topic_dist = lda.transform(cv_matrix)
    conf = topic_dist[:, sel_id]
    fig_conf = go.Figure(go.Histogram(
        x=conf, nbinsx=30,
        marker_color="#1f77b4",
        opacity=0.8
    ))
    fig_conf.update_layout(
        title=f"How confident is the model when assigning '{topic_names[sel_id]}'?",
        xaxis_title="Confidence Score", yaxis_title="Number of Comments",
        height=350, plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # Topic similarity heatmap
    st.subheader("🔄 Topic Similarity Matrix")
    components = lda.components_
    norm = components / components.sum(axis=1, keepdims=True)
    similarity = np.dot(norm, norm.T)
    topic_lbls = [topic_names[i] for i in range(lda.n_components)]
    fig_sim = go.Figure(go.Heatmap(
        z=similarity,
        x=topic_lbls, y=topic_lbls,
        colorscale="Blues",
        text=np.round(similarity, 2),
        texttemplate="%{text}",
        colorbar=dict(title="Similarity")
    ))
    fig_sim.update_layout(
        title="How similar are topics to each other?",
        height=500, xaxis=dict(tickangle=45)
    )
    st.plotly_chart(fig_sim, use_container_width=True)

    # Compare all topics
    st.subheader("🔀 Compare with All Topics")
    fig_cmp = go.Figure()
    for col in topic_cols:
        tid    = int(col.split("_")[1])
        is_sel = (col == sel_col)
        fig_cmp.add_trace(go.Scatter(
            x=ts_f["week"], y=ts_f[col], mode="lines",
            name=topic_names[tid],
            opacity=1.0 if is_sel else 0.25,
            line=dict(width=3 if is_sel else 1)
        ))
    fig_cmp.update_layout(height=400, hovermode="x unified",
                          xaxis_title="Week", yaxis_title="Posts",
                          plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════
# PAGE 3 — EMERGING TRENDS
# ══════════════════════════════════════════
elif page == "🚀 Emerging Trends":
    st.title("🚀 Emerging Trends")
    st.markdown("Topics flagged by Change Point Detection as rapidly growing.")
    st.markdown("---")

    emerging = trends[trends["status"] == "🚀 EMERGING"]
    growing  = trends[trends["status"] == "📈 GROWING"]
    stable   = trends[trends["status"] == "➡️ STABLE"]

    m1, m2, m3 = st.columns(3)
    m1.metric("🚀 Emerging", len(emerging))
    m2.metric("📈 Growing",  len(growing))
    m3.metric("➡️ Stable",   len(stable))
    st.markdown("---")

    if len(emerging) > 0:
        st.subheader("🚀 Emerging Topics")
        cols = st.columns(min(len(emerging), 3))
        for i, (_, row) in enumerate(emerging.iterrows()):
            with cols[i % 3]:
                st.success(f"**🚀 {row['topic_label']}**\n\n"
                           f"📅 Since: {row['change_date']}\n\n"
                           f"📈 Growth: {row['growth_rate']:.2f}x\n\n"
                           f"Before: {row['before_avg']:.1f} → After: {row['after_avg']:.1f} posts/wk")

    if len(growing) > 0:
        st.subheader("📈 Growing Topics")
        cols = st.columns(min(len(growing), 3))
        for i, (_, row) in enumerate(growing.iterrows()):
            with cols[i % 3]:
                st.info(f"**📈 {row['topic_label']}**\n\n"
                        f"📅 Since: {row['change_date']}\n\n"
                        f"📊 Growth: {row['growth_rate']:.2f}x\n\n"
                        f"Before: {row['before_avg']:.1f} → After: {row['after_avg']:.1f} posts/wk")

    st.markdown("---")

    # Growth rate chart
    st.subheader("📊 Growth Rate — All Topics")
    latest = trends.drop_duplicates("topic_id", keep="last").sort_values("growth_rate", ascending=True)
    bar_colors = []
    for s in latest["status"]:
        if "EMERGING" in s: bar_colors.append("#e74c3c")
        elif "GROWING" in s: bar_colors.append("#3498db")
        else: bar_colors.append("#bdc3c7")

    fig_gr2 = go.Figure(go.Bar(
        x=latest["growth_rate"], y=latest["topic_label"],
        orientation="h", marker_color=bar_colors,
        text=[f"{v:.2f}x" for v in latest["growth_rate"]],
        textposition="outside"
    ))
    fig_gr2.add_vline(x=2.0, line_dash="dash", line_color="red",
                      annotation_text="Emerging threshold (2x)")
    fig_gr2.update_layout(height=450, xaxis_title="Growth Rate (x)",
                          plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_gr2, use_container_width=True)

    # Timeline chart
    st.subheader("📅 Trend Timeline")
    timeline_data = trends[trends["status"] != "➡️ STABLE"].copy()
    if len(timeline_data) > 0:
        timeline_data["change_date"] = pd.to_datetime(timeline_data["change_date"])
        timeline_data = timeline_data.sort_values("change_date")
        colors_tl = ["#e74c3c" if "EMERGING" in s else "#3498db" for s in timeline_data["status"]]
        fig_tl = go.Figure(go.Scatter(
            x=timeline_data["change_date"],
            y=timeline_data["topic_label"],
            mode="markers",
            marker=dict(size=timeline_data["growth_rate"]*8, color=colors_tl, opacity=0.8),
            text=[f"Growth: {g:.2f}x" for g in timeline_data["growth_rate"]],
            hovertemplate="<b>%{y}</b><br>Date: %{x}<br>%{text}<extra></extra>"
        ))
        fig_tl.update_layout(
            title="When did each trend start? (bubble size = growth rate)",
            height=420, plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Change Date"
        )
        st.plotly_chart(fig_tl, use_container_width=True)

    # Velocity chart
    st.subheader("⚡ Trend Velocity")
    vel_data = trends.drop_duplicates("topic_id", keep="last").copy()
    vel_data["velocity"] = (vel_data["after_avg"] - vel_data["before_avg"]).round(1)
    vel_data = vel_data.sort_values("velocity", ascending=True)
    vel_colors = ["#e74c3c" if v > 0 else "#95a5a6" for v in vel_data["velocity"]]
    fig_vel = go.Figure(go.Bar(
        x=vel_data["velocity"],
        y=vel_data["topic_label"],
        orientation="h",
        marker_color=vel_colors,
        text=[f"{v:+.1f} posts/wk" for v in vel_data["velocity"]],
        textposition="outside"
    ))
    fig_vel.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_vel.update_layout(
        title="Posts per week gained/lost after change point",
        height=420, xaxis_title="Velocity (posts/week)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_vel, use_container_width=True)

    # Full table
    st.subheader("📋 Full Results Table")
    disp = trends[["topic_label","change_date","growth_rate","before_avg","after_avg","status"]].copy()
    disp.columns = ["Topic","Change Date","Growth Rate","Before Avg","After Avg","Status"]
    disp["Growth Rate"] = disp["Growth Rate"].round(2)
    disp["Before Avg"]  = disp["Before Avg"].round(1)
    disp["After Avg"]   = disp["After Avg"].round(1)
    st.dataframe(disp, use_container_width=True, height=420)

# ══════════════════════════════════════════
# PAGE 4 — LIVE DATA
# ══════════════════════════════════════════
elif page == "🔴 Live Data":
    st.title("🔴 Live Data — Real Time Reddit")
    st.markdown("Live posts from 40+ subreddits — automatically updated every hour by GitHub Actions.")
    st.markdown("---")

    live = load_live()

    if live is not None:
        latest_time = live["timestamp"].max()
        latest = live[live["timestamp"] == latest_time].copy()
        latest = latest.sort_values("post_count", ascending=False)

        st.caption(f"🕐 Last updated: {latest_time}")

        # Metric cards
        total_live = latest["post_count"].sum()
        top_live = latest.iloc[0]["topic"]
        c1, c2, c3 = st.columns(3)
        c1.metric("📝 Total Live Posts", f"{total_live}")
        c2.metric("🔥 Top Topic", top_live)
        c3.metric("⏱️ Updated", latest_time)
        st.markdown("---")

        # Live bar chart
        st.subheader("📊 Live Topic Activity")
        fig_live = go.Figure(go.Bar(
            x=latest["post_count"],
            y=latest["topic"],
            orientation="h",
            marker=dict(color=latest["post_count"], colorscale="Reds", showscale=True),
            text=latest["post_count"],
            textposition="outside"
        ))
        fig_live.update_layout(
            title="Posts per Topic — Last Hour",
            xaxis_title="Posts", height=420,
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_live, use_container_width=True)

        # Live pie chart
        st.subheader("🥧 Live Topic Share")
        fig_lpie = go.Figure(go.Pie(
            labels=latest["topic"],
            values=latest["post_count"],
            hole=0.4,
            textinfo="label+percent"
        ))
        fig_lpie.update_layout(height=420)
        st.plotly_chart(fig_lpie, use_container_width=True)

        # Historical trend
        if len(live["timestamp"].unique()) > 1:
            st.subheader("📈 Live Trend History")
            fig_hist = go.Figure()
            for topic in live["topic"].unique():
                td = live[live["topic"] == topic].sort_values("timestamp")
                fig_hist.add_trace(go.Scatter(
                    x=td["timestamp"], y=td["post_count"],
                    mode="lines+markers", name=topic, line=dict(width=2)
                ))
            fig_hist.update_layout(
                title="Topic Activity Over Time (Live Updates)",
                height=420, hovermode="x unified",
                xaxis_title="Time", yaxis_title="Posts",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Hourly heatmap
            st.subheader("🌡️ Hourly Activity Heatmap")
            live_pivot = live.pivot_table(
                index="topic", columns="timestamp",
                values="post_count", fill_value=0
            )
            fig_lheat = go.Figure(go.Heatmap(
                z=live_pivot.values,
                x=live_pivot.columns,
                y=live_pivot.index,
                colorscale="Reds",
                colorbar=dict(title="Posts")
            ))
            fig_lheat.update_layout(
                title="Topic × Time Heatmap",
                height=420, xaxis=dict(tickangle=45)
            )
            st.plotly_chart(fig_lheat, use_container_width=True)

            # Topic momentum
            st.subheader("⚡ Live Topic Momentum")
            momentum = []
            for topic in live["topic"].unique():
                td = live[live["topic"] == topic].sort_values("timestamp")
                if len(td) >= 2:
                    change = td["post_count"].iloc[-1] - td["post_count"].iloc[-2]
                    momentum.append({"topic": topic, "change": change})

            if momentum:
                mom_df = pd.DataFrame(momentum).sort_values("change", ascending=True)
                mom_colors = ["#e74c3c" if v > 0 else "#3498db" for v in mom_df["change"]]
                fig_mom = go.Figure(go.Bar(
                    x=mom_df["change"],
                    y=mom_df["topic"],
                    orientation="h",
                    marker_color=mom_colors,
                    text=[f"{v:+d}" for v in mom_df["change"]],
                    textposition="outside"
                ))
                fig_mom.add_vline(x=0, line_dash="dash", line_color="gray")
                fig_mom.update_layout(
                    title="Which topics are rising vs falling right now?",
                    height=420, xaxis_title="Change in posts",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_mom, use_container_width=True)

        # Raw data table
        st.subheader("📋 Live Raw Data")
        st.dataframe(live.sort_values("timestamp", ascending=False),
                     use_container_width=True, height=350)

    else:
        st.info("⏳ Live data loading... Check back in a few minutes.")
