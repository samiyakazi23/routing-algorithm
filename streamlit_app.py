# streamlit_app.py
"""
Routing Algorithm Visualizer — Streamlit (deployment-ready)
For repo: samiyakazi23/routing-algorithm
Main file: streamlit_app.py

Features:
 - Random or uploaded graph visualization
 - Dijkstra & Bellman-Ford algorithms
 - Step-by-step animation with color transitions
 - Light/Dark themes
 - Safe for Streamlit Cloud (no file writes)
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import heapq
import json
import random
import time

# ==========================
# Helpers: Graph generation
# ==========================
def build_random_graph(n_nodes=8, edge_prob=0.35, weight_range=(1, 20), seed=None):
    if seed is not None:
        random.seed(seed)
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_prob:
                w = random.randint(*weight_range)
                G.add_edge(i, j, weight=w)
    # connect disconnected components
    comps = list(nx.connected_components(G))
    for k in range(len(comps) - 1):
        a = random.choice(list(comps[k]))
        b = random.choice(list(comps[k + 1]))
        G.add_edge(a, b, weight=random.randint(*weight_range))
    return G


def graph_to_json(G):
    data = {"nodes": list(G.nodes()), "edges": []}
    for u, v, d in G.edges(data=True):
        data["edges"].append({"u": int(u), "v": int(v), "weight": float(d.get("weight", 1))})
    return json.dumps(data)


def json_to_graph(js):
    data = json.loads(js)
    G = nx.Graph()
    for n in data.get("nodes", []):
        G.add_node(int(n))
    for e in data.get("edges", []):
        G.add_edge(int(e["u"]), int(e["v"]), weight=float(e.get("weight", 1)))
    return G


# ==========================
# Algorithms producing steps
# ==========================
def dijkstra_steps(G, source):
    dist = {v: math.inf for v in G.nodes()}
    prev = {v: None for v in G.nodes()}
    dist[source] = 0
    visited = set()
    heap = [(0, source)]
    steps = []

    def record(action, u=None, v=None):
        steps.append({
            "action": action,
            "u": u,
            "v": v,
            "dist": dict(dist),
            "prev": dict(prev),
            "visited": set(visited)
        })

    record("init")
    while heap:
        d, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)
        record("visit_node", u=u)
        for v in G.neighbors(u):
            w = G[u][v]["weight"]
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(heap, (dist[v], v))
                record("relax_edge", u=u, v=v)
    record("done")
    return steps


def bellman_ford_steps(G, source):
    nodes = list(G.nodes())
    dist = {v: math.inf for v in nodes}
    prev = {v: None for v in nodes}
    dist[source] = 0
    steps = []

    def record(action, u=None, v=None):
        steps.append({
            "action": action,
            "u": u,
            "v": v,
            "dist": dict(dist),
            "prev": dict(prev)
        })

    record("init")
    n = len(nodes)
    edges = [(u, v, G[u][v]["weight"]) for u, v in G.edges()]
    for i in range(n - 1):
        any_change = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                any_change = True
                record("relax_edge", u=u, v=v)
            if dist[v] + w < dist[u]:
                dist[u] = dist[v] + w
                prev[u] = v
                any_change = True
                record("relax_edge", u=v, v=u)
        if not any_change:
            break
        record("iteration_end")
    record("done")
    return steps


# ==========================
# Visualization helpers
# ==========================
def extract_path(prev, src, dst):
    if prev.get(dst) is None:
        return None
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        if cur == src:
            break
        cur = prev.get(cur)
    if len(path) == 0 or path[-1] != src:
        return None
    return path[::-1]


def draw_state(G, pos, state, source, target, theme="light", figsize=(6, 4)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(f"Step: {state.get('action','')}")
    # theme colors
    if theme == "dark":
        ax.set_facecolor("#222222")
        node_color_default = "#555555"
        visited_color = "#1f9c74"
        visiting_color = "#f4d35e"
        edge_color_default = "#666666"
        text_color = "white"
    else:
        ax.set_facecolor("white")
        node_color_default = "lightgray"
        visited_color = "lightgreen"
        visiting_color = "yellow"
        edge_color_default = "lightgray"
        text_color = "black"

    dist = state.get("dist", {})
    prev = state.get("prev", {})
    visited = state.get("visited", set())

    # nodes
    node_colors = []
    labels = {}
    for n in G.nodes():
        if state.get("action") == "visit_node" and state.get("u") == n:
            node_colors.append(visiting_color)
        elif n in visited:
            node_colors.append(visited_color)
        else:
            node_colors.append(node_color_default)
        d = dist.get(n, math.inf)
        labels[n] = f"{n}\n(∞)" if d == math.inf else f"{n}\n({int(d)})"

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, edgecolors="k", linewidths=0.6)
    nx.draw_networkx_labels(G, pos, labels, font_color=text_color)

    # edges
    last_u = state.get("u", None)
    last_v = state.get("v", None)
    for (u, v) in G.edges():
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        linewidth = 1.2
        color = edge_color_default
        if last_u is not None and last_v is not None:
            if (u == last_u and v == last_v) or (u == last_v and v == last_u):
                color = "orange" if state.get("action") == "relax_edge" else "red"
                linewidth = 3.0
        ax.plot(x, y, linewidth=linewidth, color=color, zorder=1)

    # draw shortest-path tree edges from prev
    for n in G.nodes():
        p = prev.get(n)
        if p is not None:
            x = [pos[n][0], pos[p][0]]
            y = [pos[n][1], pos[p][1]]
            ax.plot(x, y, linewidth=3.0, color="green", zorder=3)

    # final path highlight
    if state.get("action") == "done":
        path = extract_path(prev, source, target)
        if path:
            for a, b in zip(path[:-1], path[1:]):
                x = [pos[a][0], pos[b][0]]
                y = [pos[a][1], pos[b][1]]
                ax.plot(x, y, linewidth=5.0, color="blue", zorder=5)
            ax.text(0.02, 0.02, f"Final path: {' -> '.join(map(str,path))}\nDistance: {dist.get(target,'∞')}",
                    transform=ax.transAxes, bbox=dict(boxstyle="round", fc="wheat", alpha=0.6))

    ax.set_axis_off()
    st.pyplot(plt.gcf())
    plt.close()


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Routing Visualizer", layout="wide")
st.title("Routing Algorithm Visualizer — Streamlit")

left_col, right_col = st.columns([2, 1])

with left_col:
    st.header("Graph Setup")
    graph_source = st.radio("Create graph:", ("Random", "Upload JSON"), index=0, horizontal=True)
    if graph_source == "Random":
        n_nodes = st.slider("Nodes", 5, 20, 8)
        edge_prob = st.slider("Edge probability (%)", 10, 70, 35) / 100.0
        seed = st.number_input("Random seed (optional)", value=7, step=1)
        if st.button("Generate Random Graph"):
            G = build_random_graph(n_nodes=n_nodes, edge_prob=edge_prob, seed=seed)
            st.session_state["graph_json"] = graph_to_json(G)
    else:
        uploaded = st.file_uploader("Upload graph JSON", type=["json"])
        if uploaded is not None:
            content = uploaded.read().decode("utf-8")
            try:
                G = json_to_graph(content)
                st.session_state["graph_json"] = content
                st.success("Graph uploaded")
            except Exception:
                st.error("Invalid JSON format")

    if "graph_json" in st.session_state:
        G = json_to_graph(st.session_state["graph_json"])
    else:
        G = build_random_graph(n_nodes=8, edge_prob=0.35, seed=7)
        st.session_state["graph_json"] = graph_to_json(G)

    st.write("Graph nodes:", list(G.nodes()))
    pos = nx.spring_layout(G, seed=3)


with right_col:
    st.header("Controls & Run")
    algo = st.selectbox("Algorithm", ("Dijkstra", "Bellman-Ford"))
    source = st.selectbox("Source node", list(G.nodes()), index=0)
    target = st.selectbox("Target node", list(G.nodes()), index=len(list(G.nodes()))-1)
    theme = st.selectbox("Theme", ("Light", "Dark"))

    st.write("Animation controls:")
    c1, c2, c3 = st.columns(3)
    with c1:
        run_btn = st.button("Run ▶")
    with c2:
        step_btn = st.button("Step ⏩")
    with c3:
        back_btn = st.button("Back ⏪")
    pause_checkbox = st.checkbox("Pause autoplay (stop)", value=False)
    speed = st.slider("Animation speed (ms per step)", 100, 1000, 500, step=50)


# ==========================
# Session state housekeeping
# ==========================
if "steps" not in st.session_state:
    st.session_state["steps"] = []
if "step_index" not in st.session_state:
    st.session_state["step_index"] = 0
if "algo_last" not in st.session_state:
    st.session_state["algo_last"] = None
if "auto_play" not in st.session_state:
    st.session_state["auto_play"] = False


# prepare steps if run pressed or algo changed
if run_btn or (st.session_state.get("algo_last") != algo):
    st.session_state["algo_last"] = algo
    if algo == "Dijkstra":
        st.session_state["steps"] = dijkstra_steps(G, source)
    else:
        st.session_state["steps"] = bellman_ford_steps(G, source)
    st.session_state["step_index"] = 0
    st.session_state["auto_play"] = True


# step forward/back logic
if step_btn:
    if st.session_state["step_index"] < len(st.session_state["steps"]) - 1:
        st.session_state["step_index"] += 1
    st.session_state["auto_play"] = False

if back_btn:
    if st.session_state["step_index"] > 0:
        st.session_state["step_index"] -= 1
    st.session_state["auto_play"] = False


# Auto-play loop executed by re-renders (safe approach)
if st.session_state.get("auto_play", False):
    steps = st.session_state["steps"]
    idx = st.session_state["step_index"]
    if not pause_checkbox:
        if idx < len(steps) - 1:
            st.session_state["step_index"] = idx + 1
            time.sleep(max(0.01, speed / 1000.0))
            st.experimental_rerun()
        else:
            st.session_state["auto_play"] = False


# render current state
if st.session_state["steps"]:
    idx = st.session_state["step_index"]
    state = st.session_state["steps"][idx]
    draw_state(G, pos, state, source, target, theme.lower())


# Footer: export graph JSON (download only)
st.markdown("---")
colx, coly = st.columns([1, 3])
with colx:
    st.download_button("Download graph JSON", st.session_state["graph_json"],
                       file_name="graph.json", mime="application/json")
with coly:
    st.markdown("**Tips:** Use Step or Run buttons to animate algorithm steps. Upload your own graph as JSON.")
