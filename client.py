import streamlit as st
import requests
import networkx as nx
from pyvis.network import Network
import json

st.set_page_config(page_title="Neo4j Graph Visualization", layout="wide")
st.title("Neo4j Entity Graph Visualization")

# Fetch graph data from FastAPI
@st.cache_data
def fetch_graph_data():
    response = requests.get("http://localhost:8008/graph_data/")
    if response.status_code == 200:
        return response.json().get("graph_data", [])
    else:
        st.error("Failed to fetch graph data from API")
        return []

graph_data = fetch_graph_data()

# Create a NetworkX graph from the fetched data
G = nx.Graph()
for record in graph_data:
    source = record["source"]
    target = record["target"]
    relationship = record["relationship"]

    # Add nodes and edge
    G.add_node(source, label=source, type=record["source_type"])
    G.add_node(target, label=target, type=record["target_type"])
    G.add_edge(source, target, label=relationship)

# Visualize using PyVis
net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
net.from_nx(G)

# Create HTML file for pyvis and render it
net.show("graph.html")
with open("graph.html", "r", encoding="utf-8") as f:
    html_content = f.read()
st.components.v1.html(html_content, height=800)
