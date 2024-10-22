import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt

# API URL
API_URL = "http://127.0.0.1:8008"

st.title("Entity Graph from PDFs")

# Query input
query = st.text_input("Enter query:")

if st.button("Search"):
    # Query FastAPI for entities from Neo4j
    response = requests.post(f"{API_URL}/query/", json={"query": query})
    entities = response.json().get("entities", [])

    if entities:
        st.write("Entities found:")
        G = nx.Graph()

        # Add nodes to the graph
        for entity in entities:
            G.add_node(entity["name"], label=entity["type"])
            st.write(f"Entity: {entity['name']}, Type: {entity['type']}")

        # Draw the graph
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_size=10)
        st.pyplot(plt)
    else:
        st.write("No entities found.")
