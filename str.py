import streamlit as st
import requests
from typing import Optional
from st_aggrid import AgGrid
import pandas as pd
from pyvis.network import Network
import networkx as nx

# Set up FastAPI endpoint
API_BASE_URL = "http://localhost:8008"

# Helper functions for API calls
def index_documents(folder_path: Optional[str] = None):
    try:
        response = requests.post(f"{API_BASE_URL}/index/", data={"folder_path": folder_path})
        return response.json()
    except Exception as e:
        st.error(f"Error indexing documents: {e}")
        return {}

def list_entities():
    try:
        response = requests.get(f"{API_BASE_URL}/list_entities/")
        return response.json()
    except Exception as e:
        st.error(f"Error listing entities: {e}")
        return {}

def query_neo4j(query: str):
    try:
        response = requests.post(f"{API_BASE_URL}/query/", data={"query": query})
        return response.json()
    except Exception as e:
        st.error(f"Error querying Neo4j: {e}")
        return {}

def hybrid_search(query: str):
    try:
        response = requests.post(f"{API_BASE_URL}/search/", data={"query": query})
        return response.json()
    except Exception as e:
        st.error(f"Error with hybrid search: {e}")
        return {}

def chat_with_session(query: str, session_id: str):
    try:
        response = requests.post(f"{API_BASE_URL}/chat/", data={"query": query, "session_id": session_id})
        return response.json()
    except Exception as e:
        st.error(f"Error in chat session: {e}")
        return {}

def cypher_query(query: str):
    try:
        response = requests.post(f"{API_BASE_URL}/cypher_query/", data={"query": query})
        return response.json()
    except Exception as e:
        st.error(f"Error executing Cypher query: {e}")
        return {}

def get_graph_data():
    try:
        response = requests.get(f"{API_BASE_URL}/graph_data/")
        return response.json()
    except Exception as e:
        st.error(f"Error fetching graph data: {e}")
        return {}

# Streamlit UI setup
st.title("Comprehensive Document Retrieval and Graph Analysis")

# Sidebar navigation
st.sidebar.header("Options")
page = st.sidebar.selectbox("Choose an action", ["Index Documents", "List Entities", "Query Entities", "Hybrid Search", "Chat", "Cypher Query", "View Entity Relationships"])

# Set session ID for chat persistence
if "session_id" not in st.session_state:
    st.session_state.session_id = st.sidebar.text_input("Session ID", "default_session")

# Index Documents
if page == "Index Documents":
    st.header("Upload and Index Documents")
    folder_path = st.text_input("Enter folder path for PDF documents:")
    if st.button("Index Documents"):
        with st.spinner("Indexing documents..."):
            response = index_documents(folder_path)
            st.success("Documents indexed successfully!" if "message" in response else "Failed to index documents.")
            st.json(response)

# List Entities
elif page == "List Entities":
    st.header("List Entities in Database")
    response = list_entities()
    if "entities" in response:
        st.success("Entities retrieved successfully.")
        AgGrid(pd.DataFrame(response["entities"]))
    else:
        st.error("Failed to retrieve entities.")
        st.json(response)

# Query Entities
elif page == "Query Entities":
    st.header("Query Entities in Database")
    query = st.text_input("Enter entity query:")
    if st.button("Search Entities"):
        with st.spinner("Querying database..."):
            response = query_neo4j(query)
            if "entities" in response:
                st.success("Query completed.")
                AgGrid(pd.DataFrame(response["entities"]))
            else:
                st.error("Failed to retrieve query results.")
                st.json(response)

# Hybrid Search
elif page == "Hybrid Search":
    st.header("Execute Hybrid Search")
    search_query = st.text_input("Enter search query:")
    if st.button("Execute Search"):
        with st.spinner("Searching..."):
            results = hybrid_search(search_query)
            if "results" in results:
                st.success("Search completed.")
                st.write("Results:")
                for result in results["results"]:
                    st.write(result)
            else:
                st.error("Failed to retrieve search results.")
                st.json(results)

# Chat with Session-Based Memory
elif page == "Chat":
    st.header("Conversational Chat with Memory")
    query = st.text_input("Ask a question or continue the conversation:")
    if st.button("Send"):
        with st.spinner("Chatting..."):
            response = chat_with_session(query, st.session_state.session_id)
            if "response" in response:
                st.write("**Response:**")
                st.write(response["response"])
            else:
                st.error("Failed to get a response from chat.")
                st.json(response)

# Cypher Query Execution
elif page == "Cypher Query":
    st.header("Execute Cypher Query")
    cypher_query_input = st.text_area("Enter your Cypher query:")
    if st.button("Run Cypher Query"):
        with st.spinner("Executing Cypher query..."):
            response = cypher_query(cypher_query_input)
            if "response" in response:
                st.success("Query executed successfully.")
                st.json(response["response"])
            else:
                st.error("Failed to execute Cypher query.")
                st.json(response)

# View Entities and Relations
elif page == "View Entity Relationships":
    st.header("Entity Relationships Network")

    # Fetch entities and relationships data
    graph_data = get_graph_data()
    if "graph_data" in graph_data:
        entities = graph_data["graph_data"]

        # Display entities in a table format
        st.subheader("Entities and Relationships")
        entity_data = list_entities()
        if "entities" in entity_data:
            st.success("Entities retrieved successfully.")
            AgGrid(pd.DataFrame(entity_data["entities"]))

        # Build graph using NetworkX and Pyvis for visualization
        G = nx.DiGraph()
        for relation in entities:
            G.add_node(relation["source"], label=relation["source_type"])
            G.add_node(relation["target"], label=relation["target_type"])
            G.add_edge(relation["source"], relation["target"], label=relation["relationship"])

        # Pyvis network setup
        net = Network(height="600px", width="100%", notebook=True)
        net.from_nx(G)
        net.show_buttons(filter_=['physics'])

        # Display interactive graph
        net.show("network.html")
        with open("network.html", "r") as f:
            html = f.read()
            st.components.v1.html(html, height=600)

# Display Prometheus metrics
st.sidebar.header("Metrics")
metrics_url = f"{API_BASE_URL}/metrics"
if st.sidebar.button("Refresh Metrics"):
    try:
        metrics_response = requests.get(metrics_url)
        if metrics_response.status_code == 200:
            st.sidebar.text_area("Metrics", metrics_response.text, height=300)
        else:
            st.sidebar.error("Failed to retrieve metrics.")
    except Exception as e:
        st.sidebar.error(f"Error fetching metrics: {e}")
