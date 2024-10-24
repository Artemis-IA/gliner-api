from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor

# Step 1: Document Loading and Splitting
DATA_PATH = "/home/pi/Documents/IF-SRV/4pdfs_subset/"

# Load documents
loader = PyPDFDirectoryLoader(DATA_PATH)
documents = loader.load()

# Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_documents = text_splitter.split_documents(documents)

# Step 2: Initialize the GLiNER NER extractor for entity linking
gliner_extractor = GLiNERLinkExtractor(
    labels = [
    "Board of Directors", 
    "Audit Committee", 
    "Chief Financial Officer (CFO)", 
    "Chief Sustainability Officer", 
    "Governance and Compliance Committee", 
    "ESG Funds", 
    "Risk Manager", 
    "Institutional Investors", 
    "Audit Firms", 
    "Financial Regulatory Authorities", 
    "Sustainability Committee", 
    "Accounting Standards Organizations", 
    "Impact Investment Funds", 
    "Activist Shareholders", 
    "Corporate Governance Consultants", 
    "ESG Certification Bodies", 
    "Investor Relations Managers", 
    "Consumer Advocacy Groups", 
    "ESG Data Providers", 
    "Financial Analysts"
    ],
    model="urchade/gliner_mediumv2.1"
)

# Step 3: Store the document embeddings in FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(split_documents, embedding_model)
vector_store.save_local("faiss_index")

# Step 4: Setup Meta Llama 3.2 3B Model for Question Answering
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Llama model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create HuggingFace pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, truncation=True, max_new_tokens=100, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# Step 5: NER Context Enrichment using GLiNER
def enrich_with_ner(text):
    # Extract entities using GLiNER
    extracted_entities = gliner_extractor.extract_one(text)
    # Build additional context with the extracted entities
    additional_context = " ".join([f"{entity.tag} ({entity.kind})" for entity in extracted_entities])
    return additional_context

# Step 6: Retrieval and Augmenting Llama with the Retrieved Context
def retrieve_context(query):
    # Perform retrieval from FAISS vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    results = retriever.get_relevant_documents(query)
    # Combine results into a context string
    context = " ".join([doc.page_content for doc in results])
    return context

# Create the prompt template
template = """Given the following information:

Context: {context}

Please answer the user's question: {user_input}
"""
prompt = PromptTemplate(input_variables=["context", "user_input"], template=template)

# Combine the Llama model and the prompt in a RunnableSequence
chain = prompt | llm

# Step 7: Execute the Complete RAG Process
def run_chain(user_input):
    # Retrieve context from the vector store
    context = retrieve_context(user_input)
    # Enrich the context with NER information using GLiNER
    enriched_context = enrich_with_ner(context)
    # Run the chain to generate the final response
    response = chain.invoke({"context": enriched_context, "user_input": user_input})
    return response

# Example Usage
user_prompt = """
Quels sont les clients et les fournisseurs de ces entreprises, ainsi que les flux qu'elles ont échangé ? 
"""
response = run_chain(user_prompt)
print(response)