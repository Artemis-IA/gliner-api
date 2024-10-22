from langchain_community.graph_vectorstores.extractors import GLiNERLinkExtractor
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Initialize the GLiNER NER extractor
gliner_extractor = GLiNERLinkExtractor(
    labels=["Person", "Organization", "Product"],  # Example of entity types to extract
    model="urchade/gliner_mediumv2.1"
)

# Load the Mistral model using Hugging Face
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # You can replace this with any other model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a HuggingFace pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

# Initialize LangChain's HuggingFacePipeline with the Mistral model
llm = HuggingFacePipeline(pipeline=pipe)

# Define a function to extract entities and add to context
def enrich_with_ner(text):
    # Extract entities using GLiNER
    extracted_entities = gliner_extractor.extract_one(text)
    # Build additional context with the extracted entities
    additional_context = " ".join([f"{entity.tag} ({entity.kind})" for entity in extracted_entities])
    return additional_context

# Create a LangChain prompt template
template = """Given the following information:

Context: {context}

Please answer the user's question: {user_input}
"""

# Initialize a prompt template with LangChain
prompt = PromptTemplate(input_variables=["context", "user_input"], template=template)

# Combine everything in a chain
def run_chain(user_input):
    # Enrich the context with NER information
    context = enrich_with_ner(user_input)
    # Prepare the input for LLM
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    # Generate the LLM response
    return llm_chain.run({"context": context, "user_input": user_input})

# Example usage

response = run_chain("Tell me about the contributions of Donald Trump.")
print(response)
