from typing import Dict, Union
from typing import Optional 
import pinecone
from langchain.vectorstores import Pinecone
from langchain_chroma.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client
from chromadb.api.models.Collection import Collection

from langchain_chroma.vectorstores import Chroma

# Function to detect prompt injection using vector database
def detect_pi_using_vector_database(
    input: str, similarity_threshold: float, vector_store: Union[Pinecone, Chroma]
) -> Dict:
    """
    Detects Prompt Injection using similarity search with vector database.
    """
    top_k = 20
    
    # Ensure we know the type of the vector store
    print(f"Vector Store Type: {type(vector_store)}")

    # Check the type of the vector store and perform the search accordingly
    if isinstance(vector_store, Pinecone):
        results = vector_store.similarity_search_with_score(input, top_k)
    elif isinstance(vector_store, Collection):
        # Perform Chroma-specific query
        results = vector_store.query(
            query_texts=[input], n_results=top_k
        )
        results = [
            (doc, score) for doc, score in zip(
                results["documents"][0], results["distances"][0]
            )
        ]
    else:
        raise ValueError("Unsupported vector store type.")

    # Process the results...
    top_score = 0
    count_over_max_vector_score = 0

    for _, score in results:
        if score is None:
            continue

        if score > top_score:
            top_score = score

        if score >= similarity_threshold and score > top_score:
            count_over_max_vector_score += 1

    return {
        "top_score": top_score,
        "count_over_max_vector_score": count_over_max_vector_score,
    }


# Initialize Pinecone vector store
def init_pinecone(api_key: str, index: str, openai_api_key: str = None, use_ollama: bool = False) -> Pinecone:
    """
    Initializes connection with the Pinecone vector database using existing (rebuff) index.

    Args:
        api_key (str): Pinecone API key
        index (str): Pinecone index name
        openai_api_key (str): Open AI API key
        use_ollama (bool): Whether to use Ollama
    Returns:
        vector_store (Pinecone)

    """
    if not api_key:
        raise ValueError("Pinecone apikey definition missing")

    pc = pinecone.Pinecone(api_key=api_key)
    pc_index = pc.Index(index)

    # Choose embedding model based on the `use_ollama` flag
    # Using Ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Pinecone(pc_index, embeddings, text_key="input")

    return vector_store


# Initialize Chroma vector store
def init_chroma(
    collection_name: str,
    embedding_model: Optional[Union[OpenAIEmbeddings, OllamaEmbeddings]] = None,
    use_ollama: bool = False,
    openai_api_key: Optional[str] = None
) -> Chroma:
    """
    Initializes connection with the Chroma vector database.

    Args:
        collection_name (str): The name of the Chroma collection
        embedding_model (Optional[Embeddings]): The embedding model to use (default: OllamaEmbeddings)

    Returns:
        Chroma: Vector store connected to ChromaDB
    """
    client = Client(Settings())

    # Default to OllamaEmbeddings if none is provided
    embeddings = embedding_model or OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma(
        collection_name=collection_name,
        client=client,
        embedding_function=embeddings
    )

    return vector_store
