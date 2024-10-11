import secrets
from typing import Optional, Tuple, Union
import requests
from chromadb import Client, Settings
from langchain.embeddings.base import Embeddings
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel
from rebuff.detect_pi_heuristics import detect_prompt_injection_using_heuristic_on_input
from rebuff.detect_pi_vectorbase import detect_pi_using_vector_database, init_pinecone, init_chroma
from rebuff.detect_pi_ollama import render_prompt_for_pi_detection, call_ollama_to_detect_pi
from langchain.prompts import PromptTemplate

class RebuffDetectionResponse(BaseModel):
    heuristic_score: float
    vector_score: float
    llm_score: float
    run_heuristic_check: bool
    run_vector_check: bool
    run_language_model_check: bool
    max_heuristic_score: float
    max_llm_score: float
    max_vector_score: float
    injection_detected: bool

class RebuffSdk:
    def __init__(
        self,
        vector_store_type: str = "pinecone",  # Specify vector store: 'pinecone' or 'chroma'
        pinecone_apikey: Optional[str] = None,
        pinecone_index: Optional[str] = None,
        use_ollama: bool = True,  # Set Ollama as default
        ollama_model: str = "llama3.2",  # Default LLM for Ollama
        ollama_embed_model: str = "nomic-embed-text",  # Default embedding model for Ollama
        chroma_collection_name: str = "rebuff-collection",  # Default Chroma collection name
        embedding_model: Optional[Embeddings] = None  # Custom embedding model option
    ) -> None:
        self.vector_store_type = vector_store_type
        self.pinecone_apikey = pinecone_apikey
        self.pinecone_index = pinecone_index
        self.use_ollama = use_ollama
        self.ollama_model = ollama_model
        self.ollama_embed_model = ollama_embed_model
        self.chroma_collection_name = chroma_collection_name

        # Use the provided embedding model or fallback to Ollama or OpenAI
        self.embedding_model = embedding_model or OllamaEmbeddings(model=self.ollama_embed_model)

        # For vector stores, this will be initialized dynamically
        self.vector_store = None

    def call_ollama_embedding(self, text: str) -> list:
        """Calls the Ollama API to generate embeddings for the input text."""
        url = "http://localhost:11434/api/embedding"
        headers = {"Content-Type": "application/json"}
        data = {"model": self.ollama_embed_model, "text": text}

        response = requests.post(url, json=data, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result["embedding"]
        else:
            raise Exception(f"Error from Ollama embedding API: {response.status_code}, {response.text}")

    def initialize_pinecone(self) -> None:
        """Initializes Pinecone and sets up the vector store with embeddings."""
        if not self.pinecone_apikey or not self.pinecone_index:
            raise ValueError("Pinecone API key and index are required.")

        # Initialize Pinecone and set up the vector store
        vector_store = init_pinecone(
            api_key=self.pinecone_apikey,
            index=self.pinecone_index,
            use_ollama=self.use_ollama
        )
        self.vector_store = vector_store

    def initialize_chroma(self) -> None:
        """Initializes Chroma and sets up the collection."""
        client = Client(Settings())

        # Check if the collection exists, otherwise create it
        if self.chroma_collection_name not in [col.name for col in client.list_collections()]:
            collection = client.create_collection(name=self.chroma_collection_name)
        else:
            collection = client.get_collection(name=self.chroma_collection_name)

        self.vector_store = collection

    def detect_injection(
        self,
        user_input: str,
        max_heuristic_score: float = 0.75,
        max_vector_score: float = 0.90,
        max_llm_score: float = 0.90,
        check_heuristic: bool = True,
        check_vector: bool = True,
        check_llm: bool = True,
    ) -> RebuffDetectionResponse:
        """Detects if the given user input contains an injection attempt."""
        injection_detected = False

        if check_heuristic:
            heuristic_score = detect_prompt_injection_using_heuristic_on_input(user_input)
        else:
            heuristic_score = 0

        if check_vector:
            if self.vector_store_type == "pinecone":
                self.initialize_pinecone()
            elif self.vector_store_type == "chroma":
                self.initialize_chroma()

            # Debug print to ensure we know the vector store type
            print(f"Using vector store type: {type(self.vector_store)}")

            vector_score = detect_pi_using_vector_database(user_input, max_vector_score, self.vector_store)["top_score"]
        else:
            vector_score = 0

        if check_llm:
            rendered_input = render_prompt_for_pi_detection(user_input)
            model_response = call_ollama_to_detect_pi(rendered_input, self.ollama_model)
            try:
                llm_score = float(model_response.get("response", 0))
            except ValueError:
                print(f"Invalid response received from LLM: {model_response.get('response', 'None')}")
                llm_score = 0
        else:
            llm_score = 0

        if (
            heuristic_score > max_heuristic_score
            or llm_score > max_llm_score
            or vector_score > max_vector_score
        ):
            injection_detected = True

        return RebuffDetectionResponse(
            heuristic_score=heuristic_score,
            vector_score=vector_score,
            llm_score=llm_score,
            run_heuristic_check=check_heuristic,
            run_vector_check=check_vector,
            run_language_model_check=check_llm,
            max_heuristic_score=max_heuristic_score,
            max_llm_score=max_llm_score,
            max_vector_score=max_vector_score,
            injection_detected=injection_detected,
        )

    @staticmethod
    def generate_canary_word(length: int = 8) -> str:
        """Generates a secure random hexadecimal canary word."""
        return secrets.token_hex(length // 2)

    def add_canary_word(
        self,
        prompt: Union[str, PromptTemplate],
        canary_word: Optional[str] = None,
        canary_format: str = "<!-- {canary_word} -->",
    ) -> Tuple[Union[str, PromptTemplate], str]:
        """Adds a canary word to the given prompt."""
        if canary_word is None:
            canary_word = self.generate_canary_word()

        canary_comment = canary_format.format(canary_word=canary_word)

        if isinstance(prompt, str):
            prompt_with_canary = canary_comment + "\n" + prompt
            return prompt_with_canary, canary_word
        elif isinstance(prompt, PromptTemplate):
            prompt.template = canary_comment + "\n" + prompt.template
            return prompt, canary_word
        else:
            raise TypeError(f"prompt must be a str or PromptTemplate, but was {type(prompt)}")

    def is_canary_word_leaked(
        self,
        user_input: str,
        completion: str,
        canary_word: str,
        log_outcome: bool = True,
    ) -> bool:
        """Checks if the canary word is leaked in the completion."""
        if canary_word in completion:
            if log_outcome:
                self.log_leakage(user_input, completion, canary_word)
            return True
        return False

    def log_leakage(self, user_input: str, completion: str, canary_word: str) -> None:
        """Logs the leakage of a canary word."""
        if self.vector_store is None:
            self.initialize_pinecone()

        self.vector_store.add_texts(
            [user_input],
            metadatas=[{"completion": completion, "canary_word": canary_word}],
        )
