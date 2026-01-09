import numpy as np
from tqdm import tqdm
from typing import List, Union, Optional, Literal
import torch
from dotenv import load_dotenv

# OpenAI API
from openai import OpenAI

# Sentence Transformers
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL_POOL = [
    "NovaSearch/stella_en_400M_v5",
    "text-embedding-3-large",
    "text-embedding-3-small",
    "multi-qa-mpnet-base-dot-v1",
    "nvidia/NV-Embed-v2",
    "bm25",
]


class EmbeddingModel:
    """
    Unified embedding model class that supports both OpenAI and SentenceTransformer models
    """

    def __init__(
        self,
        model_type: Literal["openai", "sentence_transformer"],
        model_name: str = None,
        api_key: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the embedding model

        Args:
            model_type: Type of model to use ("openai" or "sentence_transformer")
            model_name: Name of the model to use
                        - For OpenAI: "text-embedding-3-large", "text-embedding-3-small", etc.
                        - For SentenceTransformer: "multi-qa-mpnet-base-dot-v1", "all-mpnet-base-v2", etc.
            api_key: API key for OpenAI (only needed for OpenAI models)
            device: Device to use for computation ("cpu" or "cuda")
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device

        self._model = SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )

    def create(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        dimensions: Optional[int] = None,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Create embeddings for the given texts

        Args:
            texts: A string or list of strings to embed
            batch_size: Batch size for processing
                        - Default for OpenAI: 1536
                        - Default for SentenceTransformer: 32
            dimensions: Dimension of the embeddings (only used for OpenAI models)
            show_progress_bar: Whether to show a progress bar
            **kwargs: Additional arguments passed to the underlying embedding method

        Returns:
            numpy.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Replace empty strings with space to prevent errors
        texts = [text if text != "" else " " for text in texts]

        # Set default batch size based on model type
        if batch_size is None:
            batch_size = 1536 if self.model_type == "openai" else 32

        if self.model_type == "openai":
            return self._create_openai_embeddings(
                texts, batch_size, dimensions, show_progress_bar, **kwargs
            )
        else:  # sentence_transformer
            return self._create_st_embeddings(texts, batch_size, show_progress_bar, **kwargs)

    def _create_openai_embeddings(
        self, texts, batch_size, dimensions, show_progress_bar, **kwargs
    ):
        """Create embeddings using OpenAI API"""
        output = []
        for batch_texts in self.batch_generator(texts, batch_size, show_progress_bar):
            embedding_args = {
                "input": batch_texts,
                "model": self.model_name,
            }

            # Add dimensions if specified
            if dimensions is not None:
                embedding_args["dimensions"] = dimensions

            # Add any additional kwargs
            embedding_args.update(kwargs)

            response = self._client.embeddings.create(**embedding_args)
            response = [data.embedding for data in response.data]
            output.extend(response)

        return np.array(output)

    def _create_st_embeddings(self, texts, batch_size, show_progress_bar, **kwargs):
        """Create embeddings using SentenceTransformer"""
        output = []
        for batch_texts in self.batch_generator(texts, batch_size, show_progress_bar):
            embeddings = self._model.encode(
                batch_texts,
                show_progress_bar=False,  # We're handling our own progress bar
                convert_to_numpy=True,
                **kwargs,
            )
            embeddings = self.normalize_embeddings(embeddings)
            output.extend(embeddings)

        return np.array(output)

    def batch_generator(self, texts, batch_size, show_progress_bar=False):
        """
        Generate batches of texts

        Args:
            texts: List of texts to batch
            batch_size: Size of each batch
            show_progress_bar: Whether to show a progress bar

        Yields:
            List of texts for each batch
        """
        total_batches = int(np.ceil(len(texts) / batch_size))
        if show_progress_bar:
            model_name = (
                self.model_name.split("/")[-1] if "/" in self.model_name else self.model_name
            )
            batch_range = tqdm(range(total_batches), desc=f"Creating {model_name} embeddings")
        else:
            batch_range = range(total_batches)

        for batch in batch_range:
            batch_start = batch * batch_size
            batch_end = min(batch_start + batch_size, len(texts))
            yield texts[batch_start:batch_end]

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model

        Returns:
            int: Dimension of the embeddings

        Note: For OpenAI models, this returns None unless dimensions was specified during creation
        """
        if self.model_type == "sentence_transformer":
            return self._model.get_sentence_embedding_dimension()
        else:
            # OpenAI models have configurable dimensions
            return None

    def normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize the embeddings to have unit length.

        Args:
            embeddings: A numpy array of embeddings to normalize.

        Returns:
            numpy.ndarray: Normalized embeddings.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        return normalized_embeddings
