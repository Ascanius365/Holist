import os
import torch
import numpy as np
from premem_module.src.model.embedding import EmbeddingModel
from premem_module.src.utils.io import read_pickle, save_pickle
from nltk.tokenize import word_tokenize
from premem_module.src.memory import MemoryCompressor, ConversationSegmentor

import os
import sys

# Wir müssen die Pfade hinzufügen, damit Python die PREMem-Klassen findet
sys.path.append(".")
from premem_module.src.data.schema import Session

import os
import torch
import numpy as np
from tqdm import tqdm

def ensure_nltk_resources():
    import nltk

    resources = [
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("wordnet", "corpora/wordnet"),
        ("stopwords", "corpora/stopwords"),
    ]

    for resource_name, resource_path in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)


class EmbeddingDB:
    def __init__(
        self,
        dataset_name: str,
        embedding_model_name: str,
        mode: str,  # One of "session", "turn", "segment", "segment_compressed"
        model_type: str = "sentence_transformer",
        device: str = "cuda",
        base_cache_dir: str = ".cache",
        data_dir: str = "premem_module/dataset/processed",  # Default path where session pkl is stored
        batch_size: int = 128,
        show_progress_bar: bool = True,
        compress_rate: float = 0.9,
    ):

        self.dataset_name = dataset_name
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.mode = mode
        self.model_type = model_type
        self.device = device
        self.base_cache_dir = base_cache_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.segmentor = ConversationSegmentor() if mode == "segment" else None
        self.compressor = None
        self.compress_rate = compress_rate
        self.id2index = {}
        self.id2text = {}
        self.embeddings = {}

        # Load sessions
        self.sessions = self._load_sessions()

        # Set cache path
        self.cache_dir = os.path.join(base_cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        if "ours" in self.mode:
            self.embedding_path = os.path.join(self.cache_dir, "embeddings.pkl")
        else:
            self.embedding_path = os.path.join(self.cache_dir, "embeddings.npy")
        self.index_path = os.path.join(self.cache_dir, "id2index.pkl")
        self.text_path = os.path.join(self.cache_dir, "id2text.pkl")
        self.segments_path = os.path.join(self.cache_dir, "segments.pkl")

    @classmethod
    async def create(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        await self.async_init()
        return self

    async def async_init(self):
        # Create embedder
        self.embedder = EmbeddingModel(
            model_type=self.model_type,
            model_name=self.embedding_model_name,
            device=self.device,
        )

        # Load or create embeddings
        if os.path.exists(self.embedding_path) and os.path.exists(self.index_path):
            print(f"[EmbeddingDB] Loading cached embeddings")
            self._load_data()
        elif "ours_final" in self.mode:
            print("Loading our final model")
            self._load_data()
        else:
            print(
                f"[EmbeddingDB] No cache → Starting '{self.embedding_model_name}/{self.mode}' embedding creation"
            )
            await self._build_and_save()


    def _preprocess(self, text, lemmatizer, stop_words):
        """Tokenize, lowercase, lemmatize text and remove stopwords"""
        # Tokenize and lowercase
        tokens = word_tokenize(str(text).lower())

        # Remove stopwords & apply lemmatization
        return [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in stop_words
        ]

    def _load_sessions(self):
        session_path = os.path.join("premem_module/dataset/sessions.pkl")
        if not os.path.exists(session_path):
            raise FileNotFoundError(f"Session file not found: {session_path}")
        sessions = read_pickle(session_path)
        sessions = [Session(**session) for session in sessions]
        return sessions

    def _load_data(self):
        if "ours_final" in self.mode:
            raw = read_pickle(self.index_path.replace("ours_final", "ours"))
            reason = read_pickle(self.index_path.replace("ours_final", "ours_reasoning"))
            for key, raw_value in raw.items():
                reason_value = reason[key]
                self.id2index[key] = {"raw_" + str(k): v for k, v in raw_value.items()}
                self.id2index[key].update(
                    {"reason_" + str(k): v + len(raw_value) for k, v in reason_value.items()}
                )

            raw = read_pickle(self.text_path.replace("ours_final", "ours"))
            reason = read_pickle(self.text_path.replace("ours_final", "ours_reasoning"))
            for key, raw_value in raw.items():
                reason_value = reason[key]
                self.id2text[key] = {"raw_" + str(k): v for k, v in raw_value.items()}
                self.id2text[key].update(
                    {"reason_" + str(k): v for k, v in reason_value.items()}
                )

            raw = read_pickle(self.embedding_path.replace("ours_final", "ours"))
            reason = read_pickle(self.embedding_path.replace("ours_final", "ours_reasoning"))
            for key, raw_value in raw.items():
                reason_value = reason[key]
                self.embeddings[key] = torch.tensor(
                    np.concatenate([raw_value, reason_value], axis=0)
                )

        else:
            if self.embedding_path[-4:] == ".pkl":
                self.embeddings = {
                    k: torch.tensor(v) for k, v in read_pickle(self.embedding_path).items()
                }
            else:
                self.embeddings = torch.tensor(np.load(self.embedding_path))
            self.id2index = read_pickle(self.index_path)
            self.id2text = read_pickle(self.text_path)

    async def _build(self):
        texts, ids, session_date = await self._extract_texts_and_ids()
        id2index = {id_: i for i, id_ in enumerate(ids)}
        id2text = {
            id_: {"text": text, "session_date": date}
            for id_, text, date in zip(ids, texts, session_date)
        }

        return texts, ids, id2index, id2text

    async def _build_and_save(self):
        texts, _, id2index, id2text = await self._build()
        embeddings = self.embedder.create(
            texts, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar
        )
        embeddings = np.array(embeddings)
        save_pickle(id2index, self.index_path)
        save_pickle(id2text, self.text_path)
        np.save(self.embedding_path, embeddings)

        self.embeddings = torch.tensor(embeddings)
        self.id2index = id2index
        self.id2text = id2text

    async def _build_text_data(self):
        """Build text data only for BM25"""
        _, ids, id2index, id2text = await self._build()

        save_pickle(id2index, self.index_path)
        save_pickle(id2text, self.text_path)

        self.id2index = id2index
        self.id2text = id2text
        self.id_list = ids  # Save ID list for BM25 search

    async def _extract_texts_and_ids(self):
        texts = []
        ids = []
        session_date = []

        if self.mode == "session":
            for s in self.sessions:
                # text = "\n".join(f"[{m['role']}]: {m['content']}" for m in s.conversation)
                texts.append(s.conversation)
                ids.append(s.session_id)
                session_date.append(s.session_date)

        if self.mode == "turn" or self.mode == "ours":
            for s in self.sessions:
                conversation = s.conversation
                for i in range(0, len(conversation), 2):
                    texts.append(conversation[i : i + 2])
                    ids.append(
                        tuple(
                            [s.session_id]
                            + [conv["message_id"] for conv in conversation[i : i + 2]]
                        )
                    )
                    session_date.append(s.session_date)

        elif self.mode in ["segment", "segment_compressed"]:
            if os.path.exists(self.segments_path):
                segments = read_pickle(self.segments_path)
            else:
                segments = await self.segmentor.segment(
                    self.sessions, save_file_name=self.segments_path
                )
            for segment in segments:
                original_session = segment.original_session
                segment_result = segment.result
                for seg in segment_result:
                    text = seg.conversation
                    texts.append(text)
                    ids.append(seg.segment_id)
                    session_date.append(original_session.session_date)

        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if self.mode == "segment_compressed":
            compressed_results = self.compressor.compress(
                texts, compress_rate=self.compress_rate
            )
            formatted_texts = []
            for compressed in compressed_results:
                if isinstance(compressed, list):
                    formatted_texts.append("\n".join(compressed))
                else:
                    formatted_texts.append(str(compressed))
        else:
            formatted_texts = [
                "\n".join(f"[{m['role']}]: {m['content']}" for m in s) for s in texts
            ]

        return formatted_texts, ids, session_date


    def get(self, ids, question_id=None):
        """Get embeddings or texts for specified IDs"""
        if isinstance(ids, str):
            ids = [ids]

        if question_id is not None:
            indices = [self.id2index[i] for i in ids if i in self.id2index]
            return self.embeddings[indices]
        else:
            indices = [
                self.id2index[question_id][i]
                for i in ids
                if i in self.id2index[question_id]
            ]
            return self.embeddings[question_id][indices]

    def get_text(self, ids, question_id=None):
        """Get texts for specified IDs"""
        if isinstance(ids, str):
            ids = [ids]
        if question_id is not None:
            return [
                self.id2text[question_id][i] for i in ids if i in self.id2text[question_id]
            ]
        else:
            return [self.id2text[i] for i in ids if i in self.id2text]

    def retrieve(self, query_text, k=5, question_id=None, **kwargs):
        """
        Perform embedding or BM25-based search depending on the model

        Args:
            query_text (str): Text query to search for
            k (int): Number of results to return

        Returns:
            list: List of dictionaries containing search results
                 [{'id': id, 'text': text, 'score': score}, ...]
        """
        return self._retrieve_embedding(query_text, k, question_id=question_id, **kwargs)

    def _retrieve_embedding(self, query_text, k=5, question_id=None, **kwargs):
        """Embedding-based search"""
        # Convert input text to embedding
        if "stella_en_" in self.embedding_model_name:
            kwargs["prompt"] = "s2p_query"
        query_embedding = torch.tensor(
            self.embedder.create([query_text], show_progress_bar=False, **kwargs)[0]
        )

        query_embedding_normalized = query_embedding / query_embedding.norm()
        if question_id is not None:
            all_embeddings_normalized = self.embeddings[question_id] / self.embeddings[
                question_id
            ].norm(dim=1, keepdim=True)
        else:
            all_embeddings_normalized = self.embeddings / self.embeddings.norm(
                dim=1, keepdim=True
            )

        scores = all_embeddings_normalized @ query_embedding_normalized.T
        top_indices = torch.topk(scores, min(k, len(scores))).indices

        # Compile results
        results = []
        if question_id is not None:
            index_to_id = {idx: id_ for id_, idx in self.id2index[question_id].items()}
        else:
            index_to_id = {idx: id_ for id_, idx in self.id2index.items()}

        for idx in top_indices:
            idx = idx.item()
            result_id = index_to_id[idx]
            if question_id is not None:
                result_text = self.id2text[question_id][result_id]
            else:
                result_text = self.id2text[result_id]

            result_dict = {"id": result_id, **result_text, "score": float(scores[idx])}

            results.append(result_dict)

        return results


    def __len__(self):
        return len(self.id2text)
