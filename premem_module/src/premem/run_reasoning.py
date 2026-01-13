import os
import sys
import argparse
import asyncio
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import hashlib
import gc
import threading
import concurrent.futures
from sklearn.cluster import KMeans
import warnings

# Pfade und Modul-Importe
sys.path.append(".")
from premem_module.src.db import EmbeddingDB
from premem_module.src.model.llm import LLMAgent
from premem_module.src.utils.io import read_pickle, read_jsonl, save_jsonl
from premem_module.src.model.embedding import EMBEDDING_MODEL_POOL

warnings.filterwarnings("ignore")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run episodic memory reasoning experiment with 10x compression.")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-nano")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument("--embedding_model_name", type=str, default="NovaSearch/stella_en_400M_v5",
                        choices=EMBEDDING_MODEL_POOL)
    parser.add_argument("--dataset_name", type=str, default="minecraft",
                        choices=["locomo", "longmemeval_s", "minecraft"])
    parser.add_argument("--mode", type=str, default="turn")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_cache_dir", type=str, default=".cache")
    parser.add_argument("--token_budget", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--show_progress_bar", type=bool, default=True)
    parser.add_argument("--compress_rate", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--cpu_workers", type=int, default=None)
    parser.add_argument("--compression_factor", type=int, default=10,
                        help="Target compression factor (e.g., 10 means 10 sessions into 1 cluster)")
    return parser.parse_args()


def get_md5_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def get_optimized_text(df):
    time_col = "session_date" if "session_date" in df.columns else ("time" if "time" in df.columns else "date")
    return (
            "[" + df["key"] + ", " + df[time_col] + "]: " + df["role"] + " " +
            df["value"].apply(lambda x: x[0].lower() + x[1:] if isinstance(x, str) and len(x) > 1 else x)
    )


def parse_session_date(date_str):
    if not date_str or not isinstance(date_str, str):
        return pd.NaT
    if " to " in date_str:
        date_str = date_str.split(" to ")[-1].strip()
    formats = ["%Y-%m-%d %A %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y"]
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    try:
        return pd.to_datetime(date_str, format='mixed')
    except:
        return pd.NaT


async def async_batch_process_tasks_and_save(tasks, params, output_path, batch_size=128):
    for i in tqdm(range(0, len(tasks), batch_size)):
        batch_tasks = tasks[i: i + batch_size]
        batch_results = await asyncio.gather(*[asyncio.wait_for(t, timeout=60) for t in batch_tasks],
                                             return_exceptions=True)

        new_data = []
        for response, param in zip(batch_results, params[i: i + batch_size]):
            if isinstance(response, Exception) or response is None:
                continue
            new_data.append({**param, **response.to_dict()})

        if os.path.exists(output_path):
            existing = read_jsonl(output_path)
            save_jsonl(existing + new_data, output_path)
        else:
            save_jsonl(new_data, output_path)


def main():
    args = parse_arguments()
    loop = asyncio.get_event_loop()

    db = loop.run_until_complete(EmbeddingDB.create(
        dataset_name=args.dataset_name, embedding_model_name=args.embedding_model_name,
        mode=args.mode, device=args.device, base_cache_dir=args.base_cache_dir,
        data_dir=args.data_dir, batch_size=args.embedding_batch_size
    ))

    agent = LLMAgent(model_name=args.model_name)
    embeddings_dict = read_pickle(f".cache/embeddings.pkl")
    session_pools = read_pickle(f".cache/session_pool.pkl")
    session_map = {s.session_id: s for s in db.sessions}

    all_keys = []
    all_tasks = []

    print(f"Starte Verarbeitung von {len(session_pools)} Session-Pools...")

    for q_id, pool_df in tqdm(session_pools.items(), desc="Pools"):
        hash_key = get_md5_hash(str(pool_df))
        pool_embeddings = embeddings_dict[q_id]

        # 1. Sessions innerhalb des Pools sammeln und kompakt zusammenfassen
        session_ids = pool_df["session_id"].unique()
        session_data = []

        for s_id in session_ids:
            s_df = pool_df[pool_df["session_id"] == s_id]
            s_emb = pool_embeddings[s_df.index]

            # Kompaktierung: Mittelwert der Embeddings und Text-Join
            mean_emb = s_emb.mean(axis=0)
            text_block = "\n".join(get_optimized_text(s_df))

            session_data.append({
                "id": s_id,
                "embedding": mean_emb,
                "text": text_block
            })

        if not session_data:
            continue

        # 2. Globales Clustering über alle Sessions im Pool (10-fache Verdichtung)
        num_sessions = len(session_data)
        # Ziel: 10-fache Verdichtung, aber immer zwischen 2 und 10 Clustern
        # Wir berechnen zuerst den Rohwert
        raw_target = num_sessions // args.compression_factor

        # Dann wenden wir die Grenzen (2 bis 10) an
        clamped_target = max(2, min(10, raw_target))

        # Schließlich stellen wir sicher, dass wir nicht mehr Cluster als Sessions haben
        target_clusters = min(clamped_target, num_sessions)

        X = np.array([s["embedding"] for s in session_data])

        if num_sessions > 1:
            kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
        else:
            labels = np.array([0])
            target_clusters = 1

        # 3. Pro Cluster ein Reasoning-Task erstellen
        for c_id in range(target_clusters):
            cluster_indices = np.where(labels == c_id)[0]
            cluster_texts = [session_data[i]["text"] for i in cluster_indices]

            # Kombiniere alle Session-Texte dieses Clusters
            memory_fragments = "\n\n--- Nächste Session ---\n\n".join(cluster_texts)

            all_keys.append({
                "hash_key": hash_key,
                "cluster_id": c_id,
                "session_count": len(cluster_indices),
                "memory_fragments": memory_fragments
            })

            all_tasks.append(agent.get_completion(
                "prompts/reason_info.yaml",
                memory_fragments=memory_fragments,
                api_base=args.api_base
            ))

    # 4. Asynchrone Ausführung und Speicherung
    output_file = ".cache/ours_reasoning.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)

    loop.run_until_complete(async_batch_process_tasks_and_save(
        all_tasks, all_keys, output_file, batch_size=args.batch_size
    ))
    print(f"Fertig! Ergebnisse in {output_file} gespeichert.")


if __name__ == "__main__":
    main()