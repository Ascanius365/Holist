import os
import sys
import argparse
import asyncio
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import hashlib
import gc
import threading
import concurrent.futures
from functools import lru_cache
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# 경로 및 모듈 임포트
sys.path.append(".")
from premem_module.src.db import EmbeddingDB
from premem_module.src.model.llm import LLMAgent
from premem_module.src.utils.io import read_pickle, read_jsonl, save_jsonl
from premem_module.src.model.embedding import EMBEDDING_MODEL_POOL

warnings.filterwarnings("ignore", category=ConvergenceWarning)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run episodic memory reasoning experiment.")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-nano")
    parser.add_argument("--api_base", type=str, default=None)
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="NovaSearch/stella_en_400M_v5",
        choices=EMBEDDING_MODEL_POOL,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="minecraft",  # Setze es direkt als Standard
        choices=["locomo", "longmemeval_s", "minecraft"],
    )
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
    parser.add_argument(
        "--cpu_workers",
        type=int,
        default=None,
        help="Number of CPU workers. Default: cpu_count - 8",
    )
    parser.add_argument(
        "--batch_items",
        type=int,
        default=None,
        help="Number of items per batch. Default: min(30, len(unique_session_pools) // 10 + 1)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for clustering",
    )
    return parser.parse_args()


def get_md5_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def get_optimized_text(df):
    # Nutzt 'session_date', da es Datum und Uhrzeit enthält
    time_col = "session_date" if "session_date" in df.columns else ("time" if "time" in df.columns else "date")
    return (
        "["
        + df["key"]
        + ", "
        + df[time_col]
        + "]: "
        + df["role"]
        + " "
        + df["value"].apply(
            lambda x: x[0].lower() + x[1:] if isinstance(x, str) and len(x) > 1 else x
        )
    )


def find_optimal_k_and_cluster(embeddings, k_range=range(2, 10)):
    best_k = 1
    best_score = -1
    best_labels = np.zeros(len(embeddings), dtype=np.int32)
    try:
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            score = silhouette_score(embeddings, labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
    except ValueError:
        return 1, np.zeros(len(embeddings), dtype=np.int32)
    return best_k, best_labels


async def async_batch_process_tasks_and_save(tasks, params, output_path, batch_size=128):
    async def safe_execute_task(task):
        try:
            return await task
        except Exception as e:
            print(e)
            return None

    for i in tqdm(range(0, len(tasks), batch_size)):
        batch_tasks = tasks[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[safe_execute_task(task) for task in batch_tasks]
        )
        save_results_to_jsonl(batch_results, params[i : i + batch_size], output_path)


def save_results_to_jsonl(batch_results, batch_params, output_path):
    if os.path.exists(output_path):
        existing_data = read_jsonl(output_path)
    else:
        existing_data = []
    new_data = []
    for response, param in zip(batch_results, batch_params):
        if response is None:
            continue
        new_data.append({**param, **response.to_dict()})
    combined_data = existing_data + new_data
    save_jsonl(combined_data, output_path)


def main():
    args = parse_arguments()

    # 출력 경로
    output_path = os.path.join(
        args.output_dir,
        args.dataset_name,
        args.mode,
        args.model_name,
        args.embedding_model_name,
        f"token_budget={args.token_budget}_temperature={args.temperature}",
        "results.jsonl",
    )

    # EmbeddingDB, QA, Agent 초기화
    loop = asyncio.get_event_loop()
    db = loop.run_until_complete(
        EmbeddingDB.create(
            dataset_name=args.dataset_name,
            embedding_model_name=args.embedding_model_name,
            mode=args.mode,
            device=args.device,
            base_cache_dir=args.base_cache_dir,
            data_dir=args.data_dir,
            batch_size=args.embedding_batch_size,
            show_progress_bar=args.show_progress_bar,
            compress_rate=args.compress_rate,
        )
    )

    agent = LLMAgent(model_name=args.model_name)
    sessions = db.sessions

    # 임베딩, 세션풀 로드
    embeddings = read_pickle(f".cache/embeddings.pkl")
    session_pools = read_pickle(f".cache/session_pool.pkl")

    # unique session pool 생성
    unique_session_pools = {}
    unique_session_pools_question_mapping = {}
    question_unique_session_pools_mapping = {}
    unique_session_pools_embeddings_mapping = {}
    str_df_to_hash = {}

    for question_id, temp_df in session_pools.items():
        str_df = str(temp_df)
        if str_df in unique_session_pools:
            continue
        unique_session_pools[str_df] = temp_df
        unique_session_pools_question_mapping[str_df] = question_id
        question_unique_session_pools_mapping[question_id] = str_df
        unique_session_pools_embeddings_mapping[str_df] = embeddings[question_id]
        str_df_to_hash[str_df] = get_md5_hash(str_df)

    # 이미 처리된 항목 로드
    """cache_jsonl = f".cache/{args.dataset_name}/{args.model_name}/ours_reasoning.jsonl"
    if os.path.exists(cache_jsonl):
        already_done = [
            (key["hash_key"], tuple(key["idx_key"]), tuple(key["remain_key"]))
            for key in read_jsonl(cache_jsonl)
        ]
        print("Already done!")
    else:"""

    already_done = []

    session_map = {session.session_id: session for session in sessions}
    keys = []
    tasks = []
    session_pool_info = {}

    CPU_COUNT = os.cpu_count()

    # FIX: Nutze max(1, ...), damit niemals 0 oder negativ herauskommt
    MAX_WORKERS = args.cpu_workers if args.cpu_workers is not None else max(1, CPU_COUNT - 1)

    BATCH_SIZE = (
        args.batch_items
        if args.batch_items is not None
        else max(1, len(unique_session_pools) // 10)  # Sicherstellen, dass auch hier min. 1 steht
    )
    threshold = args.similarity_threshold

    @lru_cache(maxsize=2048)
    def calculate_cluster(embedding_data, k_range):
        return find_optimal_k_and_cluster(embedding_data, k_range=k_range)

    def process_session_pool(item):
        key, session_pool_df = item
        hash_key = str_df_to_hash[key]
        embedding = unique_session_pools_embeddings_mapping[key]
        session_ids = session_pool_df["session_id"].unique()
        dates = [session_map[session_id].session_date for session_id in session_ids]
        sorted_index = pd.to_datetime(
            pd.DataFrame(dates)[0],
            format="%Y-%m-%d %A %H:%M:%S",
        ).argsort()
        session_ids = session_ids[sorted_index]
        info = {}
        for idx, session_id in enumerate(session_ids):
            session = session_map[session_id]
            session_df = session_pool_df[session_pool_df["session_id"] == session_id]
            session_embedding = embedding[session_df.index]

            # Zeitinformation für das Clustering hinzufügen
            if "session_date" in session_df.columns:
                # Spezifisches Format parsen: "2026-01-07 Wednesday 16:58:43"
                times = pd.to_datetime(session_df["session_date"], format="%Y-%m-%d %A %H:%M:%S")
            elif "time" in session_df.columns:
                times = pd.to_datetime(session_df["time"])
            else:
                times = pd.to_datetime(session_df["date"])

            k_range = range(2, min(10, len(session_embedding)))
            embedding_tuple = tuple(map(tuple, session_embedding))
            k_range_tuple = tuple(k_range)
            optimal_k, optimal_labels = calculate_cluster(embedding_tuple, k_range_tuple)
            clustered_session_embedding = np.array(
                [
                    session_embedding[optimal_labels == i].mean(axis=0)
                    for i in range(optimal_k)
                ]
            )
            optimized_text = get_optimized_text(session_df)
            info[idx] = {
                "session_id": session_id,
                "session": session,
                "session_df": session_df,
                "session_embedding": session_embedding,
                "clustered_session_embedding": clustered_session_embedding,
                "optimal_k": optimal_k,
                "optimal_labels": optimal_labels,
                "optimized_text": optimized_text,
            }
        session_pool_info[hash_key] = {"session_pool_df": session_pool_df, "info": info}
        remain_keys = {}
        if 0 in info:
            remain_keys = {
                (0, j.item()): {
                    "clustered_session_embedding": info[0]["clustered_session_embedding"][
                        j.item()
                    ],
                    "optimized_text": info[0]["optimized_text"][
                        info[0]["optimal_labels"] == j.item()
                    ],
                }
                for j in np.unique(info[0]["optimal_labels"])
            }
        local_keys = []
        local_tasks = []
        for i in range(1, max(info.keys()) + 1 if info else 0):
            if i not in info:
                continue
            if not remain_keys:
                remain_keys = {
                    (i, j.item()): {
                        "clustered_session_embedding": info[i]["clustered_session_embedding"][
                            j.item()
                        ],
                        "optimized_text": info[i]["optimized_text"][
                            info[i]["optimal_labels"] == j.item()
                        ],
                    }
                    for j in np.unique(info[i]["optimal_labels"])
                }
                continue
            remain_embeddings = np.array(
                [value["clustered_session_embedding"] for value in remain_keys.values()]
            )
            clustered_session_embeddings = []
            optimized_texts = []
            for j in np.unique(info[i]["optimal_labels"]):
                clustered_session_embeddings.append(
                    info[i]["clustered_session_embedding"][j.item()]
                )
                optimized_texts.append(
                    info[i]["optimized_text"][info[i]["optimal_labels"] == j.item()]
                )
            clustered_session_embeddings = np.array(clustered_session_embeddings)
            sim = remain_embeddings @ clustered_session_embeddings.T
            session_info = info[i]
            target_index = np.where(sim > threshold)
            for x, y in zip(*target_index):
                remain_key = list(remain_keys.keys())[x]
                memory_fragments = (
                    "\n".join(list(remain_keys.values())[x]["optimized_text"]) + "\n"
                )
                memory_fragments += "\n".join(
                    session_info["optimized_text"][session_info["optimal_labels"] == y]
                )
                if (hash_key, (i, y.item()), remain_key) in already_done:
                    continue
                local_keys.append(
                    {
                        "hash_key": hash_key,
                        "idx_key": (i, y.item()),
                        "remain_key": remain_key,
                        "memory_fragments": memory_fragments,
                    }
                )
                local_tasks.append(
                    agent.get_completion(
                        "prompts/reason_info.yaml",
                        memory_fragments=memory_fragments,
                        api_base=args.api_base,
                    )
                )
            for delete_key in set([list(remain_keys.keys())[i] for i in target_index[0]]):
                del remain_keys[delete_key]
            remain_keys.update(
                {
                    (i, j.item()): {
                        "clustered_session_embedding": info[i]["clustered_session_embedding"][
                            j.item()
                        ],
                        "optimized_text": info[i]["optimized_text"][
                            info[i]["optimal_labels"] == j.item()
                        ],
                    }
                    for j in np.unique(info[i]["optimal_labels"])
                    if j not in target_index[1]
                }
            )
        return hash_key, local_keys, local_tasks

    def process_in_batches(items, batch_size=BATCH_SIZE):
        results_keys = []
        results_tasks = []
        total_batches = (len(items) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(items))
            current_batch = items[batch_start:batch_end]
            print(
                f"\n배치 {batch_idx+1}/{total_batches} 처리 시작 (항목 {batch_start+1}-{batch_end}/{len(items)})"
            )
            print(f"현재 스레드 수: {threading.active_count()}")
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(MAX_WORKERS, len(current_batch))
            ) as executor:
                futures = [
                    executor.submit(process_session_pool, item) for item in current_batch
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"배치 {batch_idx+1}/{total_batches}",
                ):
                    try:
                        hash_key, local_keys, local_tasks = future.result()
                        results_keys.extend(local_keys)
                        results_tasks.extend(local_tasks)
                    except Exception as e:
                        print(f"작업 처리 오류: {str(e)}")
            del futures, current_batch
            gc.collect()
            print(
                f"배치 {batch_idx+1}/{total_batches} 완료 - 현재까지 {len(results_keys)} 키, {len(results_tasks)} 태스크 처리됨"
            )
        return results_keys, results_tasks

    print(f"\n작업 시작: {len(unique_session_pools)}개 세션 풀 처리")
    pool_items = list(unique_session_pools.items())
    try:
        print("배치 처리 시작...")
        batch_keys, batch_tasks = process_in_batches(pool_items)
        keys = batch_keys
        tasks = batch_tasks
        print(f"\n모든 작업 완료: 총 {len(keys)} 키, {len(tasks)} 태스크")
        print(f"최종 스레드 수: {threading.active_count()}")
    except Exception as e:
        print(f"전체 처리 중 오류 발생: {str(e)}")
        print(f"부분 결과: {len(keys)} 키, {len(tasks)} 태스크")

    reasoning_jsonl = ".cache/ours_reasoning.jsonl"
    if os.path.exists(reasoning_jsonl):
        try:
            os.remove(reasoning_jsonl)
            print(f"🗑️  Alte JSONL gelöscht: {reasoning_jsonl}")
        except Exception as e:
            print(f"⚠️ Konnte nicht löschen: {e}")
    # 실제 태스크 실행 및 저장
    loop.run_until_complete(
        async_batch_process_tasks_and_save(
            tasks,
            keys,
            f".cache/ours_reasoning.jsonl",
            batch_size=args.batch_size,
        )
    )


if __name__ == "__main__":
    main()
