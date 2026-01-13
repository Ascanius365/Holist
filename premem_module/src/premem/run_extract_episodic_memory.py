import os
import sys
import argparse
import asyncio
import logging
from tqdm import tqdm

sys.path.append(".")
from premem_module.src.db import EmbeddingDB
from premem_module.src.model.llm import LLMAgent
from premem_module.src.utils.io import read_jsonl
from premem_module.src.model.embedding import EMBEDDING_MODEL_POOL

logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run episodic memory extraction experiment.")
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
        default="minecraft",
        choices=["locomo", "longmemeval_s", "minecraft"],
    )
    parser.add_argument("--mode", type=str, default="turn")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_dir", type=str, default="dataset/processed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_cache_dir", type=str, default=".cache")
    parser.add_argument("--token_budget", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embedding_batch_size", type=int, default=256)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--show_progress_bar", type=bool, default=True)
    parser.add_argument("--compress_rate", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--timeout", type=int, default=10)
    return parser.parse_args()


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
    from premem_module.src.utils.io import read_jsonl, save_jsonl

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


async def main():
    args = parse_arguments()
    from dotenv import load_dotenv

    load_dotenv()

    output_path = os.path.join(
        args.output_dir,
        args.dataset_name,
        args.mode,
        args.model_name,
        args.embedding_model_name,
        f"token_budget={args.token_budget}_temperature={args.temperature}",
        "results.jsonl",
    )

    db = await EmbeddingDB.create(
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

    agent = LLMAgent(model_name="openrouter/qwen/qwen-2.5-coder-32b-instruct")

    # Test
    response = await agent.get_completion(
        messages=[
            {"role": "user", "content": "Please tell me that the 'Experiment is starting!'"}
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        api_base=args.api_base,
        timeout=args.timeout,
    )
    print("Test response:", response.response)

    sessions = db.sessions

    # Check already done
    cache_jsonl = f".cache/ours.jsonl"
    if os.path.exists(cache_jsonl):
        already_done = [session["session_id"] for session in read_jsonl(cache_jsonl)]
    else:
        already_done = []

    from collections import defaultdict

    tasks = []
    params = []

    for session in tqdm(sessions[:20000]):
        if session.session_id in already_done:
            continue
        conversation = defaultdict(list)
        for message in session.conversation:
            conversation["all"].append(
                f"[{message['message_id']}] ({' '.join(session.session_date.split()[:2])}) {message['content']}"
            )
        for role in conversation:
            temp_conv = "\n".join(conversation[role])
            tasks.append(
                agent.get_completion(
                    "prompts/extract_personal_info_few_shot.yaml",
                    conversation=temp_conv,
                    timeout=50,
                    api_base=args.api_base,
                    api_key=os.environ.get("OPENAI_API_KEY"),
                )
            )
            params.append(
                {
                    "session_id": session.session_id,
                    "session_date": session.session_date,
                    "role": role,
                    "conversation": temp_conv,
                }
            )

    print(args.dataset_name, args.model_name, len(tasks))
    await async_batch_process_tasks_and_save(
        tasks,
        params,
        cache_jsonl,
        batch_size=1,
    )


if __name__ == "__main__":
    asyncio.run(main())
