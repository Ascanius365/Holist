import asyncio
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any
from premem_module.src.data.schema import Result

from premem_module.src.model.llm import LLMAgent


def results_to_context(agent, retrieved_results, token_budget, include_date=True):
    context = ""
    length = 0
    used_retrieved_results = []
    for result in retrieved_results:
        temp = ""
        if include_date:
            temp += f"Date: {result['session_date']}\n"
        temp += str(result["text"]) + "\n\n"
        temp_length = agent.count_tokens(temp)
        context += temp
        length += temp_length
        used_retrieved_results.append(result)
        if length + temp_length > token_budget:
            break
    return (
        agent.truncate_text(
            results_to_context_simple(used_retrieved_results, include_date), token_budget
        ),
        used_retrieved_results,
    )


def results_to_context_simple(retrieved_results, include_date=True):
    context = ""
    retrieved_results_with_date = []
    retrieved_results_without_date = []
    for result in retrieved_results:
        if "session_date" in result:
            retrieved_results_with_date.append(result)
        else:
            retrieved_results_without_date.append(result)

    retrieved_results_with_date = sorted(
        retrieved_results_with_date,
        key=lambda x: datetime.strptime(x["session_date"], "%Y-%m-%d %A %H:%M:%S"),
    )
    for result in retrieved_results_without_date:
        context += str(result["text"]) + "\n\n"

    for result in retrieved_results_with_date:
        if include_date:
            context += f"Date: {result['session_date']}\n"
        context += str(result["text"]) + "\n\n"

    return context.strip()


async def async_batch_process_tasks(tasks, batch_size=128):
    responses = [None] * len(tasks)  # 미리 결과 리스트 생성

    async def safe_execute_task(task, index):
        try:
            result = await task
            return index, result
        except Exception as e:
            # 에러 발생 시 None 결과 반환 (별도의 에러 인덱스 불필요)
            return index, None

    for i in tqdm(range(0, len(tasks), batch_size)):
        batch_tasks = []
        indices = range(i, min(i + batch_size, len(tasks)))

        # 각 태스크에 인덱스를 부여하여 안전하게 실행하는 래퍼 생성
        for j, idx in enumerate(indices):
            batch_tasks.append(safe_execute_task(tasks[idx], idx))

        # gather로 병렬 실행
        batch_results = await asyncio.gather(*batch_tasks)

        # 결과 정리
        for idx, result in batch_results:
            responses[idx] = result

    return responses


async def async_batch_completion(
    agent: LLMAgent,
    prompt_path: str,
    list_of_placeholder: List[Dict[str, Any]],
    shared_parameter: Dict[str, Any],
    batch_size: int = 128,
):
    tasks = []
    for placeholder in list_of_placeholder:
        params = {**placeholder, **shared_parameter}
        tasks.append(agent.get_completion(prompt_path, **params))

    return await async_batch_process_tasks(tasks, batch_size)
