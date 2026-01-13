import asyncio
from dataclasses import dataclass, asdict
import json
import traceback
from typing import List, Optional
import logging

from premem_module.src.model.llm import LLMAgent, LLMResponse
from premem_module.src.utils.io import save_pickle
from premem_module.src.utils.parse import extract_result
from premem_module.src.data.schema import Session, Message
from premem_module.src.model.utils import async_batch_process_tasks

logger = logging.getLogger(__name__)


@dataclass
class SegmentContent:
    segment_id: int
    start_exchange_number: int
    end_exchange_number: int
    num_exchanges: int
    summary: str
    conversation: List[Message]

    def to_dict(self):
        return asdict(self)


@dataclass
class SegmentResult:
    original_session: Session
    agent_response: LLMResponse
    result: List[SegmentContent]

    def to_dict(self):
        return asdict(self)


class ConversationSegmentor:
    """
    A class for segmenting conversation histories into meaningful units using LLM-based segmentation.

    This class takes a conversation session (a list of message turns) and splits it into multiple segments,
    where each segment represents a coherent part of the conversation. Segmentation is performed by prompting
    a large language model (LLM) with a custom prompt. If the LLM-based segmentation fails or does not cover
    all turns, a fallback rule-based segmentation is applied (splitting every 3 turns).

    Features:
        - Supports batch asynchronous segmentation of multiple sessions using async_batch_process_tasks.
        - Ensures that all original conversation turns are included in the final segments, with no missing or duplicated turns.
        - Saves segmentation results to a pickle file if a file name is provided.

    Args:
        segment_model_name (str): Name of the LLM model to use for segmentation. (default: "gpt-4.1-nano")
        prompt_path (str): Path to the prompt template for segmentation. (default: "prompts/secom_segment.yaml")
        incremental_prompt_path (str): Path to the prompt template for incremental segmentation. (default: "prompts/secom_incremental_segment.yaml")
    """
    def __init__(
        self, 
        segment_model_name: str = "gpt-4.1-nano", 
        prompt_path: str = "prompts/secom_segment.yaml", 
        incremental_prompt_path: str = "prompts/secom_incremental_segment.yaml"
    ):
        self.segment_model_name = segment_model_name
        self.agent = LLMAgent(segment_model_name)
        
        self.segment_prompt = prompt_path
        self.incremental_segment_prompt = incremental_prompt_path

    def _convert_messages_to_string(self, conversation: List[Message], session_date: Optional[str] = None):
        """
        Convert a list of Message objects into a list of formatted strings for LLM input.

        If session_date is provided, it is prepended as the first line.
        Each message is formatted as "[role]: content".

        Args:
            conversation (List[Message]): The conversation history to convert.
            session_date (Optional[str]): The session date to include (optional).

        Returns:
            List[str]: The formatted conversation as a list of strings.
        """
        if session_date is not None:
            return_texts = [f"[{session_date}]"] + [f"[{turn['role']}]: {turn['content']}" for turn in conversation]
        else:
            return_texts = [f"[{turn['role']}]: {turn['content']}" for turn in conversation]
        return return_texts

    async def _segment_single(self, session: Session, include_date: bool):
        """
        Segment a single conversation session into coherent segments using an LLM.

        This function sends the conversation to the LLM for segmentation. If the LLM output is valid and covers all turns,
        it parses the result into SegmentContent objects. If the LLM output is invalid or incomplete, it falls back to
        rule-based segmentation (splitting every 3 turns).

        Args:
            session (dict): The session dictionary containing 'session_date' and 'conversation'.
            include_date (bool): Whether to include the session date in the LLM input.

        Returns:
            SegmentResult: The segmentation result for the session.
        """
        session_date = session.session_date
        conversation = session.conversation

        output = await self.agent.get_completion(
            self.segment_prompt,
            conversation=self._convert_messages_to_string(
                conversation,
                session_date=session_date if include_date else None,
            ),
        )

        segment_outputs = []

        segmentation_content, is_extraction_successful = extract_result(output.response, "segmentation")
        if is_extraction_successful:
            lines = segmentation_content.strip().split("\n")
            prev_idx = 0
            try:
                for line in lines:
                    line = line.strip().strip(",")
                    line_dict = json.loads(line)
                    line_dict["segment_id"] = f"session-id={session.session_id}_message-id={line_dict['start_exchange_number']}-{line_dict['end_exchange_number']}_seg-id={line_dict['segment_id']}"
                    num_exchanges = int(line_dict.get("num_exchanges", 0))
                    segmented_conversation = conversation[prev_idx : prev_idx + num_exchanges]
                    prev_idx += num_exchanges

                    segment_outputs.append(
                        SegmentContent(
                            **line_dict,
                            conversation=segmented_conversation,
                        )
                    )
            except Exception:
                logger.error(traceback.format_exc())
                is_extraction_successful = False
                segment_outputs = []

        # Check if all conversations exist in the segments
        if is_extraction_successful and len(segment_outputs) > 0:
            merged = []
            for seg in segment_outputs:
                merged.extend(seg.conversation)
            if merged != conversation:
                is_extraction_successful = False
                segment_outputs = []

        if not is_extraction_successful:
            for i in range(0, len(conversation), 3):
                segment_outputs.append(
                    SegmentContent(
                        segment_id=f"session-id={session.session_id}_message-id={i}-{min(i + 2, len(conversation) - 1)}_seg-id={i // 3}",
                        start_exchange_number=i,
                        end_exchange_number=min(i + 2, len(conversation) - 1),
                        num_exchanges=len(conversation[i : i + 3]),
                        summary="",
                        conversation=conversation[i : i + 3],
                    )
                )

        segment_result = SegmentResult(
            original_session=session,
            agent_response=output,
            result=segment_outputs,
        )
        return segment_result

    async def segment(
        self,
        sessions: List[Session],
        save_file_name: Optional[str] = None,
        include_date: bool = False,
        batch_size: int = 128,
    ) -> List[SegmentResult]:
        """
        Segment multiple conversation sessions asynchronously in batches.

        This function creates segmentation tasks for each session and processes them in parallel using async_batch_process_tasks.
        The results are saved to a pickle file.

        Args:
            sessions (List[Session]): List of conversation sessions to segment.
            save_file_name (str): File name to save the segmentation results (pickle format).
            include_date (bool): Whether to include the session date in the LLM input.
            batch_size (int): Number of sessions to process in each async batch.

        Returns:
            List[SegmentResult]: List of segmentation results for all sessions.
        """
        tasks = [
            self._segment_single(session, include_date)
            for session in sessions
        ]
        segment_results = await async_batch_process_tasks(tasks, batch_size=batch_size)
        if save_file_name is not None:
            save_pickle(segment_results, save_file_name)
        return segment_results



async def test():
    segmentor = ConversationSegmentor()
    from premem_module.src.utils.io import read_pickle
    sessions = read_pickle("dataset/processed/mt_bench/sessions.pkl")
    # test_sessions = [sessions[30]]
    test_sessions = sessions

    results = await segmentor.segment(test_sessions)
    # results = await segmentor.segment(test_sessions, include_date=True)

    for result in results:
        print("=" * 50)
        # print(f"original conv: {result.original_session.conversation}")
        for segment in result.result:
            print(segment.conversation)
            print("-" * 50)
        print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test())
