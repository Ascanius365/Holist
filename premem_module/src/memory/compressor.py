import logging
from dataclasses import dataclass
from typing import List, Union

from tqdm import tqdm
from llmlingua import PromptCompressor

from premem_module.src.memory.segmentor import SegmentContent
from premem_module.src.data.schema import Message


class MemoryCompressor:
    """
    Compressor for memory units.

    This class compresses conversation segments to reduce noise and improve retrieval.

    Args:
    """

    def __init__(
        self, compress_model_name: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    ):
        self.compress_model_name = compress_model_name
        self.compressor = PromptCompressor(
            self.compress_model_name,
            use_llmlingua2=True,
            llmlingua2_config={
                "max_batch_size": 512,
            },
        )

    def _convert_to_str_list(self, target):
        """
        Convert List[Message] for compression.
        """
        # If target is a list of Message dicts
        if (
            isinstance(target, list)
            and target
            and isinstance(target[0], dict)
            and "role" in target[0]
            and "content" in target[0]
        ):
            return [f"[{msg['role']}]: {msg['content']}" for msg in target]
        else:
            logging.error(f"Unsupported target type for compression: {target}")
            raise ValueError("Unsupported target type for compression.")

    def _extract_role_tokens(self, targets) -> list:
        """
        Extract unique role tokens from the input targets for use in force_tokens.
        """
        roles = set()
        for target in targets:
            # If target is SegmentContent
            if isinstance(target, SegmentContent):
                for msg in target.conversation:
                    if isinstance(msg, dict) and "role" in msg:
                        roles.add(f"[{msg['role']}]")
            # If target is a list of Message dicts
            elif (
                isinstance(target, list)
                and target
                and isinstance(target[0], dict)
                and "role" in target[0]
            ):
                for msg in target:
                    roles.add(f"[{msg['role']}]")
        return list(roles)

    def compress(
        self,
        targets: List[Message],
        compress_rate: float = 0.9,
    ) -> List[Union[str, List[str]]]:
        """
        Compress each target (List[Message]) into a compressed prompt list.
        The force_tokens are dynamically set based on the roles found in the input.
        """

        if not isinstance(targets, list):
            raise ValueError("targets must be in list format.")
        if not targets:
            return []
        for target in targets:
            if not (
                isinstance(target, list)
                and all(
                    isinstance(msg, dict) and "role" in msg and "content" in msg
                    for msg in target
                )
            ):
                raise ValueError("Each item in targets must be in the form of List[Message]")

        if compress_rate >= 1.0 or not self.compressor:
            # No compression needed, or compressor not available
            return targets

        # Dynamically extract role tokens
        force_tokens = self._extract_role_tokens(targets)
        # Always add newline and period tokens
        force_tokens += ["\n", "."]

        logging.debug(f"Force tokens: {force_tokens}")

        compressed_results = []
        for target in tqdm(targets, desc="Compressing targets"):
            if not target:
                continue
            input_str_list = self._convert_to_str_list(target)
            compressed = self.compressor.compress_prompt(
                input_str_list,
                rate=compress_rate,
                use_context_level_filter=False,
                force_tokens=force_tokens,
            )["compressed_prompt_list"]
            compressed_results.append(compressed)

        return compressed_results


if __name__ == "__main__":
    from premem_module.src.utils.io import read_pickle
    from premem_module.src.memory.segmentor import SegmentResult

    segment_results = read_pickle("dataset/segment/mt_bench_segment_gpt41-nano.pkl")

    segmentation_targets = []
    for segment_result in segment_results:
        for segment_content in segment_result.result:
            segmentation_targets.append(segment_content.conversation)

    compressor = MemoryCompressor()

    print("=== SegmentContent test ===")
    compressed_segments = compressor.compress(segmentation_targets, compress_rate=0.2)

    for compressed_segment in compressed_segments:
        print(compressed_segment)
        print("-" * 100)
