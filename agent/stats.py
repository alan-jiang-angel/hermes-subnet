from langchain_core.callbacks import BaseCallbackHandler
from enum import Enum
from langchain_core.messages import BaseMessage
from loguru import logger
import common.utils as utils
from datetime import datetime

class ToolCountHandler(BaseCallbackHandler):
    counter: dict[str, int] = {}
    def __init__(self):
        self.counter = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = (serialized.get("name")
                or serialized.get("id")
                or "unknown_tool")
        if name in ["graphql_schema_info", "graphql_query_validator", "graphql_execute", "graphql_type_detail"]:
            return
        self.counter[name] = self.counter.get(name, 0) + 1

    def stats(self) -> dict[str, int]:
        return self.counter
    

class ProjectCounter:

    # { cid -> [suc, fail] }
    counter: dict[str, list[int]] = {}
    def __init__(self):
        self.counter = {}

    def incr(self, cid: str, success: bool = True) -> dict[str, list[int]]:
        if cid not in self.counter:
            self.counter[cid] = [0, 0]
    
        self.counter[cid][0] += 1 if success else 0
        self.counter[cid][1] += 0 if success else 1

        return self.counter

    def stats(self) -> dict[str, list[int]]:
        return self.counter

class ToolCounter:
    counter: dict[str, int] = {}
    def __init__(self):
        self.counter = {}

    def incr(self, tool_name: str, count: int) -> dict[str, int]:
        self.counter[tool_name] = self.counter.get(tool_name, 0) + count
        return self.counter

    def stats(self) -> dict[str, int]:
        return self.counter


class ProjectUsageMetrics:

    def __init__(self):
        self._synthetic_tool_counter = ToolCounter()
        self._organic_tool_counter = ToolCounter()
        self._synthetic_project_counter = ProjectCounter()
        self._organic_project_counter = ProjectCounter()

    @property
    def synthetic_tool_usage(self) -> ToolCounter:
        return self._synthetic_tool_counter

    @property
    def organic_tool_usage(self) -> ToolCounter:
        return self._organic_tool_counter

    @property
    def synthetic_project_usage(self) -> ProjectCounter:
        return self._synthetic_project_counter

    @property
    def organic_project_usage(self) -> ProjectCounter:
        return self._organic_project_counter
    
    def stats(self) -> dict[str, any]:
        return {
            "synthetic_tool_usage": self.synthetic_tool_usage.stats(),
            "organic_tool_usage": self.organic_tool_usage.stats(),
            "synthetic_project_usage": self.synthetic_project_usage.stats(),
            "organic_project_usage": self.organic_project_usage.stats()
        }


class Phase(Enum):
    GENERATE_QUESTION = "generate_question"
    GENERATE_GROUND_TRUTH = "generate_ground_truth"
    GENERATE_MINER_GROUND_TRUTH_SCORE = "ground_truth_score"

    MINER_SYNTHETIC = "miner_synthetic_challenge"
    MINER_ORGANIC = "miner_organic_challenge"

class TokenUsageMetrics:
    datas: list[any] = []

    def __init__(self):
        self.datas = []

    def append(
            self,
            cid_hash: str,
            phase: Phase,
            response: BaseMessage | dict[str, any]
        ) -> dict[str, int]:
        extra_input_tokens = 0
        extra_output_tokens = 0

        if isinstance(response, dict):
            messages = response.get('messages', [])
            extra_input_tokens = response.get('intermediate_graphql_agent_input_token_usage', 0)
            extra_output_tokens = response.get('intermediate_graphql_agent_output_token_usage', 0)
        else:
            messages = [response]

        input_tokens, output_tokens = utils.extract_token_usage(messages)
        logger.info(f"[TokenUsageMetrics] - append called with cid_hash: {cid_hash}, phase: {phase}, response: {response} input_tokens: {input_tokens}, output_tokens: {output_tokens}, extra_input_tokens: {extra_input_tokens}, extra_output_tokens: {extra_output_tokens}")

        data = {
            "cid_hash": cid_hash,
            "phase": phase.value,
            "input_tokens": input_tokens + extra_input_tokens,
            "output_tokens": output_tokens + extra_output_tokens,
            "timestamp":  int(datetime.now().timestamp())
        }
        self.datas.append(data) 
        return data

    def stats(self, since_timestamp: int) -> list[any]:
        return [data for data in self.datas if data["timestamp"] > since_timestamp]
