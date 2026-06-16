from typing import Protocol, runtime_checkable
from cogno_anima.types import PipelineContext
from cogno_anima.llm import LLMBackend

@runtime_checkable
class BaseStage(Protocol):
    """Protocol that every stage in the cognitive pipeline must implement."""
    name: str

    async def process(self, ctx: PipelineContext, llm: LLMBackend) -> PipelineContext:
        """Processes the context and returns the updated context."""
        ...
