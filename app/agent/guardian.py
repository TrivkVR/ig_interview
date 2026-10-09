"""Guardian wrapper around NeMo Guardrails (toxicity, PII, jailbreak)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app import config

GUARDRAILS_DIR = Path(__file__).parent / "guardrails"


@dataclass
class GuardResult:
    blocked: bool
    rail: str | None = None


class Guardian:
    """Checks user input and agent output with NeMo Guardrails' check_async."""

    def __init__(self, rails=None):
        self._rails = rails

    @property
    def rails(self):
        if self._rails is None:
            from nemoguardrails import LLMRails, RailsConfig

            cfg = RailsConfig.from_path(str(GUARDRAILS_DIR))
            for m in cfg.models:
                if m.type == "main":
                    m.model = config.GUARDRAIL_MODEL
            self._rails = LLMRails(cfg)
        return self._rails

    async def check_input(self, user_text: str) -> GuardResult:
        from nemoguardrails.rails.llm.options import RailStatus, RailType

        res = await self.rails.check_async(
            [{"role": "user", "content": user_text}], rail_types=[RailType.INPUT]
        )
        return GuardResult(res.status == RailStatus.BLOCKED, res.rail)

    async def check_output(self, user_text: str, answer: str) -> GuardResult:
        from nemoguardrails.rails.llm.options import RailStatus, RailType

        res = await self.rails.check_async(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ],
            rail_types=[RailType.OUTPUT],
        )
        return GuardResult(res.status == RailStatus.BLOCKED, res.rail)
