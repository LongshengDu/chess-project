"""Coaching time, response and token budgets with per-response accounting."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .settings import CONFIG


class CoachingLimitError(ValueError):
    """A bounded coaching run stopped; local evidence remains available."""


@dataclass
class RunBudget:
    max_model_responses: int = CONFIG['COACH']['MAX_MODEL_RESPONSES']
    max_tokens: int = CONFIG['COACH']['TOTAL_TOKEN_BUDGET']
    timeout: float = CONFIG['COACH']['RUN_TIMEOUT_SECONDS']
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    unknown_usage_responses: int = 0
    estimated_tokens: int = 0
    started: float = field(default_factory=time.monotonic)
    usage_path: Path | None = None
    usage_context: Callable | None = None
    response_usage: list = field(default_factory=list)
    context_sizes: dict = field(default_factory=dict)

    def check(self):
        if time.monotonic() - self.started >= self.timeout:
            raise CoachingLimitError('Coaching reached its time limit. Saved analysis and any draft remain available.')
        if self.input_tokens + self.output_tokens + self.estimated_tokens >= self.max_tokens:
            raise CoachingLimitError('Coaching reached its cumulative token budget. Saved evidence remains available.')

    def record_usage(self, inputs=None, outputs=None, cached=0, estimate=0, *, source='provider_response', reported_last=None):
        known = type(inputs) is int and type(outputs) is int and inputs >= 0 and outputs >= 0
        if known:
            self.input_tokens += inputs
            self.output_tokens += outputs
            self.cached_input_tokens += cached if type(cached) is int and cached >= 0 else 0
        else:
            self.unknown_usage_responses += 1
            self.estimated_tokens += estimate
        context = self.usage_context() if self.usage_context else {}
        previous = self.response_usage[-1] if self.response_usage else {}
        cached = cached if type(cached) is int and cached >= 0 else 0
        event = {'response': len(self.response_usage)+1, 'source': source,
            'input_tokens': inputs if known else None, 'output_tokens': outputs if known else None,
            'cached_input_tokens': cached if known else None,
            'uncached_input_tokens': max(0, inputs-cached) if known else None,
            'total_tokens': inputs+outputs if known else None, 'estimated_tokens': 0 if known else estimate,
            'cumulative_input_tokens': self.input_tokens, 'cumulative_output_tokens': self.output_tokens,
            'cumulative_cached_input_tokens': self.cached_input_tokens,
            'cumulative_total_tokens': self.input_tokens+self.output_tokens,
            'elapsed_seconds': round(time.monotonic()-self.started, 3),
            **self.context_sizes, **context}
        for key in ('tool_calls_completed', 'tool_response_chars'):
            if key in context:
                event['new_'+key] = context[key]-previous.get(key, 0)
        if reported_last is not None:
            event['provider_last'] = reported_last
        self.response_usage.append(event)
        if self.usage_path:
            with self.usage_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + '\n')

    def record_cumulative(self, usage, last=None):
        """Codex usage notifications are cumulative; duplicates are not responses."""
        inputs, outputs = usage.input_tokens-self.input_tokens, usage.output_tokens-self.output_tokens
        if inputs < 0 or outputs < 0 or (inputs == 0 and outputs == 0):
            return False
        cached = max(0, usage.cached_input_tokens-self.cached_input_tokens)
        reported = None if last is None else {key: getattr(last, key, None) for key in
            ('input_tokens', 'output_tokens', 'cached_input_tokens', 'reasoning_output_tokens', 'total_tokens')}
        self.requests += 1
        self.record_usage(inputs, outputs, cached, source='codex_cumulative_delta', reported_last=reported)
        return True

    def summary(self):
        return {key: getattr(self, key) for key in ('requests', 'input_tokens', 'output_tokens',
                'cached_input_tokens', 'unknown_usage_responses', 'estimated_tokens', 'max_tokens')} | {
            'elapsed_seconds': round(time.monotonic() - self.started, 2)}
