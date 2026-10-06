"""Typed local chess tools and JSON schemas for the Codex dynamic-tool protocol."""
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import get_args, get_origin, get_type_hints


def _schema(annotation):
    args, origin = get_args(annotation), get_origin(annotation)
    if type(None) in args:
        return {'anyOf': [_schema(t) for t in args]}
    if annotation is type(None):
        return {'type': 'null'}
    if origin is list:
        return {'type': 'array', 'items': _schema(args[0])}
    types = {int: 'integer', str: 'string', float: 'number', bool: 'boolean', dict: 'object'}
    if annotation not in types:
        raise TypeError(f'Unsupported chess tool annotation: {annotation}')
    return {'type': types[annotation]}


def _validate(value, schema, path):
    if 'anyOf' in schema:
        for option in schema['anyOf']:
            try:
                _validate(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f'{path} does not match an allowed type.')
    kind = schema['type']
    valid = {'null': value is None, 'integer': type(value) is int,
             'number': type(value) in (int, float), 'string': isinstance(value, str),
             'boolean': type(value) is bool, 'object': isinstance(value, dict),
             'array': isinstance(value, list)}[kind]
    if not valid:
        raise ValueError(f'{path} must be {kind}.')
    if kind in ('integer', 'number'):
        if kind == 'number' and not math.isfinite(value):
            raise ValueError(f'{path} must be finite.')
        if 'minimum' in schema and value < schema['minimum']:
            raise ValueError(f'{path} must be at least {schema["minimum"]}.')
        if 'maximum' in schema and value > schema['maximum']:
            raise ValueError(f'{path} must be at most {schema["maximum"]}.')
    if kind == 'array':
        for index, item in enumerate(value):
            _validate(item, schema['items'], f'{path}[{index}]')
    if kind == 'object' and 'properties' in schema:
        for key in schema.get('required', []):
            if key not in value:
                raise ValueError(f'Argument {key} is required.')
        for key, item in value.items():
            if key not in schema['properties']:
                raise ValueError(f'Unknown argument: {key}.')
            _validate(item, schema['properties'][key], key)


@dataclass(frozen=True)
class ChessTool:
    name: str
    description: str
    input_schema: dict
    function: object

    def __call__(self, **arguments):
        validate_tool_arguments(self, arguments)
        return self.function(**arguments)


def validate_tool_arguments(tool, arguments):
    _validate(arguments, tool.input_schema, 'arguments')


def chess_tool(function):
    """Expose a typed function; argument descriptions come from its Args docstring."""
    doc = inspect.getdoc(function) or ''
    description, _, arguments = doc.partition('\nArgs:')
    descriptions = dict(re.findall(r'^\s*(\w+): (.+)$', arguments, re.M))
    hints, properties, required = get_type_hints(function), {}, []
    for name, parameter in inspect.signature(function).parameters.items():
        properties[name] = {**_schema(hints[name]), 'description': descriptions.get(name, name)}
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            properties[name]['default'] = parameter.default
    return ChessTool(function.__name__, description.strip(),
        {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}, function)


def chess_tools(library):
    """Bind the nine typed chess tools to one investigation session."""
    @chess_tool
    def get_leadup(ply: int, lookback_plies: int = 8) -> dict:
        """Trace the actual moves BEFORE a position, saved evals/flags and factual positional changes. To go farther back, call again at from_ply. Use earlier plies with comparison/branch tools to test prevention ideas.

        Args:
            ply: One-based game ply BEFORE its move; N+1 is the final board.
            lookback_plies: Number of preceding half-moves, 1–16; default 8.
        """
        return library.call('get_leadup', dict(ply=ply, lookback_plies=lookback_plies))

    @chess_tool
    def investigate_batch(requests: list[dict]) -> dict:
        """Request 1–6 independent chess investigations in one model response. Each item is {tool, arguments}; results keep input order. Use existing evidence first. No nested batches.

        Args:
            requests: Objects with tool (a supplied chess tool except this one) and arguments (its normal JSON arguments). Example: [{"tool":"stockfish_analyze","arguments":{"ply":13,"movetime_ms":1000}},{"tool":"get_position","arguments":{"ply":21}}].
        """
        return library.call('investigate_batch', dict(requests=requests))

    @chess_tool
    def get_game_analysis() -> dict:
        """Read the compact overview; do not repeat it when already supplied."""
        return library.call('get_game_analysis', {})

    @chess_tool
    def get_position(ply: int, line: list[str] | None = None) -> dict:
        """Inspect a game or branch and render its SVG without engine searches. Use for earlier setup, PV endpoints or exercise boards missing from supplied diagrams.

        Args:
            ply: One-based game ply BEFORE its move; N+1 is the final board.
            line: Optional UCI branch moves starting at that ply; omit for the actual position.
        """
        return library.call('get_position', dict(ply=ply, line=line))

    @chess_tool
    def maia_analyze(ply: int, player_elo: int, opponent_elo: int, line: list[str] | None = None, topk: int = 3) -> dict:
        """Predict likely human moves at a game or branch position, preserving full history.

        Args:
            ply: One-based game ply before the branch begins.
            player_elo: Mover rating on Maia's Lichess Blitz scale, 600–3000.
            opponent_elo: Opponent rating on Maia's Lichess Blitz scale, 600–3000.
            line: UCI moves from that ply; omit for the actual position.
            topk: Number of choices, 1–5.
        """
        return library.call('maia_analyze', dict(ply=ply, player_elo=player_elo, opponent_elo=opponent_elo, line=line, topk=topk))

    @chess_tool
    def maia_compare(ply: int, ratings: list[int], line: list[str] | None = None, topk: int = 3) -> dict:
        """Compare stronger human continuations at a game or branch position. Baseline choices include eval/loss.

        Args:
            ply: One-based game ply before the branch begins.
            ratings: Up to six mover ratings on Maia's Lichess Blitz scale, 600–3000.
            line: UCI moves from that ply; omit for the actual position.
            topk: Number of choices at each rating, 1–5.
        """
        return library.call('maia_compare', dict(ply=ply, ratings=ratings, line=line, topk=topk))

    @chess_tool
    def stockfish_analyze(ply: int, movetime_ms: int, line: list[str] | None = None,
                          multipv: int = 2, root_moves: list[str] | None = None, pv_plies: int = 8) -> dict:
        """Verify tactical claims and strongest defenses. Returns checked SAN lines and White-perspective evals.

        Args:
            ply: One-based game ply before the branch begins.
            movetime_ms: Search milliseconds, within the configured run ceiling.
            line: UCI moves from that ply; omit for the actual position.
            multipv: Number of lines, 1–5; prefer 1 for a focused check.
            root_moves: Optional legal UCI candidates to restrict the search.
            pv_plies: Maximum continuation length, 1–16.
        """
        return library.call('stockfish_analyze', dict(ply=ply, movetime_ms=movetime_ms, line=line,
            multipv=multipv, root_moves=root_moves, pv_plies=pv_plies))

    @chess_tool
    def explore_candidate(ply: int, candidate: str, line: list[str] | None = None,
                          stockfish_ms: int | None = None, maia_elos: list[int] | None = None, max_plies: int = 8) -> dict:
        """Assess a move through immediate human_replies first: Maia probabilities, checked evals and legal mate-in-one replies. Follow likely human branches to judge practical difficulty. stockfish/after_defense separately check strongest resistance.

        Args:
            ply: One-based game ply before the branch begins.
            candidate: Legal UCI move from the position after line.
            line: UCI branch from ply; append a human_replies move to human_replies.position.line to continue there.
            stockfish_ms: Verification milliseconds; null uses the configured budget.
            maia_elos: Up to six Lichess Blitz ratings; null uses converted actual Elo and +200/+400/+600 in that native scale.
            max_plies: Checked variation length, 2–16.
        """
        return library.call('explore_candidate', dict(ply=ply, candidate=candidate, line=line,
            stockfish_ms=stockfish_ms, maia_elos=maia_elos, max_plies=max_plies))

    @chess_tool
    def compare_played_vs_candidate(ply: int, candidate: str) -> dict:
        """Compare both moves' likely human replies and tactical checks. Base praise/criticism on human_replies and executable continuations, not engine-PV ranking. Includes decision and engine-defense SVGs; get_position renders likely human branches.

        Args:
            ply: One-based played game ply.
            candidate: Legal UCI alternative BEFORE that move; choose from baseline Maia candidates.
        """
        return library.call('compare_played_vs_candidate', dict(ply=ply, candidate=candidate))

    stockfish_analyze.input_schema['properties']['movetime_ms'].update(
        minimum=1, maximum=library.engines.limits.max_ms,
        description=f'Search milliseconds, 1–{library.engines.limits.max_ms}; normally {library.engines.limits.verify_ms}.')
    return [get_game_analysis, get_position, maia_analyze, maia_compare, stockfish_analyze,
            explore_candidate, compare_played_vs_candidate, investigate_batch, get_leadup]
