"""Codex SDK + ChatGPT authentication; only compact local chess tools are exposed.

Dynamic tools use the documented experimental app-server protocol. Pinning the
SDK/runtime keeps that protocol reproducible. Authentication uses ChatGPT sign-in.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time

from coach.settings import CONFIG
from .agent_budget import CoachingLimitError
from .report_output import clean_report


def tool_specs(tools):
    return [{'type': 'function', 'name': t.name, 'description': t.description,
             'inputSchema': t.input_schema} for t in tools]


class CodexChessSession:
    def __init__(self, library, budget, *, allow_tools=True):
        self.library, self.budget = library, budget
        self.tools = {t.name: t for t in library.tools} if allow_tools else {}
        self.tool_requests = 0
        self.stop_reason = None

    def handle_request(self, method, params):
        if method != 'item/tool/call':
            # Built-in shell/files/network are disabled; never auto-approve them.
            return {'decision': 'decline'} if 'requestApproval' in method else {}
        try:
            self.budget.check()
            self.tool_requests += 1
            if self.library.max_calls is not None and self.tool_requests > self.library.max_calls:
                self.stop_reason = 'Codex exceeded the chess tool-call limit.'
                raise CoachingLimitError(self.stop_reason)
            if not isinstance(params, dict):
                raise ValueError('Tool request must be a JSON object.')
            name, args = params.get('tool'), params.get('arguments')
            if name not in self.tools or not isinstance(args, dict):
                raise ValueError('Use a supplied chess tool with a JSON object of arguments.')
            result = self.tools[name](**args)
            return {'success': True, 'contentItems': [{'type': 'inputText', 'text': str(result)}]}
        except (ValueError, TypeError) as exc:
            return {'success': False, 'contentItems': [{'type': 'inputText', 'text': json.dumps({'error': str(exc)[:400]})}]}


def run_codex(library, gate, budget, instructions, task, *, model_id=None, allow_tools=True, output_limit=CONFIG['COACH']['OUTPUT_TOKEN_ESTIMATE'], client_factory=None):
    from codex_cli_bin import bundled_codex_path
    from openai_codex import CodexConfig, Thread
    from openai_codex.client import CodexClient
    from openai_codex.generated.v2_all import GetAccountResponse, ThreadStartResponse

    session = CodexChessSession(library, budget, allow_tools=allow_tools)
    specs = tool_specs(list(session.tools.values()))
    budget.context_sizes = {'instructions_chars': len(instructions), 'initial_task_chars': len(task),
                           'tool_schema_chars': len(json.dumps(specs, separators=(',', ':')))}
    binary = str(CONFIG['COACH']['CODEX']['BINARY'] or bundled_codex_path())
    # Override unrelated tool integrations while retaining Codex-managed auth.
    # app-server does not support the exec-only --ignore-user-config flag.
    with tempfile.TemporaryDirectory(prefix='chess-coach-codex-') as working:
        config = CodexConfig(codex_bin=binary, cwd=working, config_overrides=(
            'mcp_servers={}', 'plugins={}', 'features.apps=false',
            'features.shell_tool=false', 'features.multi_agent=false',
            'features.apply_patch_freeform=false', 'web_search="disabled"'), experimental_api=True)
        client = (client_factory or CodexClient)(config=config, approval_handler=session.handle_request)
        timed_out = threading.Event()
        active_request = False
        request_estimate = (len(instructions) + len(task)) // 3 + output_limit
        def expire():
            timed_out.set()
            client.close()
        timer = threading.Timer(max(.01, budget.timeout - (time.monotonic()-budget.started)), expire)
        timer.daemon = True
        try:
            timer.start()
            client.start()
            client.initialize()
            account = client.request('account/read', {'refreshToken': False}, response_model=GetAccountResponse)
            account_data = account.model_dump(mode='json', by_alias=True).get('account') or {}
            if account_data.get('type') not in ('chatgpt', 'chatgptAuthTokens'):
                raise ValueError('Codex needs ChatGPT sign-in. Run codex login; API-key authentication is not used by this provider.')
            started = client.request('thread/start', {
                'cwd': working, 'ephemeral': True, 'approvalPolicy': 'never', 'sandbox': 'read-only',
                'modelProvider': 'openai', **({'model': model_id} if model_id else {}),
                'baseInstructions': instructions,
                'developerInstructions': 'Use supplied initial evidence before calling tools. Batch independent missing investigations with investigate_batch. Return the Markdown report directly when evidence suffices. No shell, file operations, web search or subagents. Do not repeat evidence already supplied.',
                'dynamicTools': specs,
                'config': {'features.shell_tool': False, 'features.apply_patch_freeform': False,
                           'features.multi_agent': False, 'features.apps': False, 'web_search': 'disabled',
                           'model_reasoning_effort': CONFIG['COACH']['CODEX']['REASONING_EFFORT']}}, response_model=ThreadStartResponse)
            thread = Thread(client, started.thread.id)
            library.model_id = started.model
            prompt = task
            seen_commentary = set()
            for attempt in range(gate.max_attempts):
                budget.check()
                if budget.requests >= budget.max_model_responses:
                    raise CoachingLimitError('Codex reached its model-response limit; no report repair was started.')
                requests_before = budget.requests
                active_request = True
                turn = thread.turn(prompt, effort=CONFIG['COACH']['CODEX']['REASONING_EFFORT'])
                answer, status = '', None
                for event in turn.stream():
                    payload = event.payload
                    if event.method == 'thread/tokenUsage/updated':
                        usage = payload.token_usage.total
                        budget.record_cumulative(usage, getattr(payload.token_usage, 'last', None))
                        if budget.requests > budget.max_model_responses:
                            session.stop_reason = 'Codex reached its model-response limit.'
                    elif event.method == 'item/completed':
                        item = payload.item.root if hasattr(payload.item, 'root') else payload.item
                        if getattr(item, 'type', None) == 'agentMessage':
                            phase = getattr(getattr(item, 'phase', None), 'value', getattr(item, 'phase', None))
                            if phase == 'commentary':
                                key = getattr(item, 'id', None) or item.text
                                if key not in seen_commentary:
                                    library.progress.commentary(item.text)
                                    seen_commentary.add(key)
                            elif phase in (None, 'final_answer'):
                                answer = item.text
                        elif getattr(item, 'type', None) in ('commandExecution', 'fileChange', 'webSearch', 'collabToolCall'):
                            session.stop_reason = 'Codex attempted a tool outside the chess interface.'
                    elif event.method == 'turn/completed':
                        status = getattr(payload.turn.status, 'value', payload.turn.status)
                    if session.stop_reason:
                        turn.interrupt()
                        raise CoachingLimitError(session.stop_reason)
                    try:
                        budget.check()
                    except CoachingLimitError:
                        turn.interrupt()
                        raise
                if status != 'completed':
                    raise CoachingLimitError('Codex did not complete the report. Saved evidence remains available; no automatic restart.')
                active_request = False
                if budget.requests == requests_before:
                    budget.requests += 1
                    budget.record_usage(estimate=request_estimate, source='codex_unreported_turn')
                budget.check()
                try:
                    gate.validate(answer)
                    return clean_report(answer)
                except ValueError:
                    gate.check_retry()
                    library.progress.stage('Correcting the report using the checked evidence.')
                    prompt = 'Repair only the existing report. Do not call more tools. ' + gate.last_error
            raise CoachingLimitError('Report attempt limit reached.')
        except Exception:
            if timed_out.is_set():
                raise CoachingLimitError('Codex coaching time limit reached. Engines and Codex have been closed.') from None
            raise
        finally:
            if active_request:
                # An interrupted request may have incurred unreported usage.
                budget.record_usage(estimate=request_estimate, source='codex_interrupted_turn_estimate')
            timer.cancel()
            client.close()
