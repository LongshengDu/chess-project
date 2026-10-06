"""One short Codex report from saved evidence; never scans a whole game."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coach.coach import failure_detail
from analysis.engine_session import Engines
from analysis.cache import write_json
from tests.coach.smoke_scenario import run_smoke
from tests.coach import smoke_settings as config
from coach.settings import CONFIG


def parser():
    cli = argparse.ArgumentParser(description='Test one 100–180 word Codex report using saved analysis.')
    cli.add_argument('analysis', type=Path, help='Path to a saved analysis.json with move evidence.')
    cli.add_argument('--output-dir', type=Path, help='Test output directory; defaults to tests/coach/output/<game>-codex-smoke.')
    cli.add_argument('--model', default=CONFIG['COACH']['CODEX']['MODEL'])
    cli.add_argument('--cache-dir', type=Path, default=CONFIG['ANALYSIS']['CACHE_DIR'])
    return cli


def main(argv=None):
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    args = parser().parse_args(argv)
    try:
        source = args.analysis.resolve()
        output = (args.output_dir or config.OUTPUT_DIR / (source.parent.name + '-codex-smoke')).resolve()
        if output == source.parent:
            raise ValueError('Use a separate smoke-test output directory to preserve the saved analysis and full report.')
        analysis = json.loads(source.read_text(encoding='utf-8'))
        if analysis.get('schema_version') not in (3, 4):
            raise ValueError('Regenerate the saved analysis with --analysis-only for the current move schema.')
        write_json(output / 'analysis.json', analysis)
        print(f'Small report test with {args.model}: one position, one short report via Codex.', flush=True)
        with Engines(analysis['start_fen'], args.cache_dir) as engines:
            run_smoke(analysis, engines, output, model_id=args.model)
        print(f'Saved: {output / "coaching-smoke.md"}', flush=True)
        print('Usage: ' + json.dumps(analysis['agent_run']['usage']), flush=True)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Local engines closed.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Smoke test failed: {failure_detail(exc)}. Local engines closed.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
