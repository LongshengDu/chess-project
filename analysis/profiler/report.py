"""Profiler reports: summarize recorded timings as tables, documents, and charts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.pgn


PHASES = {"screening": "Screen all moves", "engine_best": "Best engine move",
          "human_mid": "Human/played intermediate",
          "engine_top": "Top engine moves", "human_final": "Human/played to target",
          "exhaustive": "All moves to target"}
COLORS = {"screening": "#94c8e8", "engine_best": "#416a50", "human_mid": "#5aa89c", "engine_top": "#2c659d",
          "human_final": "#e6974c", "exhaustive": "#2c659d", "maia": "#9477b4", "other": "#bbc2cc"}
MAIA_PHASES = ("model_load", "prepare", "forward", "normalize_transfer", "format", "validate", "queue")
SEARCH_FIELDS = ('ply', 'position', 'phase', 'candidate', 'root_move', 'is_played_move',
                 'target_depth', 'achieved_depth', 'stop_reason', 'time_limit_seconds', 'multipv',
                 'wall_seconds', 'engine_reported_seconds', 'nodes', 'peak_nps', 'seldepth', 'hashfull_permille')


def load_timings(directory: Path):
    """Load current runtime timings, falling back to the historical filename."""
    directory = Path(directory)
    path = directory / 'runtime-profiler.json'
    historical = directory / 'profile.json'
    if not path.exists() and historical.is_file():
        path = historical
    return json.loads(path.read_text(encoding='utf-8')), path


def seconds(value):
    return value / 1000


def stockfish_allocation(configuration):
    """Describe recorded resources without assuming the original benchmark host."""
    if 'stockfish_threads_per_worker' in configuration:
        return (f"{configuration['stockfish_workers']} workers × "
                f"{configuration['stockfish_threads_per_worker']} threads "
                f"({configuration['stockfish_total_threads']} total)")
    return f"{configuration.get('stockfish_threads', 'unrecorded')} threads"


def summarize(timings):
    if timings['configuration'].get('pipeline') == 'shared':
        return _summarize_shared(timings)
    return _summarize_serial(timings)


def _summarize_serial(timings):
    bounded = timings['configuration'].get('strategy') == 'bounded'
    if not timings["complete"]:
        raise ValueError("The full analysis is not complete; refusing to publish a final report")
    if sorted(row["ply"] for row in timings["positions"]) != list(range(timings["configuration"]["total_positions"])):
        raise ValueError("The runtime timings do not contain every position exactly once")
    by_fen = defaultdict(list)
    for event in timings["events"]:
        if "fen" in event:
            by_fen[chess.Board(event["fen"]).fen()].append(event)
    rows, searches = [], []
    for position in sorted(timings["positions"], key=lambda row: row["ply"]):
        board = chess.Board(position["fen"])
        events = by_fen[board.fen()]
        maia = [event for event in events if event["kind"] == "maia"]
        sf = [event for event in events if event["kind"] == "stockfish"]
        if len(maia) != 1 or len(sf) != 1 or maia[0]["cache_hit"] or sf[0]["cache_hit"]:
            raise ValueError(f"Position {position['ply']} was not evaluated exactly once without cached results")
        maia, sf = maia[0], sf[0]
        target = timings["configuration"]["target_depth"]
        if not sf["complete"] or (sf.get("target_depth", sf["depth"]) if bounded else sf["depth"]) != target:
            raise ValueError("An engine result is incomplete")
        required_depth = 1 if bounded else target
        if position["move"] and sf["root_move_depth_vec"][position["move"]] < required_depth:
            raise ValueError("The played move did not reach the requested depth")
        candidates = sf.get("options", {}).get("maiaCandidateMoves", [])[:4]
        candidates += sf.get("options", {}).get("forcedCandidateMoves", [])
        if any(sf["root_move_depth_vec"].get(move, 0) < required_depth for move in candidates):
            raise ValueError("A selected candidate did not reach the requested depth")
        search_events = [event for event in events if event["kind"] == "stockfish_search"]
        def search_key(event):
            return (event["phase"], event["target_depth"], event["multipv"], event["root_move"])
        if Counter(search_key(event) for event in search_events if event["status"] == "started") != Counter(search_key(event) for event in search_events if event["status"] == "finished"):
            raise ValueError("A native search is missing its completion event")
        row = {"ply": position["ply"], "move": position["label"], "legal_moves": board.legal_moves.count(),
               "achieved_depth": sf["depth"], "target_reached": sf.get("target_reached", True),
               "stop_reason": sf.get("stop_reason", "depth"),
               "played_depth": sf.get("root_move_depth_vec", {}).get(position["move"]),
               "position_seconds": seconds(position["position_ms"]),
               "maia_http_seconds": seconds(position["maia_http_ms"]),
               "maia_server_seconds": seconds(maia["wall_ms"]),
               "stockfish_http_seconds": seconds(position["stockfish_http_ms"]),
               "stockfish_server_seconds": seconds(sf["wall_ms"]),
               "stockfish_startup_seconds": seconds(sf["startup_ms"]),
               "stockfish_queue_seconds": seconds(sf["queue_ms"]),
               "tree_update_seconds": seconds(position["tree_update_ms"]),
               "streamed_frames": position["frames"], "fen": position["fen"]}
        for key in ("model_load", "prepare", "forward", "normalize_transfer", "format", "validate", "queue"):
            row[f"maia_{key}_seconds"] = seconds(maia.get(key + "_ms", 0))
        for phase in PHASES:
            row[phase + "_seconds"] = 0.
        row["nodes"] = 0
        for event in events:
            if event["kind"] != "stockfish_search" or event["status"] != "finished":
                continue
            if event["cancelled"] or (event["achieved_depth"] < event["target_depth"] and not
                    (bounded and event.get("stop_reason") == "time" and event.get("time_limit_seconds", 0) > 0)):
                raise ValueError("A search was cancelled or did not finish")
            root = event["root_move"]
            search = {"ply": row["ply"], "position": row["move"], "phase": event["phase"],
                      "candidate": board.san(chess.Move.from_uci(root)) if root else "MultiPV",
                      "root_move": root, "is_played_move": bool(root and root == position["move"]),
                      "target_depth": event["target_depth"], "achieved_depth": event["achieved_depth"],
                      "stop_reason": event.get("stop_reason", "depth"), "time_limit_seconds": event.get("time_limit_seconds"),
                      "multipv": event["multipv"], "wall_seconds": seconds(event["wall_ms"]),
                      "engine_reported_seconds": event.get("time", 0), "nodes": event.get("nodes", 0),
                      "peak_nps": event.get("nps", 0), "seldepth": event.get("seldepth", 0),
                      "hashfull_permille": event.get("hashfull", 0)}
            searches.append(search)
            row[event["phase"] + "_seconds"] += search["wall_seconds"]
            row["nodes"] += search["nodes"]
        row["other_seconds"] = row["position_seconds"] - row["maia_server_seconds"] - row["stockfish_startup_seconds"] - sum(row[p + "_seconds"] for p in PHASES)
        rows.append(row)
    total = seconds(timings["browser"]["browser_run_ms"])
    totals = {phase: sum(row[phase + "_seconds"] for row in rows) for phase in PHASES}
    totals.update(maia=sum(row["maia_server_seconds"] for row in rows),
                  startup=sum(row["stockfish_startup_seconds"] for row in rows))
    totals["other"] = total - sum(totals.values())
    return {"total_seconds": total, "components_seconds": totals, "positions": rows, "searches": searches,
            "save_seconds": seconds(timings["browser"]["save_ms"]),
            "between_positions_seconds": total - sum(row["position_seconds"] for row in rows)}


def _summarize_shared(timings):
    """Keep batch wall time separate from concurrent per-position worker time."""
    if not timings['complete']:
        raise ValueError('The full analysis is not complete; refusing to publish a final report')
    positions = sorted(timings['positions'], key=lambda row: row['ply'])
    if [row['ply'] for row in positions] != list(range(timings['configuration']['total_positions'])):
        raise ValueError('The runtime timings do not contain every position exactly once')
    by_ply = defaultdict(list)
    for event in timings['events']:
        if event['kind'] in ('stockfish', 'stockfish_search'):
            ply = event.get('ply')
            if type(ply) is not int or not 0 <= ply < len(positions):
                raise ValueError('Every Stockfish event must identify its game ply')
            if chess.Board(event['fen']).fen() != chess.Board(positions[ply]['fen']).fen():
                raise ValueError('Stockfish event does not match its recorded game position')
            by_ply[ply].append(event)
    rows, searches = [], []
    bounded = timings['configuration'].get('strategy') == 'bounded'
    target = timings['configuration']['target_depth']
    for position in positions:
        board = chess.Board(position['fen'])
        events = by_ply[position['ply']]
        completions = [event for event in events if event['kind'] == 'stockfish']
        if len(completions) != 1 or not position.get('complete'):
            raise ValueError('Every position needs exactly one completed Stockfish result')
        sf = completions[0]
        terminal = sf.get('terminal', False)
        if not sf.get('complete') or terminal != board.is_game_over(claim_draw=False):
            raise ValueError('An engine result is incomplete or has an invalid terminal marker')
        if not terminal and (sf.get('target_depth', sf['depth']) if bounded else sf['depth']) != target:
            raise ValueError('An engine result is incomplete')
        depths = sf.get('root_move_depth_vec', {})
        played = position.get('move')
        candidates = sf.get('options', {}).get('maiaCandidateMoves', [])[:4]
        candidates += sf.get('options', {}).get('forcedCandidateMoves', [])
        if played:
            candidates.append(played)
        if not terminal and any(depths.get(move, 0) < (1 if bounded else target) for move in candidates):
            raise ValueError('A selected candidate did not reach the requested depth')
        native = [event for event in events if event['kind'] == 'stockfish_search']
        def search_key(event):
            return event['phase'], event['target_depth'], event['multipv'], event.get('root_move')
        if Counter(search_key(event) for event in native if event['status'] == 'started') != Counter(
                search_key(event) for event in native if event['status'] == 'finished'):
            raise ValueError('A native search is missing its completion event')
        if (terminal or sf.get('cache_hit')) and native:
            raise ValueError('Cached and terminal results must not contain measured native searches')
        if not terminal and not sf.get('cache_hit') and not native:
            raise ValueError('An uncached result is missing its native search timing')
        row = {'ply': position['ply'], 'move': position['label'], 'legal_moves': board.legal_moves.count(),
               'achieved_depth': sf['depth'], 'target_reached': sf.get('target_reached', True),
               'stop_reason': sf.get('stop_reason', 'terminal' if terminal else 'depth'),
               'played_depth': depths.get(played), 'position_seconds': seconds(sf['wall_ms']),
               'maia_http_seconds': None, 'maia_server_seconds': None,
               'stockfish_http_seconds': None, 'stockfish_server_seconds': seconds(sf['wall_ms']),
               'stockfish_startup_seconds': None, 'stockfish_queue_seconds': None,
               'stockfish_acquire_seconds': seconds(sf.get('acquire_ms', 0)),
               'tree_update_seconds': None, 'streamed_frames': None, 'fen': position['fen'],
               'completed_at_seconds': seconds(position['elapsed_ms']),
               'cache_hit': bool(sf.get('cache_hit')), 'terminal': terminal}
        row.update({f'maia_{phase}_seconds': None for phase in MAIA_PHASES})
        row.update({phase+'_seconds': 0. for phase in PHASES})
        row.update(nodes=0, other_seconds=None)
        for event in native:
            if event['status'] != 'finished':
                continue
            if event.get('cancelled') or (event['achieved_depth'] < event['target_depth'] and not (
                    bounded and event.get('stop_reason') == 'time' and event.get('time_limit_seconds', 0) > 0)):
                raise ValueError('A search was cancelled or did not finish')
            root = event.get('root_move')
            if event['phase'] not in PHASES:
                raise ValueError(f"Unknown native search phase: {event['phase']}")
            search = {'ply': row['ply'], 'position': row['move'], 'phase': event['phase'],
                      'candidate': board.san(chess.Move.from_uci(root)) if root else 'MultiPV',
                      'root_move': root, 'is_played_move': bool(root and root == played),
                      'target_depth': event['target_depth'], 'achieved_depth': event['achieved_depth'],
                      'stop_reason': event.get('stop_reason', 'depth'),
                      'time_limit_seconds': event.get('time_limit_seconds'), 'multipv': event['multipv'],
                      'wall_seconds': seconds(event['wall_ms']), 'engine_reported_seconds': event.get('time', 0),
                      'nodes': event.get('nodes', 0), 'peak_nps': event.get('nps', 0),
                      'seldepth': event.get('seldepth', 0), 'hashfull_permille': event.get('hashfull', 0)}
            searches.append(search)
            row[event['phase']+'_seconds'] += search['wall_seconds']
            row['nodes'] += search['nodes']
        rows.append(row)
    batches = []
    for event in timings['events']:
        if event['kind'] != 'maia_batch':
            continue
        for item in event['positions']:
            ply, fen = item['ply'], item['fen']
            if type(ply) is not int or not 0 <= ply < len(positions) or chess.Board(fen).fen() != chess.Board(positions[ply]['fen']).fen():
                raise ValueError('Maia batch does not match its recorded game positions')
        breakdown = event.get('profile', event)
        batches.append({'batch': len(batches), 'positions': event['positions'],
                        'rating_pairs_inferred': event['batch_size'], 'wall_seconds': seconds(event['wall_ms']),
                        'inference_seconds': seconds(event['inference_ms']),
                        **{key+'_seconds': seconds(breakdown.get(key+'_ms', 0)) for key in MAIA_PHASES}})
    totals = {phase: sum(row[phase+'_seconds'] for row in rows) for phase in PHASES}
    totals.update(maia=sum(batch['wall_seconds'] for batch in batches),
                  player_rating=sum(seconds(event['wall_ms']) for event in timings['events'] if event['kind'] == 'player_rating'),
                  startup=None, other=None)
    browser = timings.get('browser', {})
    total_ms = browser.get('browser_run_ms', timings.get('server_wall_ms'))
    if total_ms is None:
        raise ValueError('Complete runtime timings need measured total wall time')
    return {'timing_mode': 'parallel', 'components_additive': False, 'total_seconds': seconds(total_ms),
            'components_seconds': totals, 'positions': rows, 'searches': searches, 'maia_batches': batches,
            'maia_details_seconds': {key: sum(batch[key+'_seconds'] for batch in batches) for key in MAIA_PHASES},
            'maia_inference_seconds': sum(batch['inference_seconds'] for batch in batches),
            'pipeline_seconds': next((seconds(event['wall_ms']) for event in timings['events']
                                      if event['kind'] == 'game_analysis_completed'), None),
            'save_seconds': seconds(browser['save_ms']) if 'save_ms' in browser else None,
            'between_positions_seconds': None}


def write_csv(path, rows, *, fieldnames=None):
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        if not rows and fieldnames is None:
            return
        writer = csv.DictWriter(output, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def chart(report, directory):
    if report.get('timing_mode') == 'parallel':
        return shared_chart(report, directory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["positions"]
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig = plt.figure(figsize=(17, 11), facecolor="#f7f9fc", layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax = fig.add_subplot(grid[0, :])
    phases = [p for p in PHASES if report["components_seconds"][p] > 0]
    def bars(axis, selected, horizontal=False):
        baseline = [0.] * len(selected)
        for phase in [*phases, "maia", "other"]:
            key = "maia_server_seconds" if phase == "maia" else phase + "_seconds"
            values = [max(0, row[key] + (row["stockfish_startup_seconds"] if phase == "other" else 0)) for row in selected]
            label = PHASES.get(phase, {"maia":"Maia (21 ratings)", "other":"Other / transport / UI"}.get(phase))
            if horizontal:
                axis.barh(range(len(selected)), values, left=baseline, color=COLORS[phase], label=label, height=.7)
            else:
                axis.bar(range(len(selected)), values, bottom=baseline, color=COLORS[phase], label=label, width=.8)
            baseline = [a + b for a, b in zip(baseline, values)]
    bars(ax, rows)
    ax.set_xticks(range(len(rows)), [row["move"] for row in rows], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Seconds per position")
    ax.set_title(f"All {len(rows)} positions — timing the position before each displayed move", loc="left", fontsize=14, pad=14)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, frameon=False, fontsize=9, loc="upper left")
    slow = sorted(rows, key=lambda row: row["position_seconds"], reverse=True)[:10][::-1]
    bottom = fig.add_subplot(grid[1, 0])
    bars(bottom, slow, True)
    bottom.set_yticks(range(len(slow)), [row["move"] for row in slow])
    bottom.set_xlabel("Seconds")
    bottom.set_title("Ten slowest positions", loc="left", fontsize=14)
    right = fig.add_subplot(grid[1, 1])
    comps = [(PHASES[p], report["components_seconds"][p], COLORS[p]) for p in phases]
    comps += [("Maia (including cold load)", report["components_seconds"]["maia"], COLORS["maia"]),
              ("Startup + other overhead", report["components_seconds"]["startup"] + report["components_seconds"]["other"], COLORS["other"])]
    right.barh([name for name, _, _ in comps][::-1], [value for _, value, _ in comps][::-1], color=[color for _, _, color in comps][::-1])
    right.set_xlabel("Seconds across the complete run")
    right.set_title("Where the total time went", loc="left", fontsize=14)
    for i, (_, value, _) in enumerate(comps[::-1]):
        right.text(value + max(v for _, v, _ in comps) * .015, i, f"{value:.2f}s", va="center", fontsize=10)
    right.set_xlim(0, max(v for _, v, _ in comps) * 1.22)
    config = report['configuration']
    allocation = stockfish_allocation(config)
    fig.suptitle(f"{report['source_name']}  /  {config['strategy']} d{config['target_depth']} target  /  {report['total_seconds'] / 60:.2f} minutes\n{config['model']} on {config['device']} · Stockfish, {allocation} · no cached evaluations", fontsize=19, fontweight="bold", x=.04, ha="left")
    fig.savefig(directory / "timings.png", dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(directory / "timings.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def shared_chart(report, directory):
    """Chart measured worker jobs and batches without constructing a false partition."""
    rows = report['positions']
    measured = shared_components(report)
    config = report['configuration']
    height = 220 + 30*len(measured) + 26*len(rows)
    content = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 {height}" role="img">',
               '<title>Measured Stockfish jobs and Maia batches</title>',
               '<rect width="100%" height="100%" fill="#f7f9fc"/>',
               '<style>text{font:14px system-ui,sans-serif;fill:#203047}.title{font-size:23px;font-weight:bold}</style>']

    def label(x, y, value, css=''):
        content.append(f'<text x="{x}" y="{y}" class="{css}">{html.escape(str(value))}</text>')

    def bar(y, value, maximum, color, start=0.):
        content.append(f'<rect x="{260+820*start/maximum:.2f}" y="{y-15}" '
                       f'width="{820*value/maximum:.2f}" height="18" fill="{color}"/>')

    label(25, 38, f"{report['source_name']} · {report['total_seconds']:.3f}s elapsed", 'title')
    label(25, 66, f"{config.get('strategy', 'staged')} / depth {config['target_depth']} ceiling / "
                  f"{config.get('model', 'Maia')} on {config.get('device', '?')} / "
                  f"{config.get('stockfish_workers', '?')} workers x {config.get('stockfish_threads_per_worker', '?')} threads")
    label(25, 100, 'Accumulated worker and batch seconds; concurrent measurements are not an elapsed-time partition.')
    maximum = max([value for _, value in measured] + [.001])
    y = 132
    for name, value in measured:
        label(25, y, name)
        bar(y, value, maximum, '#416a50')
        label(1090, y, f'{value:.3f}s')
        y += 30
    y += 36
    label(25, y, 'Native search phases for each position before the displayed move; jobs overlap in time.')
    y += 30
    maximum = max([sum(row[phase+'_seconds'] for phase in PHASES) for row in rows] + [.001])
    for row in rows:
        label(25, y, f"{row['ply']}: {row['move']}")
        baseline = 0.
        for phase in PHASES:
            value = row[phase+'_seconds']
            bar(y, value, maximum, COLORS[phase], baseline)
            baseline += value
        label(1090, y, f'{baseline:.3f}s')
        y += 26
    content.append('</svg>')
    (directory/'timings.svg').write_text('\n'.join(content), encoding='utf-8')


def shared_components(report):
    labels = {**PHASES, 'maia': 'Maia batches', 'player_rating': 'Player-rating fit'}
    return [(labels[key], value) for key, value in report['components_seconds'].items()
            if value is not None and value > 0]


def write_shared_report(report, timings, directory, *, raw_filename):
    """Publish measured facts, leaving timing unavailable at this granularity blank."""
    def display(value):
        return '—' if value is None else f'{value:.3f}' if isinstance(value, float) else str(value)

    config = report['configuration']
    title = f"Full {report['source_name']} runtime timings"
    notes = [
        'Stockfish component totals are accumulated worker wall time. Positions run concurrently, so totals can exceed overall elapsed time and are not percentages or partitions of it.',
        'Per-position duration measures the Stockfish job, including acquiring a worker. Completion time is relative to run start, not a position duration.',
        'Maia is measured per batch. Batch wall includes cache lookup, inference and result handling. Inference wall and internal phases are included in that batch total, not additional costs. Independent per-position Maia timings are not available.',
        'Worker acquisition combines queue wait and process startup; these are not measured separately. Blank CSV values mean unmeasured, not zero. Unattributed overhead is not computed from overlapping totals.',
        'Move labels identify the next move to play. The final row is the position after the last move. Terminal positions need no native search; cached results have no newly measured native searches.',
        'Depth is the achieved engine depth. Bounded search stops at its depth or time limit. The FEN sequence was checked against the PGN. This is one run, not a repeated statistical benchmark.',
    ]
    columns = [('ply', 'Ply'), ('move', 'Move to play'), ('position_seconds', 'SF job s'),
               ('stockfish_acquire_seconds', 'Acquire s'), ('completed_at_seconds', 'Completed at s'),
               ('achieved_depth', 'Depth'), ('target_reached', 'Target met'), ('cache_hit', 'Cached')]
    columns += [(phase+'_seconds', label+' s') for phase, label in PHASES.items()
                if report['components_seconds'][phase] > 0]
    lines = [f'# {title}', '', f"Complete run: **{report['total_seconds']:.3f} seconds**.", '',
             f"Started {timings.get('started_utc', 'unrecorded')}; finished {timings.get('finished_utc', 'unrecorded')}.", '',
             f"Configuration: `{json.dumps(config)}`.", '', '![Measured timings](timings.svg)', '',
             '## Component measurements', '', '**These measurements are not additive elapsed time.**', '',
             '| Component | Measured seconds |', '| --- | ---: |']
    lines += [f'| {name} | {value:.3f} |' for name, value in shared_components(report)]
    lines += ['', f"Shared pipeline wall: {display(report['pipeline_seconds'])} s. "
              f"Maia inference-call wall: {report['maia_inference_seconds']:.3f} s.", '',
              '## Maia batches', '', '| Batch | Game plies | New rating pairs | Wall s | Inference s |',
              '| --- | --- | ---: | ---: | ---: |']
    for batch in report['maia_batches']:
        plies = ', '.join(str(item['ply']) for item in batch['positions'])
        lines.append(f"| {batch['batch']} | {plies} | {batch['rating_pairs_inferred']} | {batch['wall_seconds']:.3f} | {batch['inference_seconds']:.3f} |")
    lines += ['', '## Maia internal phases', '', 'Included in batch time above.', '',
              '| Phase | Seconds |', '| --- | ---: |']
    lines += [f"| {key.replace('_', ' ')} | {value:.6f} |" for key, value in report['maia_details_seconds'].items()]
    lines += ['', '## Every position', '', '| '+' | '.join(label for _, label in columns)+' |',
              '| '+' | '.join('---' for _ in columns)+' |']
    lines += ['| '+' | '.join(display(row[key]) for key, _ in columns)+' |' for row in report['positions']]
    lines += ['', '## Slowest native searches', '', '| Position before | Phase | Candidate | Seconds | Nodes |',
              '| --- | --- | --- | ---: | ---: |']
    for search in sorted(report['searches'], key=lambda row: row['wall_seconds'], reverse=True)[:15]:
        lines.append(f"| {search['position']} | {search['phase']} | {search['candidate']} | {search['wall_seconds']:.3f} | {search['nodes']} |")
    lines += ['', '## Timing boundaries', '', *['- '+note for note in notes], '',
              f"Save: {display(report['save_seconds'])} s. Source SHA-256: `{report['source_sha256']}`.", '',
              f'Data: [positions](positions.csv), [native searches](searches.csv), [raw runtime timings]({raw_filename}), [summary](summary.json).', '']
    (directory/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    headings = ''.join(f'<th>{html.escape(label)}</th>' for _, label in columns)
    table = ''.join('<tr>'+''.join(f'<td>{html.escape(display(row[key]))}</td>' for key, _ in columns)+'</tr>'
                    for row in report['positions'])
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{html.escape(title)}</title><style>body{{font:16px system-ui,sans-serif;background:#f7f9fc;color:#203047;max-width:1450px;margin:40px auto;padding:0 24px}}p,li{{line-height:1.6}}img{{width:100%}}table{{border-collapse:collapse;background:white;width:100%}}th,td{{padding:10px;border-bottom:1px solid #dfe5ed;text-align:right}}th{{cursor:pointer}}td:nth-child(2){{text-align:left}}.table{{overflow:auto}}</style>
<h1>{html.escape(title)}</h1><p><strong>{report['total_seconds']:.3f} seconds elapsed</strong> · {len(report['positions'])} positions · {html.escape(config.get('strategy', 'staged'))} · depth {config['target_depth']} ceiling</p>
<p><a href="REPORT.md">Detailed report</a> · <a href="positions.csv">Position CSV</a> · <a href="searches.csv">Search CSV</a> · <a href="{html.escape(raw_filename, quote=True)}">Raw runtime timings</a></p>
<img src="timings.svg" alt="Measured position jobs and component timings"><ul>{''.join('<li>'+html.escape(note)+'</li>' for note in notes)}</ul>
<div class="table"><table><thead><tr>{headings}</tr></thead><tbody>{table}</tbody></table></div>
<script>document.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{{let b=document.querySelector('tbody'),d=h.dataset.direction==='asc'?-1:1;h.dataset.direction=d===1?'asc':'desc';[...b.rows].sort((a,c)=>{{let x=a.cells[i].textContent,y=c.cells[i].textContent;return d*(isNaN(Number(x))||isNaN(Number(y))?x.localeCompare(y):Number(x)-Number(y))}}).forEach(r=>b.append(r))}})</script></html>'''
    (directory/'report.html').write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--pgn", type=Path, default=Path("example.pgn"))
    args = parser.parse_args()
    timings, raw_path = load_timings(args.directory)
    report = summarize(timings)
    with args.pgn.open(encoding="utf-8-sig") as source:
        game = chess.pgn.read_game(source)
    boards = [game.board()]
    for move in game.mainline_moves():
        board = boards[-1].copy()
        board.push(move)
        boards.append(board)
    assert [chess.Board(row["fen"]).fen() for row in report["positions"]] == [board.fen() for board in boards]
    report["source_sha256"] = hashlib.sha256(args.pgn.read_bytes()).hexdigest()
    report["configuration"] = timings["configuration"]
    report["source_name"] = args.pgn.name
    bounded = timings['configuration']['strategy'] == 'bounded'
    target = timings['configuration']['target_depth']
    coverage = (f"All {len(report['positions'])} positions completed their search policy. " +
                ("Time-limited candidates retain their actual depths; the target is a ceiling, not a guarantee. " if bounded else
                 f"Every selected candidate and played move reached d{target}. ") +
                "Other legal moves retain shallower screening scores. Terminal positions require no Stockfish search. The FEN sequence was checked against the source PGN.")
    (args.directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(args.directory / "positions.csv", report["positions"])
    write_csv(args.directory / "searches.csv", report["searches"], fieldnames=SEARCH_FIELDS)
    chart(report, args.directory)
    if report.get('timing_mode') == 'parallel':
        write_shared_report(report, timings, args.directory, raw_filename=raw_path.name)
        print(json.dumps({'total_seconds': report['total_seconds'], 'components': report['components_seconds']}, indent=2))
        return
    rows = report["positions"]
    slow = sorted(rows, key=lambda row: row["position_seconds"], reverse=True)
    components = [(PHASES[p], report["components_seconds"][p]) for p in PHASES if report["components_seconds"][p] > 0]
    components += [("Maia inference", report["components_seconds"]["maia"]),
                   ("Stockfish process startup", report["components_seconds"]["startup"]),
                   ("Other: transport, UI, adapters, save, profiling", report["components_seconds"]["other"])]
    lines = [f"# Full {args.pgn.name} runtime timings", "",
        f"Complete uncached browser run: **{report['total_seconds']:.3f} seconds ({report['total_seconds']/60:.2f} minutes)**.",
        "", f"Started {timings['started_utc']}; finished {timings['finished_utc']}.", "",
        f"Depth target {target}, {timings['configuration']['strategy']} search. Configuration: `{json.dumps(timings['configuration'])}`.",
        "", coverage,
        "", "![Per-position and component timing](timings.png)", "", "## Component totals", "",
        "| Component | Seconds | Share of button-to-save time |", "| --- | ---: | ---: |"]
    for name, value in components:
        lines.append(f"| {name} | {value:.3f} | {value/report['total_seconds']:.2%} |")
    slowest_search = max(report["searches"], key=lambda row: row["wall_seconds"], default=None)
    slow_three = sum(row["position_seconds"] for row in slow[:3])
    sf_total = sum(report["components_seconds"][phase] for phase in PHASES)
    lines += ["", "## Findings", "",
        f"- Stockfish search accounts for **{sf_total/report['total_seconds']:.2%}** of recorded elapsed time; Maia accounts for {report['components_seconds']['maia']/report['total_seconds']:.2%}.",
        f"- The three slowest positions account for **{slow_three/report['total_seconds']:.2%}** of total time; whole-game averages hide this concentration.",
        f"- Recorded Stockfish allocation: {stockfish_allocation(timings['configuration'])}.",
        ("- The bounded policy uses both depth and time limits. Achieved depths and saved scores describe its coverage tradeoff." if bounded else
         "- This depth-targeted runtime measurement records the search policy and achieved depths. Depth alone does not establish a wall-clock limit.")]
    if slowest_search is not None:
        lines.append(f"- The largest individual search is {slowest_search['phase']} before **{slowest_search['position']}**: {slowest_search['wall_seconds']:.3f}s measured versus {slowest_search['engine_reported_seconds']:.3f}s reported by Stockfish, {slowest_search['nodes']:,} nodes, MultiPV {slowest_search['multipv']}, target depth {slowest_search['target_depth']}, selective depth {slowest_search['seldepth']}.")
    else:
        lines.append('- No native searches were recorded; terminal positions do not require a search.')
    lines += ["", "## Maia details", "", "These timings are included in the Maia total above, not additional costs.", "",
              "| Component | Total seconds |", "| --- | ---: |"]
    for key in ("model_load", "prepare", "forward", "normalize_transfer", "format", "validate", "queue"):
        lines.append(f"| {key.replace('_',' ')} | {sum(row['maia_'+key+'_seconds'] for row in rows):.6f} |")
    lines += ["", "## Every position", "", "Move labels identify the **next move to play**: the row is the cost of analyzing the position before that move. Maia values are milliseconds; other timing columns are seconds. Depth is the achieved main engine depth; played-move and per-search depths are in the CSVs.", "",
              "| Move to play | Depth | Target met | Total s | Maia ms | Screen s | Best s | Human mid s | Engine top s | Human final s | Other s |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['move']} | {row['achieved_depth']} | {row['target_reached']} | {row['position_seconds']:.3f} | {row['maia_server_seconds']*1000:.1f} | {row['screening_seconds']:.3f} | {row['engine_best_seconds']:.3f} | {row['human_mid_seconds']:.3f} | {row['engine_top_seconds']:.3f} | {row['human_final_seconds']:.3f} | {row['other_seconds']+row['stockfish_startup_seconds']:.3f} |")
    lines += ["", "## Slowest individual native searches", "",
              "| Position before | Stage | Candidate | Played? | Seconds | Nodes (million) |", "| --- | --- | --- | --- | ---: | ---: |"]
    for search in sorted(report["searches"], key=lambda row: row["wall_seconds"], reverse=True)[:15]:
        lines.append(f"| {search['position']} | {search['phase']} | {search['candidate']} | {'yes' if search['is_played_move'] else ''} | {search['wall_seconds']:.3f} | {search['nodes']/1e6:.2f} |")
    lines += ["", "## Timing boundaries and limitations", "",
        "- The total is measured in the browser from Start Analysis until the final study save finishes, including profiling setup and the per-position logging calls. Import, page loading, and report generation are excluded.",
        "- Maia timing separates the initial model load, preparation, model forward pass, normalization/output transfer, and Python result formatting. When using CUDA, synchronization at profiling boundaries adds instrumentation overhead.",
        "- Stockfish phase timings measure the actual native search stream, including Python handling and any response backpressure. Native-reported search time, nodes, peak NPS, selective depth, and hash occupancy are also available in searches.csv.",
        "- Browser HTTP timings and tree-update timings in positions.csv overlap with engine time. Do not add them to the component totals. Render/network/serialization time is reported as combined residual overhead, not falsely attributed to a single component.",
        f"- Final save: {report['save_seconds']:.3f}s. Time outside the per-position intervals: {report['between_positions_seconds']:.3f}s (includes save, profiler setup, and profiling round trips).",
        "- Runtime timing validation requires uncached position results and completed native searches. " + (f"Recorded per-position search budget: {timings['configuration']['position_budget_seconds']}s." if bounded else "Search configuration is recorded above; this historical event format does not record every process watchdog limit."),
        "- This is one complete run on a live desktop, not a statistical benchmark. Stockfish multithreaded searches and timing vary between runs.",
        f"- Source SHA-256: `{report['source_sha256']}`.", "",
        f"Downloads: [per-position CSV](positions.csv), [every native search CSV](searches.csv), [full raw runtime timings]({raw_path.name}), [summary JSON](summary.json), [vector chart](timings.svg).", ""]
    (args.directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    table_rows = []
    for row in rows:
        fields = [row['ply'], row['move'], *[round(row[key], 3) for key in ('position_seconds','maia_server_seconds','screening_seconds','engine_best_seconds','human_mid_seconds','engine_top_seconds','human_final_seconds')], round(row['other_seconds']+row['stockfish_startup_seconds'],3), row['achieved_depth']]
        table_rows.append('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in fields) + '</tr>')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>REPORT_TITLE</title>
<style>body{font:16px system-ui,sans-serif;color:#203047;background:#f7f9fc;max-width:1450px;margin:40px auto;padding:0 24px}h1{font-size:36px}p{line-height:1.6}img{width:100%;background:white}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px 12px;text-align:right;border-bottom:1px solid #dfe5ed}th{cursor:pointer;background:#e7eef7;position:sticky;top:0}td:nth-child(2),th:nth-child(2){text-align:left}tr:hover{background:#eef5fc}a{color:#21568b}.table{overflow:auto}footer{margin:35px 0;color:#53647b}</style>
<h1>REPORT_TITLE</h1><p><strong>TOTAL</strong> · CONFIGURATION</p>
<p>Each row times the position <strong>before the displayed move</strong>. COVERAGE Click a table heading to sort.</p>
<p><a href="REPORT.md">Detailed report</a> · <a href="positions.csv">All position metrics (CSV)</a> · <a href="searches.csv">Every native search (CSV)</a> · <a href="RAW_FILE">Raw runtime timings</a></p><img src="timings.svg" alt="Per-position timing and component breakdown">
<div class="table"><table><thead><tr>HEADERS</tr></thead><tbody>ROWS</tbody></table></div>
<footer>All table times are seconds. “Other” includes transport, adapter, and browser overhead. Engine and browser timings overlap; consult the detailed report for timing boundaries. This was one run on a live desktop.</footer>
<script>document.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{let b=document.querySelector('tbody'),d=h.dataset.direction==='asc'?-1:1;h.dataset.direction=d===1?'asc':'desc';[...b.rows].sort((a,c)=>{let x=a.cells[i].textContent,y=c.cells[i].textContent;return d*(i===1?x.localeCompare(y):Number(x)-Number(y))}).forEach(r=>b.append(r))})</script></html>'''
    document = document.replace('TOTAL', f"{report['total_seconds']:.2f} seconds / {report['total_seconds']/60:.2f} minutes")
    document = document.replace('RAW_FILE', html.escape(raw_path.name, quote=True))
    document = document.replace('REPORT_TITLE', html.escape(f"{args.pgn.name} · {timings['configuration']['strategy']} analysis / d{target} target"))
    config = timings['configuration']
    document = document.replace('CONFIGURATION', html.escape(
        f"{len(rows)} positions · {config.get('model', 'Maia')} on {config.get('device', 'unrecorded device')} · "
        f"Stockfish, {stockfish_allocation(config)} · no cached evaluations"))
    document = document.replace('COVERAGE', html.escape(coverage))
    document = document.replace('HEADERS', ''.join(f'<th>{name}</th>' for name in ['Ply','Move to play','Total','Maia','Screen','Best','Human mid','Engine top','Human final','Other','Depth']))
    document = document.replace('ROWS', '\n'.join(table_rows))
    (args.directory / "report.html").write_text(document, encoding="utf-8")
    print(json.dumps({"total_seconds":report['total_seconds'], "components":report['components_seconds'],
                      "slowest_positions":[(r['move'],r['position_seconds']) for r in slow[:8]]}, indent=2))


if __name__ == "__main__":
    main()
