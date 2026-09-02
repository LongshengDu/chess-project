// Focused adaptation of Lila's analyse tools, ceval, and explorer move table.
import type { VNode } from 'snabbdom';

import type { AnalysisMove, PositionAnalysis } from '../../types';
import { h, onInsert } from '../snabbdom';
import { renderScore } from './eval';

export interface AnalysisToolsOpts {
  result?: PositionAnalysis;
  moves: VNode;
  loading: boolean;
  loadingText?: string;
  error?: string;
  onHover: (uci?: string) => void;
  onSelect: (uci: string) => void;
  continuationUcis?: string[];
}

// Native-DOM version of analyse/src/explorer/explorerUtil.ts:moveArrowAttributes.
function moveArrowAttributes(
  onHover: (uci?: string) => void,
  onSelect: (uci: string) => void,
) {
  const rowUci = (target: EventTarget | null): string | undefined => {
    const row = target instanceof Element ? target.closest<HTMLTableRowElement>('tr[data-uci]') : null;
    return row?.dataset.uci || undefined;
  };

  return onInsert<HTMLTableSectionElement>(element => {
    element.addEventListener('pointerover', event => onHover(rowUci(event.target)));
    element.addEventListener('pointerleave', () => onHover());
    element.addEventListener('focusin', event => onHover(rowUci(event.target)));
    element.addEventListener('focusout', event => {
      if (!element.contains(event.relatedTarget as Node | null)) onHover();
    });
    element.addEventListener('click', event => {
      const uci = rowUci(event.target);
      if (uci) onSelect(uci);
    });
    element.addEventListener('keydown', event => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      const uci = rowUci(event.target);
      if (!uci) return;
      event.preventDefault();
      onSelect(uci);
    });
  });
}

function probabilityBar(probability: number): VNode {
  const percent = Math.max(0, Math.min(100, probability * 100));
  return h('div.bar.maia-bar', [
    h('span.maia', {
      attrs: {
        style: `width: ${percent}%`,
        title: `${percent.toFixed(1)}% Maia probability`,
      },
    }),
    h('span.remainder', { attrs: { style: `width: ${100 - percent}%` } }),
  ]);
}

function moveRow(move: AnalysisMove, continuation: boolean): VNode {
  return h(
    `tr${move.played ? '.played' : ''}${continuation ? '.continuation' : ''}`,
    {
      key: move.uci,
      attrs: {
        'data-uci': move.uci,
        tabindex: '0',
        title: `Analyse ${move.san} as a variation`,
      },
    },
    [
      h('td.move-san', [
        h('span', move.san),
        move.played && h('em.pgn', 'PGN'),
        continuation && !move.played && h('em.line', 'LINE'),
      ]),
      h('td', `${(move.probability * 100).toFixed(1)}%`),
      h('td', `#${move.rank}`),
      h('td.eval', renderScore(move.stockfish)),
      h('td', probabilityBar(move.probability)),
    ],
  );
}

// Adapted from analyse/src/explorer/explorerView.ts:showMoveTable.
function showMoveTable(opts: AnalysisToolsOpts, result: PositionAnalysis): VNode {
  const otherMoveCount = Math.max(0, result.legalMoveCount - result.moves.length);
  const continuations = new Set(opts.continuationUcis);
  return h('table.moves', [
    h('thead', [
      h('tr', [
        h('th', 'Move'),
        h('th', { attrs: { colspan: 2 } }, 'Maia 1500'),
        h('th', 'Stockfish'),
        h('th', 'Probability'),
      ]),
    ]),
    h('tbody', { hook: moveArrowAttributes(opts.onHover, opts.onSelect) }, [
      ...result.moves.map(move => moveRow(move, continuations.has(move.uci))),
      result.otherProbability > 0.0005 &&
        h('tr.sum.other', [
          h('td', 'Other'),
          h('td', `${(result.otherProbability * 100).toFixed(1)}%`),
          h('td', `${otherMoveCount} moves`),
          h('td.eval', '—'),
          h('td', probabilityBar(result.otherProbability)),
        ]),
    ]),
  ]);
}

// Adapted from ui/lib/src/ceval/view/main.ts:renderCeval.
function renderCeval(opts: AnalysisToolsOpts): VNode {
  const lead = opts.result?.moves[0];
  const depth = opts.result?.stockfishDepth;
  const turn = opts.result?.turn;
  return h(`div.ceval.enabled${opts.loading ? '.computing' : ''}`, [
    h(
      'pearl',
      {
        attrs: {
          title: lead
            ? `Stockfish evaluation after Maia's most likely move, ${lead.san}`
            : 'Stockfish evaluation',
        },
      },
      lead ? renderScore(lead.stockfish) : h('icon.ddloader'),
    ),
    h('div.engine', [
      h('div', lead ? `Maia #1 · ${lead.san}` : 'Maia human move model'),
      h(
        'div.info',
        opts.loading
          ? opts.loadingText ?? 'Analysing…'
          : opts.result
            ? `Stockfish · depth ${depth ?? '—'} · ${turn} POV`
            : 'Maia3-5M · 1500 Elo',
      ),
    ]),
    h('div.bar', [h('span', { attrs: { style: `width: ${opts.loading ? 100 : 0}%` } })]),
  ]);
}

function explorerTitle(result?: PositionAnalysis): VNode {
  return h('div.explorer-title', [
    h('span.active.maia', [h('strong', `Maia ${result?.elo ?? 1500}`), ' human moves']),
    result && h('span.legal', `${result.legalMoveCount} legal`),
  ]);
}

export function renderAnalysisTools(opts: AnalysisToolsOpts): VNode {
  return h(
    'section.analysis-panel.analyse__tools',
    { attrs: { 'aria-label': 'Move probability analysis' } },
    [
      renderCeval(opts),
      h('div.analyse__moves.areplay', [opts.moves]),
      h(
        'section.explorer-box.sub-box',
        { class: { loading: opts.loading } },
        [
          h('div.overlay', { attrs: { 'aria-hidden': 'true' } }),
          opts.error && h('p.analysis-error', opts.error),
          opts.result
            ? h('div.data', [explorerTitle(opts.result), showMoveTable(opts, opts.result)])
            : h('div.data.empty', [
                explorerTitle(),
                h('div.message', [
                  opts.error
                    ? h('strong', 'Analysis unavailable')
                    : [h('icon.ddloader'), h('strong', opts.loadingText ?? 'Select a position')],
                ]),
              ]),
        ],
      ),
    ],
  );
}
