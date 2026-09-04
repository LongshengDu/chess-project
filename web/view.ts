import type { Role } from '@lichess-org/chessground/types';
import { opposite } from '@lichess-org/chessground/util';
import type { VNode } from 'snabbdom';

import type { LocalRoundController } from './controller';
import { gameSetup } from './lila/gameSetup';
import { renderAnalysisTools } from './lila/analyse/moveTable';
import { renderAnalysisTree } from './lila/analyse/treeView';
import { promotionView } from './lila/promotion';
import { repeatOnHold } from './lila/repeater';
import { h, onInsert } from './lila/snabbdom';
import type { Color, Material, MaterialSide, RoundPrefs } from './types';

function click(action: () => void) {
  return onInsert<HTMLElement>(element => element.addEventListener('click', action));
}

function button(
  selector: string,
  label: string,
  title: string,
  disabled: boolean,
  action: () => void,
): VNode {
  return h(
    `button${selector}`,
    {
      attrs: { type: 'button', title, 'aria-label': title },
      props: { disabled },
      hook: click(action),
    },
    label,
  );
}

function repeatButton(
  selector: string,
  label: string,
  title: string,
  disabled: boolean,
  action: () => void,
): VNode {
  return h(
    `button${selector}`,
    {
      attrs: { type: 'button', title, 'aria-label': title },
      props: { disabled },
      hook: onInsert<HTMLButtonElement>(element => repeatOnHold(element, action)),
    },
    label,
  );
}

function player(color: Color, ctrl: LocalRoundController): VNode {
  const human = !ctrl.analysis && color === ctrl.humanColor;
  const bottom = color === ctrl.orientation;
  const analysisName = color === 'white' ? ctrl.analysis?.white : ctrl.analysis?.black;
  const name = analysisName ?? (human ? 'You' : 'Maia');
  const active = ctrl.analysis
    ? ctrl.position?.turn === color
    : ctrl.live &&
      !ctrl.state?.gameOver &&
      (ctrl.busy ? color !== ctrl.humanColor : ctrl.state?.turn === color);
  return h(
    `div.player.ruser.${color}`,
    {
      key: `player-${color}`,
      class: {
        human,
        opponent: !human,
        'ruser-bottom': bottom,
        'ruser-top': !bottom,
      },
    },
    [
      h(`span.avatar.${color}`, { attrs: { 'aria-hidden': 'true' } }, color === 'white' ? '♙' : '♟'),
      h('div.player-info', [
        h('div', [
          h(`i${!ctrl.analysis && !human ? '.online' : ''}`),
          h('strong', name),
          !ctrl.analysis && !human && h('small', String(ctrl.modelElo)),
        ]),
        h('span', ctrl.analysis || human ? color[0].toUpperCase() + color.slice(1) : ctrl.modelName),
      ]),
      h(
        `div.clock.rclock.rclock-${bottom ? 'bottom' : 'top'}`,
        {
          class: { active },
          attrs: { 'aria-label': `${name}, ${ctrl.analysis ? 'side to move indicator' : 'untimed game'}` },
        },
        '∞',
      ),
    ],
  );
}

function groupMaterial(side: MaterialSide, color: Color): VNode[] {
  const groups: VNode[] = [];
  let current: Role | undefined;
  let pieces: VNode[] = [];

  const flush = () => {
    if (pieces.length) groups.push(h('span.material-group', pieces));
    pieces = [];
  };

  side.pieces.forEach(role => {
    if (current !== role) {
      flush();
      current = role;
    }
    pieces.push(h(`piece.${color}.${role}`));
  });
  flush();
  if (side.score > 0) groups.push(h('strong', `+${side.score}`));
  return groups;
}

function material(color: Color, data: Material, position: 'top' | 'bottom'): VNode {
  const side = data[color];
  const description = side.pieces.length
    ? `${color} material advantage: ${side.pieces.join(', ')}${side.score ? `, plus ${side.score}` : ''}`
    : `${color} has no material advantage`;
  return h(
    `div#${color}-material.material.cg-wrap`,
    {
      key: `material-${color}`,
      class: { [`material-${position}`]: true },
      attrs: { 'aria-label': description },
    },
    groupMaterial(side, color),
  );
}

function moveList(ctrl: LocalRoundController): VNode {
  const state = ctrl.state;
  const moves = ctrl.moves;
  const children: Array<VNode | false> = [];
  if (!moves.length) {
    children.push(
      h('div.start-message', [
        h('span', { attrs: { 'aria-hidden': 'true' } }, 'ⓘ'),
        h('p', ctrl.analysis ? 'This PGN has no main-line moves.' : [
          `You play the ${state?.humanColor ?? 'white'} pieces.`,
          h('br'),
          h('strong', state?.turn === state?.humanColor ? 'It is your turn.' : 'Maia is thinking...'),
        ]),
      ]),
    );
  } else {
    moves.forEach((move, index) => {
      if (index % 2 === 0) children.push(h('qZM.move-number', String(index / 2 + 1)));
      const ply = ctrl.analysisMode ? index : index + 1;
      children.push(
        h(
          'Z7yx.move',
          {
            class: { active: ctrl.viewIndex === ply },
            attrs: {
              role: 'button',
              tabindex: '0',
              title: ctrl.analysisMode ? `Analyse position before ${move}` : `View position after ${move}`,
              'aria-current': ctrl.viewIndex === ply ? 'step' : 'false',
            },
            hook: onInsert<HTMLElement>(element => {
              element.addEventListener('click', () => ctrl.navigate(ply));
              element.addEventListener('keydown', event => {
                if (event.key === 'Enter' || event.key === ' ') ctrl.navigate(ply);
              });
            }),
          },
          move,
        ),
      );
    });
    const result = ctrl.analysis?.result ?? state?.result;
    if (result && result !== '*')
      children.push(
        h('div.result-wrap', [
          h('strong', result.replace('1/2-1/2', '½-½')),
          !ctrl.analysis && state && h('em', state.status),
        ]),
      );
  }

  const scrollActive = (element: Element) =>
    requestAnimationFrame(() =>
      element.querySelector<HTMLElement>('.move.active')?.scrollIntoView({ block: 'nearest', inline: 'nearest' }),
    );

  return h(
    'aPp#moves.move-list',
    {
      attrs: { 'aria-label': 'Move history', 'aria-live': 'polite' },
      hook: {
        insert: vnode => scrollActive(vnode.elm as Element),
        postpatch: (_, vnode) => scrollActive(vnode.elm as Element),
      },
    },
    children,
  );
}

function replayControls(ctrl: LocalRoundController, analysis = false): VNode {
  const disabled = Boolean(ctrl.interactionBusy || (!ctrl.state && !ctrl.analysis));
  return h(
    `bo3.move-controls${analysis ? '.analyse__controls.analyse-controls' : ''}`,
    { attrs: { 'aria-label': 'Move navigation' } },
    [
      h('span.noop', { attrs: { 'aria-hidden': 'true' } }),
      button('#first-move', '⇤', 'Starting position (Up, 0, or Home)', disabled || ctrl.atStart, ctrl.firstPosition),
      repeatButton('#previous-move', '◀', 'Previous move (Left or K)', disabled || ctrl.atStart, ctrl.previousPosition),
      repeatButton('#next-move', '▶', 'Next move (Right or J)', disabled || ctrl.atEnd, ctrl.nextPosition),
      button('#last-move.return-live', '⇥', analysis ? 'End of current variation (Down, $, or End)' : 'Latest position (Down, $, or End)', disabled || ctrl.atEnd, ctrl.lastPosition),
      button('#show-menu', '☰', 'Board menu (H)', disabled, ctrl.toggleMenu),
    ],
  );
}

function replay(ctrl: LocalRoundController): VNode {
  return h('i5d.replay', [moveList(ctrl), replayControls(ctrl)]);
}

function menuToggle(
  id: string,
  icon: string,
  label: string,
  pref: keyof RoundPrefs,
  ctrl: LocalRoundController,
): VNode {
  return h('label', [
    h('span', { attrs: { 'aria-hidden': 'true' } }, icon),
    label,
    h('input', {
      attrs: { id, type: 'checkbox' },
      props: { checked: ctrl.prefs[pref] },
      hook: onInsert<HTMLInputElement>(element =>
        element.addEventListener('change', () => ctrl.setPref(pref, element.checked)),
      ),
    }),
  ]);
}

function boardMenu(ctrl: LocalRoundController): VNode | false {
  return (
    ctrl.menuOpen &&
    h('section#board-menu.board-menu', { attrs: { 'aria-label': 'Board menu' } }, [
      h('button#flip-board', { attrs: { type: 'button' }, hook: click(ctrl.flip) }, [
        h('span', { attrs: { 'aria-hidden': 'true' } }, '↕'),
        'Flip board',
        h('kbd', 'F'),
      ]),
      h('button#zen-mode', { attrs: { type: 'button' }, hook: click(ctrl.toggleZen) }, [
        h('span', { attrs: { 'aria-hidden': 'true' } }, '▣'),
        'Zen mode',
        h('kbd', 'Z'),
      ]),
      menuToggle('blindfold', '◉', 'Blindfold', 'blindfold', ctrl),
      menuToggle('coordinates', '▦', 'Coordinates', 'coordinates', ctrl),
      menuToggle('sound', '🔊', 'Sound', 'sound', ctrl),
      h('button#copy-pgn', { attrs: { type: 'button' }, hook: click(() => void ctrl.copyPgn()) }, [
        h('span', { attrs: { 'aria-hidden': 'true' } }, '📋'),
        'Copy PGN',
      ]),
      h('button#analyse-pgn-menu', { attrs: { type: 'button' }, hook: click(ctrl.openAnalysisDialog) }, [
        h('span', { attrs: { 'aria-hidden': 'true' } }, '↗'),
        'Analyse PGN',
      ]),
      h('button#show-help', { attrs: { type: 'button' }, hook: click(() => ctrl.setHelp(true)) }, [
        h('span', { attrs: { 'aria-hidden': 'true' } }, '?'),
        'Keyboard & drawing help',
      ]),
    ])
  );
}

function actions(ctrl: LocalRoundController): VNode | false {
  const state = ctrl.state;
  if (ctrl.analysis || !state || state.gameOver) return false;
  if (ctrl.confirmingResign)
    return h('div.action-confirm', [
      button('#cancel-action', '×', 'Cancel', false, () => ctrl.setConfirmingResign(false)),
      h('p', 'Resign the game?'),
      button('#confirm-action', '✓', 'Confirm resign', false, () => ctrl.action('resign', 'Resigning...')),
    ]);
  return h('div#game-actions.game-actions', { attrs: { 'aria-label': 'Game actions' } }, [
    button('#takeback', '↩', 'Take back the last move pair', Boolean(ctrl.busy || !state.canTakeback), () => ctrl.action('takeback', 'Taking back moves...')),
    button('#claim-draw', '½', state.canClaimDraw ? 'Claim a draw' : 'No draw can be claimed', Boolean(ctrl.busy || !state.canClaimDraw), () => ctrl.action('claim-draw', 'Claiming draw...')),
    button('#resign', '⚑', 'Resign', Boolean(ctrl.busy || !state.canResign), () => ctrl.setConfirmingResign(true)),
  ]);
}

function analysisTools(ctrl: LocalRoundController): VNode | false {
  const tree = ctrl.analysisTree;
  if (!ctrl.analysis || !tree) return false;
  return renderAnalysisTools({
    modelName: ctrl.modelName,
    modelElo: ctrl.modelElo,
    result: ctrl.analysisResult,
    moves: renderAnalysisTree(tree, ctrl.jumpToAnalysisNode, ctrl.analysis.result),
    loading: Boolean(ctrl.analysisBusy),
    loadingText: ctrl.analysisBusy,
    error: ctrl.analysisError,
    onHover: ctrl.setAnalysisHover,
    onSelect: ctrl.selectAnalysisMove,
    continuationUcis: tree.continuationUcis,
  });
}

function pgnDialog(ctrl: LocalRoundController): VNode | false {
  if (!ctrl.analysisDialogOpen) return false;
  return h('div.modal-backdrop', { hook: click(ctrl.closeAnalysisDialog) }, [
    h(
      'section.pgn-dialog',
      {
        attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'pgn-dialog-title' },
        hook: onInsert(element => element.addEventListener('click', event => event.stopPropagation())),
      },
      [
        h('header', [
          h('div', [h('h2#pgn-dialog-title', 'Analyse a PGN'), h('p', 'Paste one game in Portable Game Notation.')]),
          button('#close-pgn-dialog', '×', 'Close', Boolean(ctrl.analysisBusy), ctrl.closeAnalysisDialog),
        ]),
        h('textarea#pgn-input', {
          attrs: { placeholder: '[Event "My game"]\n\n1. e4 e5 2. Nf3 ...', 'aria-label': 'PGN text' },
          props: { value: ctrl.analysisInput, disabled: Boolean(ctrl.analysisBusy) },
          hook: onInsert<HTMLTextAreaElement>(element => {
            element.focus();
            element.addEventListener('input', () => ctrl.setAnalysisInput(element.value));
            element.addEventListener('keydown', event => {
              if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') void ctrl.importPgn();
            });
          }),
        }),
        ctrl.analysisError && h('p.pgn-error', ctrl.analysisError),
        h('footer', [
          h('small', `${ctrl.modelName} probabilities use ${ctrl.modelElo}/${ctrl.modelElo} Elo. Click a candidate or play on the board to add a variation.`),
          button('#import-pgn', ctrl.analysisBusy ? 'ANALYSING…' : 'ANALYSE', 'Analyse pasted PGN', Boolean(ctrl.analysisBusy || !ctrl.analysisInput.trim()), () => void ctrl.importPgn()),
        ]),
      ],
    ),
  ]);
}

function gameSetupDialog(ctrl: LocalRoundController): VNode | false {
  if (!ctrl.gameSetupOpen) return false;
  return gameSetup({
    color: ctrl.gameSetupColor,
    fen: ctrl.gameSetupFen,
    busy: Boolean(ctrl.busy),
    error: ctrl.gameSetupError,
    onColor: ctrl.setGameSetupColor,
    onFen: ctrl.setGameSetupFen,
    onCancel: ctrl.closeGameSetup,
    onSubmit: ctrl.startGame,
  });
}

function help(ctrl: LocalRoundController): VNode | false {
  const rows = [
    ['Left / K', 'Previous move'],
    ['Right / J', 'Next move'],
    ['Up / 0 / Home', 'Starting position'],
    ['Down / $ / End', 'Latest position'],
    ['Mouse wheel', 'Replay moves over the board'],
    ['Analysis board', 'Move either color to add a variation'],
    ['F', 'Flip board'],
    ['Z', 'Zen mode'],
    ['H', 'Board menu'],
    ['Right-drag', 'Green arrow; right-click for a circle'],
    ['Shift + right-drag', 'Red annotation'],
    ['Alt + right-drag', 'Blue annotation'],
    ['Shift + Alt + right-drag', 'Yellow annotation'],
    ['Escape', 'Clear annotations or close a dialog'],
  ];
  return (
    ctrl.helpOpen &&
    h('div.modal-backdrop', { hook: click(() => ctrl.setHelp(false)) }, [
      h(
        'section.keyboard-help',
        {
          attrs: { role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'help-title' },
          hook: onInsert(element => element.addEventListener('click', event => event.stopPropagation())),
        },
        [
          h('header', [
            h('h2#help-title', 'Board controls'),
            button('#close-help', '×', 'Close', false, () => ctrl.setHelp(false)),
          ]),
          h('dl', rows.map(([term, description]) => h('div', [h('dt', term), h('dd', description)]))),
        ],
      ),
    ])
  );
}

export function roundView(ctrl: LocalRoundController): VNode {
  const position = ctrl.position;
  // Matches Lila round/view/main.ts: player color is the default bottom color,
  // and flipping swaps the top and bottom colors as one unit.
  const bottomColor = ctrl.orientation;
  const topColor = opposite(bottomColor);
  const materialData = position?.material ?? {
    white: { pieces: [], score: 0 },
    black: { pieces: [], score: 0 },
  };

  const title = ctrl.analysis?.title ?? 'Maia Chess';
  const subtitle = ctrl.analysis?.subtitle ?? 'Standard • Casual • Untimed';
  return h('main.game-shell.round', { class: { analysis: ctrl.analysisMode } }, [
    h('section.board-side.round__app', {
      attrs: { 'aria-label': 'Maia game' },
    }, [
      player(topColor, ctrl),
      material(topColor, materialData, 'top'),
      h(
        'div.board-frame.round__app__board.main-board',
        {
          key: 'board-frame',
          hook: onInsert(element => element.addEventListener('wheel', ctrl.onWheel, { passive: false })),
        },
        [
          h('div#board.cg-wrap', {
            key: 'chessground',
            attrs: { 'aria-label': 'Chess board' },
            hook: onInsert(ctrl.mountGround),
          }),
          ctrl.promotion && promotionView(ctrl.promotion, ctrl.orientation, ctrl.finishPromotion),
        ],
      ),
      material(bottomColor, materialData, 'bottom'),
      player(bottomColor, ctrl),
    ]),
    h('aside.game-panel.round__app__table', { attrs: { 'aria-label': 'Game notation and controls' } }, [
      h('header', [
        h('span', { attrs: { 'aria-hidden': 'true' } }, ctrl.analysis ? '⌁' : '♞'),
        h('div', [h('h1', title), h('p', subtitle)]),
      ]),
      ctrl.analysis ? analysisTools(ctrl) : replay(ctrl),
      ctrl.analysis && replayControls(ctrl, true),
      boardMenu(ctrl),
      actions(ctrl),
      h('footer', [
        h('p#status', { attrs: { 'aria-live': 'polite' } }, ctrl.status),
        h('div.footer-actions', ctrl.analysis ? [
          button('#replace-pgn', 'PASTE PGN', 'Paste another PGN', Boolean(ctrl.analysisBusy), ctrl.openAnalysisDialog),
          button('#exit-analysis', 'BACK TO GAME', 'Exit PGN analysis', Boolean(ctrl.analysisBusy), ctrl.exitAnalysis),
        ] : [
          button('#analyse-pgn', 'ANALYSE PGN', 'Paste a PGN for analysis', Boolean(ctrl.busy), ctrl.openAnalysisDialog),
          button('#new-game', 'NEW GAME', 'Set up a new game', Boolean(ctrl.busy), ctrl.openGameSetup),
        ]),
      ]),
    ]),
    help(ctrl),
    pgnDialog(ctrl),
    gameSetupDialog(ctrl),
  ]);
}
