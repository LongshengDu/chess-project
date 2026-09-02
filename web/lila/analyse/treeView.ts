// Focused adaptation of Lila's treeView/columnView.ts and lib tree SCSS DOM.
import type { VNode } from 'snabbdom';

import { h, onInsert, type LooseVNodes } from '../snabbdom';
import type { AnalysisTree, AnalysisTreeNode } from './tree';

function moveNumber(node: AnalysisTreeNode): number {
  return Math.floor((node.ply + 1) / 2);
}

function renderIndex(node: AnalysisTreeNode, withDots = false): VNode {
  const black = node.ply % 2 === 0;
  return h('index', `${moveNumber(node)}${withDots ? (black ? '...' : '.') : ''}`);
}

function renderMove(node: AnalysisTreeNode, tree: AnalysisTree, withIndex = false): VNode {
  return h(
    'move',
    {
      key: node.id,
      class: { active: node === tree.current, nongame: !node.imported },
      attrs: {
        'data-node': node.id,
        p: node.id,
        role: 'button',
        tabindex: '0',
        title: node.imported ? `Jump to ${node.san}` : `Jump to variation ${node.san}`,
        'aria-current': node === tree.current ? 'step' : 'false',
      },
    },
    [withIndex && renderIndex(node, true), h('san', node.san ?? '')],
  );
}

function renderVariation(node: AnalysisTreeNode, tree: AnalysisTree, first = true): LooseVNodes {
  const [continuation, ...alternatives] = node.children;
  return [
    renderMove(node, tree, first || node.ply % 2 === 1),
    alternatives.length > 0 &&
      h(
        'lines',
        alternatives.map(alternative =>
          h('line', [h('branch'), renderVariation(alternative, tree)]),
        ),
      ),
    continuation && renderVariation(continuation, tree, false),
  ];
}

function renderMainline(parent: AnalysisTreeNode, tree: AnalysisTree): LooseVNodes {
  const [child, ...variations] = parent.children;
  if (!child) return undefined;
  const whiteMove = child.ply % 2 === 1;
  return [
    whiteMove && renderIndex(child),
    renderMove(child, tree),
    variations.length > 0 && [
      whiteMove && h('move.empty', '...'),
      h(
        'interrupt',
        h(
          'lines',
          variations.map(variation =>
            h('line', [h('branch'), renderVariation(variation, tree)]),
          ),
        ),
      ),
      whiteMove && child.children.length > 0 && [renderIndex(child), h('move.empty', '...')],
    ],
    renderMainline(child, tree),
  ];
}

function treeHooks(onSelect: (id: string) => void) {
  const selectFrom = (target: EventTarget | null): boolean => {
    const move = target instanceof Element ? target.closest<HTMLElement>('move[data-node]') : null;
    if (!move?.dataset.node) return false;
    onSelect(move.dataset.node);
    return true;
  };
  const scrollActive = (element: Element) =>
    requestAnimationFrame(() =>
      element.querySelector<HTMLElement>('move.active')?.scrollIntoView({ block: 'nearest' }),
    );

  return {
    ...onInsert<HTMLElement>(element => {
      element.addEventListener('click', event => selectFrom(event.target));
      element.addEventListener('keydown', event => {
        if ((event.key === 'Enter' || event.key === ' ') && selectFrom(event.target))
          event.preventDefault();
      });
      scrollActive(element);
    }),
    postpatch: (_: VNode, vnode: VNode) => scrollActive(vnode.elm as Element),
  };
}

export function renderAnalysisTree(
  tree: AnalysisTree,
  onSelect: (id: string) => void,
  result: string,
): VNode {
  const blackStarts = tree.root.position.turn === 'black';
  return h('div.analysis-tree', { hook: treeHooks(onSelect) }, [
    h('div.tview2.tview2-column', [
      blackStarts && [renderIndex(tree.root), h('move.empty', '...')],
      renderMainline(tree.root, tree),
    ]),
    result !== '*' && h('div.result-wrap', h('strong', result.replace('1/2-1/2', '½-½'))),
  ]);
}
