// Maia-specific metadata around Lila's canonical standard-chess tree nodes.
// Node completion and compact two-character IDs come directly from the pinned
// Lila source; this class only adapts the Flask analysis path and metadata.
import { completeNode } from '../../../deps/lichess-lila/ui/lib/src/tree/node';
import type { TreeNode } from '../../../deps/lichess-lila/ui/lib/src/tree/types';

import type {
  AnalysisGame,
  AnalysisPositionState,
  AnalysisVariationStep,
  PositionAnalysis,
} from '../../types';

export type AnalysisTreeNode = Omit<TreeNode, 'children'> & {
  children: AnalysisTreeNode[];
  imported: boolean;
  position: AnalysisPositionState;
  basePly: number;
  variationMoves: string[];
  path: string;
  parent?: AnalysisTreeNode;
  preferredChild?: string;
  analysis?: PositionAnalysis;
};

const completeStandardNode = completeNode('standard');

export class AnalysisTree {
  readonly root: AnalysisTreeNode;
  current: AnalysisTreeNode;

  private readonly nodes = new Map<string, AnalysisTreeNode>();

  constructor(game: AnalysisGame) {
    this.root = this.makeNode({
      imported: true,
      position: game.positions[0],
      basePly: 0,
      variationMoves: [],
    });
    this.current = this.root;

    let parent = this.root;
    game.moves.forEach((san, index) => {
      const child = this.makeNode({
        uci: game.ucis[index],
        san,
        imported: true,
        position: game.positions[index + 1],
        basePly: index + 1,
        variationMoves: [],
        parent,
      });
      parent.children.push(child);
      parent = child;
    });
  }

  get atStart(): boolean {
    return this.current === this.root;
  }

  get atEnd(): boolean {
    return !this.current.children.length;
  }

  get requestPath(): { ply: number; moves: string[] } {
    return { ply: this.current.basePly, moves: [...this.current.variationMoves] };
  }

  get continuationUcis(): string[] {
    return this.current.children.map(child => child.uci).filter((uci): uci is Uci => Boolean(uci));
  }

  node(path: string): AnalysisTreeNode | undefined {
    return this.nodes.get(path);
  }

  child(uci: string): AnalysisTreeNode | undefined {
    const id = this.nodeId(uci);
    return this.current.children.find(child => child.id === id);
  }

  select(node: AnalysisTreeNode): boolean {
    if (node === this.current) return false;
    if (node.parent) node.parent.preferredChild = node.id;
    this.current = node;
    return true;
  }

  first(): AnalysisTreeNode {
    return this.root;
  }

  previous(): AnalysisTreeNode | undefined {
    return this.current.parent;
  }

  next(): AnalysisTreeNode | undefined {
    return this.preferred(this.current);
  }

  last(): AnalysisTreeNode {
    let node = this.current;
    while (node.children.length) node = this.preferred(node)!;
    return node;
  }

  add(step: AnalysisVariationStep): AnalysisTreeNode {
    const existing = this.child(step.uci);
    if (existing) {
      this.select(existing);
      return existing;
    }

    const parent = this.current;
    const child = this.makeNode({
      uci: step.uci,
      san: step.san,
      imported: false,
      position: step.position,
      basePly: parent.basePly,
      variationMoves: [...parent.variationMoves, step.uci],
      parent,
    });
    parent.children.push(child);
    this.select(child);
    return child;
  }

  private preferred(node: AnalysisTreeNode): AnalysisTreeNode | undefined {
    return node.children.find(child => child.id === node.preferredChild) ?? node.children[0];
  }

  private nodeId(uci: string): string {
    return completeStandardNode({
      ply: 0,
      uci: uci as Uci,
      fen: this.current.position.fen as FEN,
      children: [],
    }).id;
  }

  private makeNode(data: {
    uci?: string;
    san?: string;
    imported: boolean;
    position: AnalysisPositionState;
    basePly: number;
    variationMoves: string[];
    parent?: AnalysisTreeNode;
  }): AnalysisTreeNode {
    const node = completeStandardNode({
      ply: data.position.ply,
      uci: data.uci as Uci | undefined,
      san: data.san,
      fen: data.position.fen as FEN,
      children: [],
    }) as AnalysisTreeNode;
    Object.assign(node, data);
    node.path = data.parent ? data.parent.path + node.id : '';
    this.nodes.set(node.path, node);
    return node;
  }
}
