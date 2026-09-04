// Small local tree model following Lila's root/node/path analysis structure.
import type {
  AnalysisGame,
  AnalysisPositionState,
  AnalysisVariationStep,
  PositionAnalysis,
} from '../../types';

export interface AnalysisTreeNode {
  id: string;
  ply: number;
  uci?: string;
  san?: string;
  imported: boolean;
  position: AnalysisPositionState;
  basePly: number;
  variationMoves: string[];
  parent?: AnalysisTreeNode;
  children: AnalysisTreeNode[];
  preferredChild?: string;
  analysis?: PositionAnalysis;
}

export class AnalysisTree {
  readonly root: AnalysisTreeNode;
  current: AnalysisTreeNode;

  private nextId = 0;
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
    return !this.current.parent;
  }

  get atEnd(): boolean {
    return !this.current.children.length;
  }

  get requestPath(): { ply: number; moves: string[] } {
    return { ply: this.current.basePly, moves: [...this.current.variationMoves] };
  }

  get continuationUcis(): string[] {
    return this.current.children.map(child => child.uci).filter((uci): uci is string => Boolean(uci));
  }

  node(id: string): AnalysisTreeNode | undefined {
    return this.nodes.get(id);
  }

  child(uci: string): AnalysisTreeNode | undefined {
    return this.current.children.find(node => node.uci === uci);
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

  private makeNode(
    data: Omit<AnalysisTreeNode, 'id' | 'ply' | 'children'>,
  ): AnalysisTreeNode {
    const node: AnalysisTreeNode = {
      ...data,
      id: `n${this.nextId++}`,
      ply: data.position.ply,
      children: [],
    };
    this.nodes.set(node.id, node);
    return node;
  }
}
