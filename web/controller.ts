import { Chessground } from '@lichess-org/chessground';
import type { Api as ChessgroundApi } from '@lichess-org/chessground/api';
import type { Key, Role } from '@lichess-org/chessground/types';

import captureUrl from '../deps/lichess-lila/public/sound/standard/Capture.mp3?url';
import checkUrl from '../deps/lichess-lila/public/sound/Silence.mp3?url';
import checkmateUrl from '../deps/lichess-lila/public/sound/standard/GenericNotify.mp3?url';
import moveUrl from '../deps/lichess-lila/public/sound/standard/Move.mp3?url';
import { analysisApi, roundApi } from './api';
import { AnalysisTree, type AnalysisTreeNode } from './lila/analyse/tree';
import stepwiseScroll from './lila/stepwiseScroll';
import type { PromotionChoice } from './lila/promotion';
import type {
  AnalysisGame,
  AnalysisPositionState,
  Color,
  MoveSound,
  PositionState,
  PositionAnalysis,
  PromotionRole,
  RoundPrefs,
  RoundState,
} from './types';

const promotionRoles: Record<PromotionRole, Role> = {
  q: 'queen',
  n: 'knight',
  r: 'rook',
  b: 'bishop',
};

export class LocalRoundController {
  state?: RoundState;
  viewIndex = 0;
  orientation: Color = 'white';
  promotion?: PromotionChoice;
  menuOpen = false;
  helpOpen = false;
  confirmingResign = false;
  busy?: string;
  notice?: string;
  analysis?: AnalysisGame;
  analysisTree?: AnalysisTree;
  analysisResult?: PositionAnalysis;
  analysisDialogOpen = false;
  analysisInput = '';
  analysisBusy?: string;
  analysisError?: string;

  readonly prefs: RoundPrefs = {
    blindfold: localStorage.getItem('maia-blindfold') === 'on',
    coordinates: localStorage.getItem('maia-coordinates') !== 'off',
    sound: localStorage.getItem('maia-sound') !== 'off',
  };

  private ground?: ChessgroundApi;
  private analysisRequest = 0;
  private readonly sounds: Record<MoveSound, HTMLAudioElement> = {
    move: new Audio(moveUrl),
    capture: new Audio(captureUrl),
    check: new Audio(checkUrl),
    checkmate: new Audio(checkmateUrl),
  };

  constructor(private readonly redraw: () => void) {
    document.body.classList.toggle('blindfold', this.prefs.blindfold);
  }

  get latestPly(): number {
    if (this.analysis) return this.analysis.positions.length - 1;
    return this.state ? this.state.history.length - 1 : 0;
  }

  get live(): boolean {
    return !this.analysis && this.viewIndex === this.latestPly;
  }

  get position(): PositionState | undefined {
    if (this.analysis) return this.analysisTree?.current.position;
    return this.state?.history[this.viewIndex];
  }

  get analysisMode(): boolean {
    return Boolean(this.analysis);
  }

  get interactionBusy(): boolean {
    return Boolean(this.busy || this.analysisBusy);
  }

  get modelName(): string {
    return this.state?.modelName ?? 'Maia3';
  }

  get modelElo(): number {
    return this.state?.modelElo ?? 1500;
  }

  get atStart(): boolean {
    return this.analysis ? Boolean(this.analysisTree?.atStart) : this.viewIndex === 0;
  }

  get atEnd(): boolean {
    return this.analysis ? Boolean(this.analysisTree?.atEnd) : this.viewIndex === this.latestPly;
  }

  get moves(): string[] {
    return this.analysis?.moves ?? this.state?.moves ?? [];
  }

  get status(): string {
    if (this.notice) return this.notice;
    if (this.analysisBusy) return this.analysisBusy;
    if (this.analysisError) return this.analysisError;
    if (this.analysis) {
      const node = this.analysisTree?.current;
      if (!node || node === this.analysisTree?.root) return 'Initial PGN position — play a move to analyse';
      if (node.position.gameOver) return node.position.status;
      return node.imported ? `Position after ${node.san}` : `Variation after ${node.san}`;
    }
    if (this.busy) return this.busy;
    if (!this.state) return 'Connecting...';
    if (this.live) return this.state.status;
    if (this.viewIndex === 0) return 'Starting position';
    const number = Math.ceil(this.viewIndex / 2);
    return `Viewing ${number}${this.viewIndex % 2 ? '. ' : '... '}${this.state.moves[this.viewIndex - 1]}`;
  }

  mountGround = (element: HTMLElement): void => {
    if (this.ground) return;
    this.ground = Chessground(element, {
      orientation: this.orientation,
      coordinates: this.prefs.coordinates,
      animation: { enabled: true, duration: 200 },
      highlight: { lastMove: true, check: true },
      movable: {
        free: false,
        color: undefined,
        showDests: true,
        events: { after: (origin, destination) => void this.userMove(origin, destination) },
      },
      premovable: { enabled: false },
      drawable: {
        enabled: true,
        visible: true,
        defaultSnapToValidMove: true,
        eraseOnMovablePieceClick: true,
      },
      disableContextMenu: true,
    });
    this.syncGround();
  };

  readonly onWheel = stepwiseScroll(
    event => (event.deltaY > 0 ? this.nextPosition() : this.previousPosition()),
    () => this.analysis ? Boolean(this.analysisBusy) : !this.state || (this.live && !this.state.gameOver),
  );

  async load(): Promise<void> {
    try {
      const state = await roundApi.state();
      this.commit(state);
      if (!state.gameOver && state.turn === 'black') await this.requestMaiaMove();
    } catch (error) {
      this.fail(error);
    }
  }

  navigate(ply: number): void {
    if (this.analysis) return;
    if (!this.state || this.busy) return;
    const previous = this.viewIndex;
    this.viewIndex = Math.max(0, Math.min(ply, this.latestPly));
    if (this.viewIndex === previous) return;
    this.notice = undefined;
    this.redrawAndSync();
    if (this.viewIndex > previous) this.playSounds(this.position?.moveSounds ?? []);
  }

  firstPosition = (): void => {
    if (this.analysisTree) this.selectAnalysisNode(this.analysisTree.first());
    else this.navigate(0);
  };

  previousPosition = (): void => {
    if (this.analysisTree) {
      const previous = this.analysisTree.previous();
      if (previous) this.selectAnalysisNode(previous);
    } else this.navigate(this.viewIndex - 1);
  };

  nextPosition = (): void => {
    if (this.analysisTree) {
      const next = this.analysisTree.next();
      if (next) this.selectAnalysisNode(next);
    } else this.navigate(this.viewIndex + 1);
  };

  lastPosition = (): void => {
    if (this.analysisTree) this.selectAnalysisNode(this.analysisTree.last());
    else this.navigate(this.latestPly);
  };

  jumpToAnalysisNode = (id: string): void => {
    const node = this.analysisTree?.node(id);
    if (node) this.selectAnalysisNode(node);
  };

  flip = (): void => {
    this.promotion = undefined;
    this.orientation = this.orientation === 'white' ? 'black' : 'white';
    this.menuOpen = false;
    this.redrawAndSync();
  };

  toggleMenu = (): void => {
    this.menuOpen = !this.menuOpen;
    this.redraw();
  };

  closeMenu = (): void => {
    if (!this.menuOpen) return;
    this.menuOpen = false;
    this.redraw();
  };

  toggleZen = (): void => {
    document.body.classList.toggle('zen');
    this.menuOpen = false;
    this.redraw();
    requestAnimationFrame(() => this.ground?.redrawAll());
  };

  setHelp(open: boolean): void {
    this.helpOpen = open;
    this.menuOpen = false;
    this.redraw();
  }

  setConfirmingResign(value: boolean): void {
    this.confirmingResign = value;
    this.redraw();
  }

  setPref(pref: keyof RoundPrefs, value: boolean): void {
    this.prefs[pref] = value;
    localStorage.setItem(`maia-${pref}`, value ? 'on' : 'off');
    if (pref === 'blindfold') document.body.classList.toggle('blindfold', value);
    if (pref === 'coordinates') {
      this.ground?.set({ coordinates: value });
      this.ground?.redrawAll();
    }
    if (pref === 'sound' && value) this.playSound('move');
    this.redraw();
  }

  async copyPgn(): Promise<void> {
    if (!this.state) return;
    try {
      await navigator.clipboard.writeText(this.state.pgn);
      this.notice = 'PGN copied';
      this.menuOpen = false;
      this.redraw();
      window.setTimeout(() => {
        this.notice = undefined;
        this.redraw();
      }, 1200);
    } catch (error) {
      this.fail(error);
    }
  }

  openAnalysisDialog = (): void => {
    this.analysisDialogOpen = true;
    this.analysisError = undefined;
    this.menuOpen = false;
    this.redraw();
  };

  closeAnalysisDialog = (): void => {
    if (this.analysisBusy) return;
    this.analysisDialogOpen = false;
    this.analysisError = undefined;
    this.redraw();
  };

  setAnalysisInput(value: string): void {
    this.analysisInput = value;
    this.redraw();
  }

  importPgn = async (): Promise<void> => {
    if (!this.analysisInput.trim() || this.analysisBusy) return;
    this.analysisBusy = 'Reading PGN...';
    this.analysisError = undefined;
    document.body.classList.add('thinking');
    this.redraw();
    try {
      this.analysis = await analysisApi.importPgn(this.analysisInput);
      this.analysisTree = new AnalysisTree(this.analysis);
      this.analysisResult = undefined;
      this.analysisDialogOpen = false;
      this.viewIndex = 0;
      this.analysisBusy = undefined;
      this.redrawAndSync();
      await this.refreshAnalysis();
    } catch (error) {
      this.analysisBusy = undefined;
      this.analysisError = error instanceof Error ? error.message : String(error);
      document.body.classList.remove('thinking');
      this.redraw();
    }
  };

  exitAnalysis = (): void => {
    this.analysisRequest += 1;
    this.analysis = undefined;
    this.analysisTree = undefined;
    this.analysisResult = undefined;
    this.analysisBusy = undefined;
    this.analysisError = undefined;
    this.viewIndex = this.state ? this.state.history.length - 1 : 0;
    document.body.classList.remove('thinking');
    this.redrawAndSync();
  };

  action(name: 'takeback' | 'claim-draw' | 'resign', message: string): void {
    void this.request(message, () => roundApi.action(name));
  }

  newGame = (): void => {
    this.menuOpen = false;
    void this.request('Starting new game...', roundApi.newGame);
  };

  finishPromotion = (role?: PromotionRole): void => {
    const choice = this.promotion;
    this.promotion = undefined;
    if (!choice || !role) {
      if (this.analysis) this.analysisBusy = undefined;
      else this.busy = undefined;
      this.redrawAndSync();
      return;
    }

    this.ground?.setPieces(
      new Map([
        [choice.destination, { color: 'white', role: promotionRoles[role], promoted: true }],
      ]),
    );
    const uci = choice.origin + choice.destination + role;
    if (this.analysis) {
      this.analysisBusy = undefined;
      void this.playAnalysisMove(uci, true);
    }
    else void this.submitMove(uci);
  };

  clearShapes(): void {
    this.ground?.setShapes([]);
  }

  setAnalysisHover = (uci?: string): void => {
    if (!uci) {
      this.ground?.setAutoShapes([]);
      return;
    }
    this.ground?.setAutoShapes([
      {
        orig: uci.slice(0, 2) as Key,
        dest: uci.slice(2, 4) as Key,
        brush: 'paleBlue',
      },
    ]);
  };

  handleKey(event: KeyboardEvent): boolean {
    if (event.key === 'Escape' && this.analysisDialogOpen) {
      this.closeAnalysisDialog();
      return true;
    }
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement)
      return false;
    const key = event.key;

    if (key === 'Escape') {
      if (this.promotion) this.finishPromotion();
      else if (this.helpOpen) this.setHelp(false);
      else if (this.confirmingResign) this.setConfirmingResign(false);
      else if (this.menuOpen) this.closeMenu();
      else this.clearShapes();
    } else if (key === 'ArrowLeft' || key.toLowerCase() === 'k') this.previousPosition();
    else if (key === 'ArrowRight' || key.toLowerCase() === 'j') this.nextPosition();
    else if (key === 'ArrowUp' || key === '0' || key === 'Home') this.firstPosition();
    else if (key === 'ArrowDown' || key === '$' || key === 'End') this.lastPosition();
    else if (key.toLowerCase() === 'f') this.flip();
    else if (key.toLowerCase() === 'z') this.toggleZen();
    else if (key.toLowerCase() === 'h') this.toggleMenu();
    else if (key === '?') this.setHelp(!this.helpOpen);
    else return false;
    return true;
  }

  private async userMove(origin: Key, destination: Key): Promise<void> {
    if (this.analysis) {
      const position = this.position as AnalysisPositionState | undefined;
      if (!position || position.gameOver || this.analysisBusy) return;
      const roles = position.promotions[origin + destination] as PromotionRole[] | undefined;
      if (roles) {
        this.promotion = { origin, destination, roles };
        this.analysisBusy = 'Choose a piece';
        this.ground?.set({ movable: { color: undefined, dests: new Map<Key, Key[]>() } });
        this.redraw();
        return;
      }
      await this.playAnalysisMove(origin + destination, true);
      return;
    }
    if (!this.state || !this.live || this.state.gameOver || this.busy) return;
    const roles = this.state.promotions[origin + destination] as PromotionRole[] | undefined;
    if (roles) {
      this.promotion = { origin, destination, roles };
      this.lock('Choose a piece', true);
      return;
    }
    await this.submitMove(origin + destination);
  }

  selectAnalysisMove = (uci: string): void => {
    void this.playAnalysisMove(uci);
  };

  private selectAnalysisNode(node: AnalysisTreeNode): void {
    const tree = this.analysisTree;
    if (!tree || this.analysisBusy || !tree.select(node)) return;
    this.analysisResult = node.analysis;
    this.analysisError = undefined;
    this.redrawAndSync();
    this.playSounds(node.position.moveSounds);
    void this.refreshAnalysis();
  }

  private async playAnalysisMove(uci: string, preserveBoard = false): Promise<void> {
    const tree = this.analysisTree;
    if (!tree || this.analysisBusy) return;

    const existing = tree.child(uci);
    if (existing) {
      this.selectAnalysisNode(existing);
      return;
    }

    const { ply, moves } = tree.requestPath;
    this.analysisBusy = 'Adding variation...';
    this.analysisError = undefined;
    document.body.classList.add('thinking');
    this.redraw();
    if (preserveBoard)
      this.ground?.set({ movable: { color: undefined, dests: new Map<Key, Key[]>() } });
    else this.syncGround();

    try {
      const step = await analysisApi.move(ply, moves, uci);
      if (!this.analysisTree || this.analysisTree !== tree) return;
      const node = tree.add(step);
      this.analysisResult = node.analysis;
      this.analysisBusy = undefined;
      document.body.classList.remove('thinking');
      this.redrawAndSync();
      this.playSounds(node.position.moveSounds);
      await this.refreshAnalysis();
    } catch (error) {
      this.analysisBusy = undefined;
      this.analysisError = error instanceof Error ? error.message : String(error);
      document.body.classList.remove('thinking');
      this.redrawAndSync();
    }
  }

  private async refreshAnalysis(): Promise<void> {
    const tree = this.analysisTree;
    if (!this.analysis || !tree) return;
    const request = ++this.analysisRequest;
    const node = tree.current;
    if (node.analysis) {
      this.analysisResult = node.analysis;
      this.analysisBusy = undefined;
      this.analysisError = undefined;
      document.body.classList.remove('thinking');
      this.redraw();
      return;
    }

    const path = tree.requestPath;
    this.analysisResult = undefined;
    this.analysisBusy = `${this.modelName} ${this.modelElo} + Stockfish are analysing...`;
    this.analysisError = undefined;
    document.body.classList.add('thinking');
    this.redraw();
    try {
      const result = await analysisApi.position(path.ply, path.moves);
      if (
        request !== this.analysisRequest ||
        !this.analysis ||
        this.analysisTree !== tree ||
        tree.current !== node
      ) {
        return;
      }
      node.analysis = result;
      this.analysisResult = result;
      this.analysisBusy = undefined;
      document.body.classList.remove('thinking');
      this.redraw();
    } catch (error) {
      if (request !== this.analysisRequest) return;
      this.analysisBusy = undefined;
      this.analysisError = error instanceof Error ? error.message : String(error);
      document.body.classList.remove('thinking');
      this.redraw();
    }
  }

  private async request(
    message: string,
    operation: () => Promise<RoundState>,
    preserveBoard = false,
  ): Promise<void> {
    this.lock(message, preserveBoard);
    try {
      this.commit(await operation());
    } catch (error) {
      await this.restore(error);
    }
  }

  // Lila plays the local move and the later server move as separate sound events.
  // These two local API phases preserve that timing without moving chess rules into TypeScript.
  private async submitMove(uci: string): Promise<void> {
    this.lock('Maia is thinking...', true);
    try {
      const state = await roundApi.move(uci);
      this.commit(state);
      if (!state.gameOver && state.turn === 'black') await this.requestMaiaMove();
    } catch (error) {
      await this.restore(error);
    }
  }

  private async requestMaiaMove(): Promise<void> {
    this.lock('Maia is thinking...');
    try {
      this.commit(await roundApi.reply());
    } catch (error) {
      await this.restore(error);
    }
  }

  private lock(message: string, preserveBoard = false): void {
    this.busy = message;
    this.notice = undefined;
    this.confirmingResign = false;
    document.body.classList.add('thinking');
    this.redraw();
    if (preserveBoard)
      this.ground?.set({ movable: { color: undefined, dests: new Map<Key, Key[]>() } });
    else this.syncGround();
  }

  private commit(state: RoundState): void {
    this.state = state;
    this.viewIndex = state.history.length - 1;
    this.busy = undefined;
    this.notice = undefined;
    this.promotion = undefined;
    this.confirmingResign = false;
    document.body.classList.remove('thinking');
    this.redrawAndSync();
    this.playSounds(state.sounds);
  }

  private async restore(error: unknown): Promise<void> {
    console.error(error);
    try {
      const state = await roundApi.state();
      this.commit(state);
      this.notice = error instanceof Error ? error.message : String(error);
      this.redraw();
    } catch (stateError) {
      console.error(stateError);
      this.fail(error);
    }
  }

  private fail(error: unknown): void {
    console.error(error);
    this.busy = undefined;
    this.notice = this.state
      ? error instanceof Error
        ? error.message
        : String(error)
      : 'Unable to reach Python';
    document.body.classList.remove('thinking');
    this.redraw();
  }

  private redrawAndSync(): void {
    this.redraw();
    this.syncGround();
  }

  private syncGround(): void {
    const state = this.state;
    const position = this.position;
    if (!this.ground || !position) return;
    this.ground.setAutoShapes([]);
    const analysisPosition = this.analysis ? position as AnalysisPositionState : undefined;
    const canAnalyseMove = Boolean(
      analysisPosition && !analysisPosition.gameOver && !this.analysisBusy,
    );
    const canPlayMove = Boolean(
      !this.analysis && state && this.live && !state.gameOver && !this.busy && state.turn === 'white',
    );
    const canMove = canAnalyseMove || canPlayMove;
    const destinations = analysisPosition?.dests ?? state?.dests ?? {};
    this.ground.set({
      fen: position.fen,
      orientation: this.orientation,
      turnColor: position.turn,
      check: position.check,
      lastMove: position.lastMove ?? undefined,
      coordinates: this.prefs.coordinates,
      movable: {
        color: canAnalyseMove ? position.turn : canPlayMove ? 'white' : undefined,
        dests: canMove
          ? new Map(Object.entries(destinations) as [Key, Key[]][])
          : new Map<Key, Key[]>(),
      },
      drawable: { enabled: true, visible: true, defaultSnapToValidMove: true },
    });
  }

  private playSound(name: MoveSound | null): void {
    if (!name || !this.prefs.sound) return;
    const sound = this.sounds[name];
    sound.currentTime = 0;
    void sound.play().catch(() => undefined);
  }

  private playSounds(names: MoveSound[]): void {
    names.forEach((name, index) =>
      window.setTimeout(() => this.playSound(name), index * 100),
    );
  }
}
