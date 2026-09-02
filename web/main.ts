import { Chessground } from '@lichess-org/chessground';
import type { Key, Role } from '@lichess-org/chessground/types';
import '@lichess-org/chessground/assets/chessground.base.css';
import '@lichess-org/chessground/assets/chessground.brown.css';
import '@lichess-org/chessground/assets/chessground.cburnett.css';
import captureUrl from '../deps/lichess-lila/public/sound/standard/Capture.mp3?url';
import checkUrl from '../deps/lichess-lila/public/sound/standard/Check.mp3?url';
import checkmateUrl from '../deps/lichess-lila/public/sound/standard/Checkmate.mp3?url';
import moveUrl from '../deps/lichess-lila/public/sound/standard/Move.mp3?url';
import './style.css';

type Color = 'white' | 'black';
type MoveSound = 'move' | 'capture' | 'check' | 'checkmate';
type MaterialSide = { pieces: Role[]; score: number };
type Material = Record<Color, MaterialSide>;

type PositionState = {
  fen: string;
  turn: Color;
  lastMove: Key[] | null;
  check: boolean;
  material: Material;
  sound: MoveSound | null;
};

type UiState = PositionState & {
  gameOver: boolean;
  result: string | null;
  status: string;
  moves: string[];
  history: PositionState[];
  dests: Record<string, Key[]>;
  promotions: Record<string, string[]>;
  canTakeback: boolean;
  canClaimDraw: boolean;
  canResign: boolean;
  sounds: MoveSound[];
  pgn: string;
};

async function requestState(path: string, body?: object): Promise<UiState> {
  const response = await fetch(path, body && {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data: UiState | { error: string } = await response.json().catch(() => ({
    error: response.statusText,
  }));
  if (!response.ok) throw new Error('error' in data ? data.error : response.statusText);
  return data as UiState;
}

const api = {
  state: () => requestState('/api/state'),
  move: (uci: string) => requestState('/api/move', { uci }),
  action: (name: string) => requestState('/api/action', { name }),
  newGame: () => requestState('/api/new-game', {}),
};

const $ = <T extends Element>(selector: string) => document.querySelector<T>(selector)!;
const status = $<HTMLParagraphElement>('#status');
const moveList = $<HTMLDivElement>('#moves');
const newGame = $<HTMLButtonElement>('#new-game');
const boardSide = $<HTMLElement>('.board-side');
const boardFrame = $<HTMLElement>('.board-frame');
const promotion = $<HTMLDivElement>('#promotion');
const promotionButtons = [...promotion.querySelectorAll<HTMLButtonElement>('button')];
const help = $<HTMLDialogElement>('#keyboard-help');
const firstButton = $<HTMLButtonElement>('#first-move');
const previousButton = $<HTMLButtonElement>('#previous-move');
const nextButton = $<HTMLButtonElement>('#next-move');
const lastButton = $<HTMLButtonElement>('#last-move');
const showMenu = $<HTMLButtonElement>('#show-menu');
const boardMenu = $<HTMLElement>('#board-menu');
const flipButton = $<HTMLButtonElement>('#flip-board');
const showHelp = $<HTMLButtonElement>('#show-help');
const zenButton = $<HTMLButtonElement>('#zen-mode');
const blindfold = $<HTMLInputElement>('#blindfold');
const coordinates = $<HTMLInputElement>('#coordinates');
const soundToggle = $<HTMLInputElement>('#sound');
const copyPgn = $<HTMLButtonElement>('#copy-pgn');
const takeback = $<HTMLButtonElement>('#takeback');
const claimDraw = $<HTMLButtonElement>('#claim-draw');
const resign = $<HTMLButtonElement>('#resign');
const gameActions = $<HTMLDivElement>('#game-actions');
const actionConfirm = $<HTMLDivElement>('#action-confirm');
const whiteClock = $<HTMLDivElement>('#white-clock');
const blackClock = $<HTMLDivElement>('#black-clock');
const materialElements: Record<Color, HTMLElement> = {
  white: $<HTMLElement>('#white-material'),
  black: $<HTMLElement>('#black-material'),
};
const navigationButtons = [firstButton, previousButton, nextButton, lastButton, showMenu];
const promotionRoles: Record<string, Role> = {
  q: 'queen',
  r: 'rook',
  b: 'bishop',
  n: 'knight',
};

let game: UiState | undefined;
let viewIndex = 0;
let orientation: Color = 'white';
let promotions: Record<string, string[]> = {};
let resolvePromotion: ((role: string) => void) | undefined;
const sounds: Record<MoveSound, HTMLAudioElement> = {
  move: new Audio(moveUrl),
  capture: new Audio(captureUrl),
  check: new Audio(checkUrl),
  checkmate: new Audio(checkmateUrl),
};

soundToggle.checked = localStorage.getItem('maia-sound') !== 'off';
coordinates.checked = localStorage.getItem('maia-coordinates') !== 'off';
blindfold.checked = localStorage.getItem('maia-blindfold') === 'on';
document.body.classList.toggle('blindfold', blindfold.checked);

function playSound(name: MoveSound | null): void {
  if (!name || !soundToggle.checked) return;
  const sound = sounds[name];
  sound.currentTime = 0;
  void sound.play().catch(() => undefined);
}

function setActiveClock(color?: Color): void {
  whiteClock.classList.toggle('active', color === 'white');
  blackClock.classList.toggle('active', color === 'black');
}

function lock(message: string): void {
  status.textContent = message;
  newGame.disabled = true;
  navigationButtons.forEach(button => button.disabled = true);
  takeback.disabled = claimDraw.disabled = resign.disabled = true;
  document.body.classList.add('thinking');
  setActiveClock('black');
  ground.set({ movable: { color: undefined, dests: new Map() } });
}

function renderMaterial(material: Material): void {
  (['white', 'black'] as const).forEach(color => {
    const side = material[color];
    const groups: HTMLElement[] = [];
    let group: HTMLElement | undefined;
    let previous: Role | undefined;
    side.pieces.forEach(role => {
      if (role !== previous) {
        group = document.createElement('span');
        group.className = 'material-group';
        groups.push(group);
        previous = role;
      }
      const piece = document.createElement('piece');
      piece.className = `${color} ${role}`;
      group!.append(piece);
    });
    if (side.score > 0) {
      const score = document.createElement('strong');
      score.textContent = `+${side.score}`;
      groups.push(score);
    }
    materialElements[color].replaceChildren(...groups);
    materialElements[color].setAttribute(
      'aria-label',
      side.pieces.length
        ? `${color} material advantage: ${side.pieces.join(', ')}${side.score ? `, plus ${side.score}` : ''}`
        : `${color} has no material advantage`,
    );
  });
}

function updateNavigation(): void {
  const latest = game ? game.history.length - 1 : 0;
  firstButton.disabled = viewIndex === 0;
  previousButton.disabled = viewIndex === 0;
  nextButton.disabled = viewIndex === latest;
  lastButton.disabled = viewIndex === latest;
  lastButton.classList.toggle('return-live', viewIndex !== latest);
  showMenu.disabled = false;

  moveList.querySelectorAll<HTMLButtonElement>('button.move').forEach(button => {
    const active = Number(button.dataset.ply) === viewIndex;
    button.classList.toggle('active', active);
    button.setAttribute('aria-current', active ? 'step' : 'false');
  });

  const active = moveList.querySelector<HTMLElement>('.move.active');
  if (active) active.scrollIntoView({ block: 'nearest', inline: 'nearest' });
}

function showPosition(index: number): void {
  if (!game) return;
  const latest = game.history.length - 1;
  viewIndex = Math.max(0, Math.min(index, latest));
  const position = game.history[viewIndex];
  const live = viewIndex === latest;

  ground.set({
    fen: position.fen,
    turnColor: position.turn,
    check: position.check,
    lastMove: position.lastMove ?? undefined,
    coordinates: coordinates.checked,
    movable: {
      color: live && !game.gameOver && game.turn === 'white' ? 'white' : undefined,
      dests: live ? new Map(Object.entries(game.dests) as [Key, Key[]][]) : new Map(),
    },
    drawable: { enabled: true, visible: true, defaultSnapToValidMove: true },
  });
  renderMaterial(position.material);
  setActiveClock(live && !game.gameOver ? game.turn : undefined);

  if (live) status.textContent = game.status;
  else if (viewIndex === 0) status.textContent = 'Starting position';
  else {
    const number = Math.ceil(viewIndex / 2);
    const dots = viewIndex % 2 ? '. ' : '... ';
    status.textContent = `Viewing ${number}${dots}${game.moves[viewIndex - 1]}`;
  }
  updateNavigation();
}

function navigate(index: number): void {
  const before = viewIndex;
  showPosition(index);
  if (game && viewIndex > before) playSound(game.history[viewIndex].sound);
}

function renderMoves(): void {
  if (!game?.moves.length) {
    const message = document.createElement('div');
    message.className = 'start-message';
    message.innerHTML = '<span aria-hidden="true">ⓘ</span><p>You play the white pieces.<br><strong>It is your turn.</strong></p>';
    moveList.replaceChildren(message);
    return;
  }

  const fragment = document.createDocumentFragment();
  game.moves.forEach((move, index) => {
    if (index % 2 === 0) {
      const number = document.createElement('span');
      number.className = 'move-number';
      number.textContent = String(index / 2 + 1);
      fragment.append(number);
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'move';
    button.dataset.ply = String(index + 1);
    button.textContent = move;
    button.title = `View position after ${move}`;
    button.onclick = () => navigate(index + 1);
    fragment.append(button);
  });
  if (game.result) {
    const wrap = document.createElement('div');
    wrap.className = 'result-wrap';
    const result = document.createElement('strong');
    result.textContent = game.result.replace('1/2-1/2', '½-½');
    const reason = document.createElement('em');
    reason.textContent = game.status;
    wrap.append(result, reason);
    fragment.append(wrap);
  }
  moveList.replaceChildren(fragment);
}

function updateActions(): void {
  if (!game) return;
  takeback.disabled = !game.canTakeback;
  claimDraw.disabled = !game.canClaimDraw;
  resign.disabled = !game.canResign;
  claimDraw.title = game.canClaimDraw ? 'Claim a draw' : 'No draw can be claimed';
  gameActions.hidden = game.gameOver;
  actionConfirm.hidden = true;
}

function render(state: UiState): void {
  game = state;
  promotions = state.promotions;
  viewIndex = state.history.length - 1;
  newGame.disabled = false;
  document.body.classList.remove('thinking');
  renderMoves();
  showPosition(viewIndex);
  updateActions();
  state.sounds.forEach((sound, index) => setTimeout(() => playSound(sound), index * 100));
}

function finishPromotion(role: string): void {
  promotion.hidden = true;
  const resolve = resolvePromotion;
  resolvePromotion = undefined;
  resolve?.(role);
}

function choosePromotion(options: string[], destination: Key): Promise<string> {
  const file = destination.charCodeAt(0) - 97;
  const left = (orientation === 'white' ? file : 7 - file) * 12.5;
  const order = ['q', 'n', 'r', 'b'];

  promotionButtons.forEach(button => {
    const index = order.indexOf(button.value);
    button.hidden = !options.includes(button.value);
    button.style.left = `${left}%`;
    button.style.top = `${(orientation === 'white' ? index : 7 - index) * 12.5}%`;
  });
  promotion.hidden = false;
  return new Promise(resolve => resolvePromotion = resolve);
}

async function restoreAfter(error: unknown): Promise<void> {
  console.error(error);
  try {
    render(await api.state());
    status.textContent = error instanceof Error ? error.message : String(error);
  } catch (stateError) {
    console.error(stateError);
    status.textContent = 'Unable to reach Python';
  }
}

async function onMove(origin: Key, destination: Key): Promise<void> {
  const base = origin + destination;
  const options = promotions[base];
  lock(options ? 'Choose a piece' : 'Maia is thinking...');

  try {
    const suffix = options ? await choosePromotion(options, destination) : '';
    if (options && !suffix) {
      newGame.disabled = false;
      document.body.classList.remove('thinking');
      showPosition(game!.history.length - 1);
      updateActions();
      return;
    }
    if (suffix) {
      ground.setPieces(
        new Map([[destination, { color: 'white', role: promotionRoles[suffix], promoted: true }]]),
      );
      status.textContent = 'Maia is thinking...';
    }
    render(await api.move(base + suffix));
  } catch (error) {
    await restoreAfter(error);
  }
}

function flipBoard(): void {
  if (!promotion.hidden) finishPromotion('');
  ground.toggleOrientation();
  orientation = orientation === 'white' ? 'black' : 'white';
  boardSide.classList.toggle('flipped', orientation === 'black');
}

function closeMenu(): void {
  boardMenu.hidden = true;
  showMenu.setAttribute('aria-expanded', 'false');
}

function toggleMenu(): void {
  boardMenu.hidden = !boardMenu.hidden;
  showMenu.setAttribute('aria-expanded', String(!boardMenu.hidden));
}

function toggleZen(): void {
  document.body.classList.toggle('zen');
  closeMenu();
  requestAnimationFrame(ground.redrawAll);
}

async function doAction(name: string, message: string): Promise<void> {
  lock(message);
  try {
    render(await api.action(name));
  } catch (error) {
    await restoreAfter(error);
  }
}

function repeatOnHold(button: HTMLButtonElement, action: () => void): void {
  let timer: number | undefined;
  let repeated = false;
  button.onclick = () => {
    if (repeated) repeated = false;
    else action();
  };
  button.addEventListener('pointerdown', () => {
    if (button.disabled) return;
    timer = window.setTimeout(() => {
      repeated = true;
      action();
      timer = window.setInterval(action, 110);
    }, 350);
  });
  const stop = () => {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
  };
  button.addEventListener('pointerup', stop);
  button.addEventListener('pointercancel', stop);
  button.addEventListener('pointerleave', stop);
}

const ground = Chessground($<HTMLElement>('#board'), {
  orientation,
  coordinates: coordinates.checked,
  animation: { enabled: true, duration: 200 },
  highlight: { lastMove: true, check: true },
  movable: { free: false, color: undefined, showDests: true, events: { after: onMove } },
  premovable: { enabled: false },
  drawable: {
    enabled: true,
    visible: true,
    defaultSnapToValidMove: true,
    eraseOnMovablePieceClick: true,
  },
});

promotion.addEventListener('click', event => {
  if (event.target === promotion) finishPromotion('');
});
promotion.addEventListener('contextmenu', event => event.preventDefault());
promotionButtons.forEach(button => button.onclick = event => {
  event.stopPropagation();
  finishPromotion(button.value);
});

firstButton.onclick = () => navigate(0);
repeatOnHold(previousButton, () => navigate(viewIndex - 1));
repeatOnHold(nextButton, () => navigate(viewIndex + 1));
lastButton.onclick = () => navigate(game ? game.history.length - 1 : 0);
showMenu.onclick = toggleMenu;
flipButton.onclick = () => {
  flipBoard();
  closeMenu();
};
zenButton.onclick = toggleZen;
showHelp.onclick = () => {
  closeMenu();
  help.showModal();
};
$<HTMLButtonElement>('#close-help').onclick = () => help.close();

blindfold.onchange = () => {
  document.body.classList.toggle('blindfold', blindfold.checked);
  localStorage.setItem('maia-blindfold', blindfold.checked ? 'on' : 'off');
};
coordinates.onchange = () => {
  localStorage.setItem('maia-coordinates', coordinates.checked ? 'on' : 'off');
  ground.set({ coordinates: coordinates.checked });
  ground.redrawAll();
};
soundToggle.onchange = () => {
  localStorage.setItem('maia-sound', soundToggle.checked ? 'on' : 'off');
  if (soundToggle.checked) playSound('move');
};
copyPgn.onclick = async () => {
  if (!game) return;
  try {
    await navigator.clipboard.writeText(game.pgn);
    status.textContent = 'PGN copied';
    closeMenu();
    setTimeout(() => showPosition(viewIndex), 1200);
  } catch (error) {
    await restoreAfter(error);
  }
};

takeback.onclick = () => void doAction('takeback', 'Taking back moves...');
claimDraw.onclick = () => void doAction('claim-draw', 'Claiming draw...');
resign.onclick = () => {
  gameActions.hidden = true;
  actionConfirm.hidden = false;
};
$<HTMLButtonElement>('#cancel-action').onclick = () => {
  actionConfirm.hidden = true;
  gameActions.hidden = Boolean(game?.gameOver);
};
$<HTMLButtonElement>('#confirm-action').onclick = () => void doAction('resign', 'Resigning...');

let lastWheel = 0;
boardFrame.addEventListener('wheel', event => {
  if (!game || (viewIndex === game.history.length - 1 && !game.gameOver)) return;
  const now = performance.now();
  if (now - lastWheel < 90) return;
  lastWheel = now;
  event.preventDefault();
  navigate(viewIndex + (event.deltaY > 0 ? 1 : -1));
}, { passive: false });

document.addEventListener('pointerdown', event => {
  if (!boardMenu.hidden && !boardMenu.contains(event.target as Node) && event.target !== showMenu)
    closeMenu();
});

document.addEventListener('keydown', event => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
  const key = event.key;
  let handled = true;

  if (!promotion.hidden && key === 'Escape') finishPromotion('');
  else if (help.open && key === 'Escape') help.close();
  else if (!actionConfirm.hidden && key === 'Escape') {
    actionConfirm.hidden = true;
    gameActions.hidden = Boolean(game?.gameOver);
  }
  else if (!boardMenu.hidden && key === 'Escape') closeMenu();
  else if (key === 'ArrowLeft' || key.toLowerCase() === 'k') navigate(viewIndex - 1);
  else if (key === 'ArrowRight' || key.toLowerCase() === 'j') navigate(viewIndex + 1);
  else if (key === 'ArrowUp' || key === '0' || key === 'Home') navigate(0);
  else if (key === 'ArrowDown' || key === '$' || key === 'End')
    navigate(game ? game.history.length - 1 : 0);
  else if (key.toLowerCase() === 'f') flipBoard();
  else if (key.toLowerCase() === 'z') toggleZen();
  else if (key.toLowerCase() === 'h') toggleMenu();
  else if (key === '?') help.open ? help.close() : help.showModal();
  else if (key === 'Escape') ground.setShapes([]);
  else handled = false;

  if (handled) event.preventDefault();
});

newGame.addEventListener('click', async () => {
  lock('Starting new game...');
  closeMenu();
  try {
    render(await api.newGame());
  } catch (error) {
    await restoreAfter(error);
  }
});

api.state().then(render).catch(restoreAfter);
