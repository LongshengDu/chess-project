import { attributesModule, classModule, init, propsModule, type VNode } from 'snabbdom';

import '@lichess-org/chessground/assets/chessground.base.css';
import '@lichess-org/chessground/assets/chessground.brown.css';
import '@lichess-org/chessground/assets/chessground.cburnett.css';
import './style.scss';

import { LocalRoundController } from './controller';
import { roundView } from './view';

// Lila's round app uses the same controller -> view -> patch loop.
const patch = init([classModule, attributesModule, propsModule]);
const root = document.querySelector<HTMLElement>('#app');
if (!root) throw new Error('Missing #app mount point');

let vnode: VNode | HTMLElement = root;
let controller: LocalRoundController;
const redraw = () => {
  vnode = patch(vnode, roundView(controller));
};

controller = new LocalRoundController(redraw);
redraw();

document.addEventListener('pointerdown', event => {
  if (controller.menuOpen && !(event.target as Element).closest?.('#board-menu, #show-menu'))
    controller.closeMenu();
});

document.addEventListener('keydown', event => {
  if (controller.handleKey(event)) event.preventDefault();
});

void controller.load();
