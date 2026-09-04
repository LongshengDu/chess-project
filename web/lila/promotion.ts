// Local version of lila/ui/lib/src/game/promotion.ts's geometry and role order.
import type { Key } from '@lichess-org/chessground/types';
import type { VNode } from 'snabbdom';

import type { Color, PromotionRole } from '../types';
import { h, onInsert } from './snabbdom';

export type PromotionChoice = {
  origin: Key;
  destination: Key;
  roles: PromotionRole[];
};

const roleNames: Record<PromotionRole, string> = {
  q: 'queen',
  n: 'knight',
  r: 'rook',
  b: 'bishop',
};

const roleOrder: PromotionRole[] = ['q', 'n', 'r', 'b'];

export function promotionView(
  choice: PromotionChoice,
  orientation: Color,
  finish: (role?: PromotionRole) => void,
): VNode {
  const file = choice.destination.charCodeAt(0) - 97;
  const left = (orientation === 'white' ? file : 7 - file) * 12.5;
  const color: Color = choice.destination[1] === '8' ? 'white' : 'black';

  return h(
    'div#promotion-choice.promotion.cg-wrap.' + (orientation === 'white' ? 'top' : 'bottom'),
    {
      hook: onInsert(element => {
        element.addEventListener('click', () => finish());
        element.oncontextmenu = () => false;
      }),
    },
    roleOrder.map((role, index) =>
      choice.roles.includes(role) &&
      h(
        'square',
        {
          attrs: {
            role: 'button',
            tabindex: '0',
            'aria-label': `Promote to ${roleNames[role]}`,
            style: `top:${(orientation === 'white' ? index : 7 - index) * 12.5}%;left:${left}%`,
          },
          hook: onInsert(element => {
            element.addEventListener('click', event => {
              event.stopPropagation();
              finish(role);
            });
            element.addEventListener('keydown', event => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.stopPropagation();
                finish(role);
              }
            });
          }),
        },
        [h(`piece.${color}.${roleNames[role]}`)],
      ),
    ),
  );
}
