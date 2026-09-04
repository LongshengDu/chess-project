// Focused adaptation of Lila's setup color cards and from-position FEN input.
import type { VNode } from 'snabbdom';

import type { Color } from '../types';
import { h, onInsert } from './snabbdom';

export interface GameSetupOpts {
  color: Color;
  fen: string;
  busy: boolean;
  error?: string;
  onColor: (color: Color) => void;
  onFen: (fen: string) => void;
  onCancel: () => void;
  onSubmit: () => void;
}

const colors: Array<{ key: Color; name: string }> = [
  { key: 'black', name: 'Black' },
  { key: 'white', name: 'White' },
];

const click = (action: () => void) =>
  onInsert<HTMLElement>(element => element.addEventListener('click', action));

function colorCards(opts: GameSetupOpts): VNode {
  return h(
    'group.radio.color-picker.color-cards',
    { attrs: { role: 'radiogroup', 'aria-label': 'Play as' } },
    colors.map(({ key, name }) =>
      h('div', { key }, [
        h(`input#color-picker-${key}`, {
          attrs: { name: 'color', type: 'radio', value: key },
          props: { checked: opts.color === key, disabled: opts.busy },
          hook: onInsert<HTMLInputElement>(element =>
            element.addEventListener('change', () => {
              if (element.checked) opts.onColor(key);
            }),
          ),
        }),
        h('label.card-radio', { attrs: { for: `color-picker-${key}` } }, [
          h(`div.color-picker__button.${key}`, h('icon')),
          h('span.text', name),
        ]),
      ]),
    ),
  );
}

export function gameSetup(opts: GameSetupOpts): VNode {
  return h('div.modal-backdrop', { hook: click(opts.onCancel) }, [
    h(
      'section.game-setup-dialog',
      {
        attrs: {
          role: 'dialog',
          'aria-modal': 'true',
          'aria-labelledby': 'game-setup-title',
        },
        hook: onInsert(element =>
          element.addEventListener('click', event => event.stopPropagation()),
        ),
      },
      [
        h('header', [
          h('div', [
            h('h2#game-setup-title', 'Game setup'),
            h('p', 'Play Maia from the standard board or a FEN position.'),
          ]),
          h(
            'button.close',
            {
              attrs: { type: 'button', title: 'Close', 'aria-label': 'Close game setup' },
              props: { disabled: opts.busy },
              hook: click(opts.onCancel),
            },
            '×',
          ),
        ]),
        h(
          'form',
          {
            hook: onInsert<HTMLFormElement>(element =>
              element.addEventListener('submit', event => {
                event.preventDefault();
                opts.onSubmit();
              }),
            ),
          },
          [
            h('div.setup-content', [
              h('div.config-group', [
                h('label.label', { attrs: { for: 'fen-input' } }, 'Position'),
                h('input#fen-input', {
                  attrs: {
                    type: 'text',
                    placeholder: 'Paste a FEN string',
                    'aria-invalid': opts.error ? 'true' : 'false',
                  },
                  props: { value: opts.fen, disabled: opts.busy },
                  hook: onInsert<HTMLInputElement>(element =>
                    element.addEventListener('input', () => opts.onFen(element.value)),
                  ),
                }),
                h('small', 'Leave blank to use the standard starting position.'),
              ]),
              h('div.config-group', [
                h('div.label', 'Play as'),
                colorCards(opts),
              ]),
              opts.error && h('p.setup-error', opts.error),
            ]),
            h('footer', [
              h(
                'button.cancel',
                {
                  attrs: { type: 'button' },
                  props: { disabled: opts.busy },
                  hook: click(opts.onCancel),
                },
                'CANCEL',
              ),
              h(
                'button.start',
                {
                  attrs: { type: 'submit' },
                  props: { disabled: opts.busy },
                },
                opts.busy ? 'STARTING…' : 'START GAME',
              ),
            ]),
          ],
        ),
      ],
    ),
  ]);
}
