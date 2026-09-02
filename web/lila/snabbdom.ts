// Small source-compatible subset of lila/ui/lib/src/view/snabbdom.ts.
import {
  h as snabbdomH,
  type Hooks,
  type VNode,
  type VNodeChildElement,
  type VNodeData,
} from 'snabbdom';

export type MaybeVNode = VNode | string | null | undefined;
export type LooseVNode = VNodeChildElement | boolean;
export type LooseVNodes = LooseVNode | LooseVNodes[];

export function onInsert<A extends HTMLElement>(effect: (element: A) => void): Hooks {
  return { insert: vnode => effect(vnode.elm as A) };
}

export function bind<K extends keyof GlobalEventHandlersEventMap>(
  eventName: K,
  handler: (event: GlobalEventHandlersEventMap[K]) => unknown,
  redraw?: () => void,
  passive = true,
): Hooks {
  return onInsert(element =>
    element.addEventListener(
      eventName,
      event => {
        const result = handler(event);
        if (result === false && !passive) event.preventDefault();
        redraw?.();
      },
      { passive },
    ),
  );
}

const validChild = (child: VNodeData | LooseVNodes): boolean =>
  (Boolean(child) && child !== true) || child === '' || child === 0;

function flatten(children: LooseVNodes, output: LooseVNode[]): void {
  if (Array.isArray(children)) children.forEach(child => flatten(child, output));
  else output.push(children);
}

function filterChildren(children: LooseVNodes): VNodeChildElement[] {
  const flattened: LooseVNode[] = [];
  flatten(children, flattened);
  return flattened.filter(validChild) as VNodeChildElement[];
}

export function h(selector: string, dataOrChildren?: VNodeData | LooseVNodes, children?: LooseVNodes): VNode {
  if (children !== undefined)
    return snabbdomH(selector, dataOrChildren as VNodeData, filterChildren(children));
  if (!validChild(dataOrChildren)) return snabbdomH(selector);
  if (
    Array.isArray(dataOrChildren) ||
    (typeof dataOrChildren === 'object' && 'sel' in dataOrChildren!)
  )
    return snabbdomH(selector, filterChildren(dataOrChildren as LooseVNodes));
  return snabbdomH(selector, dataOrChildren as VNodeData);
}
