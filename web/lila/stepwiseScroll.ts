// Adapted from lila/ui/lib/src/view/stepwiseScroll.ts.
export default function stepwiseScroll(
  action: (event: WheelEvent) => void,
  shouldSkip: (event: WheelEvent) => boolean,
): (event: WheelEvent) => void {
  let accumulatedDelta = 0;
  const isMac = /Mac|iPhone|iPad/.test(navigator.platform);

  return event => {
    if (event.ctrlKey || shouldSkip(event)) return;
    event.preventDefault();
    if (event.deltaMode === WheelEvent.DOM_DELTA_PIXEL) {
      accumulatedDelta += event.deltaY;
      if (isMac && Math.abs(accumulatedDelta) < 10) return;
    }
    accumulatedDelta = 0;
    action(event);
  };
}
