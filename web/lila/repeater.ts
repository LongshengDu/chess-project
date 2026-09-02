// Adapted from Lila's replay-button repeater behavior.
export function repeatOnHold(element: HTMLButtonElement, action: () => void): void {
  let timer: number | undefined;
  let repeated = false;

  element.addEventListener('click', () => {
    if (repeated) repeated = false;
    else action();
  });
  element.addEventListener('pointerdown', () => {
    if (element.disabled) return;
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
  element.addEventListener('pointerup', stop);
  element.addEventListener('pointercancel', stop);
  element.addEventListener('pointerleave', stop);
}
