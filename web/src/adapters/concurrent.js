// Drain running tasks on failure/cancel; never start more than the given limit.
export async function forEachConcurrent(items, limit, handle, {isCancelled = () => false, onError = () => {}} = {}) {
  if (!Number.isInteger(limit) || limit < 1) throw new Error('Invalid analysis concurrency');
  let next = 0, failed = false;
  const worker = async () => {
    while (!failed && !isCancelled() && next < items.length) {
      const index = next++;
      try { await handle(items[index], index); }
      catch (error) { failed = true; onError(error); throw error; }
    }
  };
  const results = await Promise.allSettled(Array.from({length:Math.min(limit,items.length)}, worker));
  const failure = results.find(result => result.status === 'rejected');
  if (failure) throw failure.reason;
}
