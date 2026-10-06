// A full-game job belongs to Python; the browser only streams its progress.
export class GameAnalysisRun {
  constructor(gameId, targetDepth, profilerId) {
    this.gameId = gameId;
    this.runId = crypto.randomUUID();
    this.payload = {run_id:this.runId, target_depth:targetDepth,
      ...(profilerId ? {profiler_id:profilerId} : {})};
    this.controller = new AbortController();
    this.started = false;
    this.finished = false;
  }

  cancel() {
    if (this.controller.signal.aborted || this.finished) return;
    if (this.started) fetch(`/api/platform/games/${encodeURIComponent(this.gameId)}/analyze/cancel`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify({run_id:this.runId}), keepalive:true,
    }).catch(() => {});
    this.controller.abort();
  }

  async *events() {
    let reader;
    this.started = true;
    try {
      const response = await fetch(`/api/platform/games/${encodeURIComponent(this.gameId)}/analyze`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body:JSON.stringify(this.payload), signal:this.controller.signal,
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.error || `Game analysis failed (${response.status})`);
      }
      if (!response.body) throw new Error('Game analysis returned no progress stream.');
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (true) {
        const {done, value} = await reader.read();
        buffer += decoder.decode(value, {stream:!done});
        const lines = buffer.split('\n');
        buffer = lines.pop();
        if (done && buffer.trim()) lines.push(buffer);
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line);
          if (event.type === 'error') throw new Error(event.message || 'Game analysis failed.');
          if (event.type === 'complete' || event.type === 'cancelled') this.finished = true;
          yield event;
          if (this.finished) return;
        }
        if (done) throw new Error('Game analysis ended before completion. Please retry.');
      }
    } finally {
      if (!this.finished) this.cancel();
      await reader?.cancel().catch(() => {});
    }
  }
}
