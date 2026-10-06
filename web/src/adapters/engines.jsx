import React, { createContext, useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'react-hot-toast';
import { cpToWinrate } from '@maia/lib/analysis';
import { request } from './api';
import { searchDepth, searchSatisfied } from './search-policy';
import { terminalEvaluation } from './analysis-data';

export const MaiaEngineContext = createContext(null);
export const StockfishEngineContext = createContext(null);

export function stockfishEvaluation(raw, fen) {
  if ('terminal_cp' in raw) return terminalEvaluation(raw);
  const direction = fen.split(' ')[1] === 'w' ? 1 : -1;
  const entries = Object.entries(raw.cp_vec).sort((a, b) => direction * (b[1] - a[1]));
  const [model_move, model_optimal_cp] = raw.best_move && raw.best_move in raw.cp_vec
    ? [raw.best_move, raw.cp_vec[raw.best_move]] : entries[0];
  const winrate_vec = Object.fromEntries(entries.map(([move, cp]) => [move, cpToWinrate(cp * direction, false)]));
  const bestWinrate = winrate_vec[model_move];
  return {...raw, sent:true, model_move, model_optimal_cp, is_checkmate:false,
    cp_relative_vec:Object.fromEntries(entries.map(([move, cp]) => [move, direction * (cp - model_optimal_cp)])),
    winrate_vec, winrate_loss_vec:Object.fromEntries(entries.map(([move]) => [move, winrate_vec[move] - bestWinrate]))};
}

export function EngineProvider({ children }) {
  const [searchConfig, setSearchConfig] = useState(null);
  useEffect(() => {
    request('/config').then(setSearchConfig).catch(error => toast.error(`Engine settings: ${error.message}`));
  }, []);
  const profiling = useRef(new URLSearchParams(window.location.search).get('profiler') === '1').current;
  const activeSearch = useRef(new Set());
  const activeBatch = useRef(null);
  const maia = useMemo(() => ({status:'ready', progress:1, profiling, downloadModel:async () => {},
    maia:{batchEvaluateMaia3: async (fens, ratings, opponents, history = {}) => {
      try { return await request('/maia', {fens, ratings, opponents, ...history}); }
      catch (error) { toast.error(`Maia: ${error.message}`, {id:'maia-error'}); throw error; }
    }},
  }), []);
  const stockfish = useMemo(() => {
    const stop = () => { for (const search of activeSearch.current) search.stop(); };
    return {
    status:searchConfig ? 'ready' : 'loading', error:null, isReady:() => !!searchConfig, profiling, searchConfig,
    satisfies:(evaluation, depth, policy, played) => searchSatisfied(evaluation, searchDepth(depth, searchConfig), searchConfig, policy, played),
    stopEvaluation:stop,
    beginBatch:owner => { stop(); activeBatch.current = owner; },
    endBatch:owner => { if (activeBatch.current === owner) activeBatch.current = null; },
    isBatchRunning:() => !!activeBatch.current,
    async *streamEvaluations(fen, moveCount, depth = searchConfig?.default_depth, options = {}) {
      depth = searchDepth(depth, searchConfig);
      if (activeBatch.current && !options.bulk) return;
      if (!options.bulk) stop();
      const controller = new AbortController();
      const search_id = crypto.randomUUID();
      let finished = false;
      const search = {stop:() => {
        if (controller.signal.aborted) return;
        if (!finished) fetch('/api/platform/stockfish/cancel', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({search_id}), keepalive:true,
        }).catch(() => {});
        controller.abort();
      }};
      activeSearch.current.add(search);
      let reader;
      try {
        const response = await fetch('/api/platform/stockfish', {
          method:'POST', headers:{'Content-Type':'application/json'},
          body:JSON.stringify({fen, depth, seconds:searchConfig?.budgets[depth], options, search_id}), signal:controller.signal,
        });
        if (!response.ok) throw new Error((await response.json()).error || 'Evaluation failed');
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
            const raw = JSON.parse(line);
            if (raw.error) throw new Error(raw.error);
            if (raw.complete) finished = true;
            if ('terminal_cp' in raw || Object.keys(raw.cp_vec).length) yield stockfishEvaluation(raw, fen);
          }
          if (done) break;
        }
      } catch (error) {
        if (error.name !== 'AbortError') {
          toast.error(`Stockfish: ${error.message}`, {id:'stockfish-error'});
          throw error;
        }
      } finally {
        search.stop();
        await reader?.cancel().catch(() => {});
        activeSearch.current.delete(search);
      }
    },
  }; }, [searchConfig]);
  return <MaiaEngineContext.Provider value={maia}><StockfishEngineContext.Provider value={stockfish}>
    {children}
  </StockfishEngineContext.Provider></MaiaEngineContext.Provider>;
}
