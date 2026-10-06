import { useCallback, useContext, useEffect, useRef, useState } from 'react';
import { toast } from 'react-hot-toast';
import { MaiaEngineContext, StockfishEngineContext, stockfishEvaluation } from './engines';
import { request, storeCustomGame } from './api';
import { GameAnalysisRun } from './game-analysis';
import { applyGamePosition } from './analysis-data';

const initialProgress = {currentMoveIndex:0, totalMoves:0, currentMove:'',
  isAnalyzing:false, isComplete:false, isCancelled:false};

function finishProfiler(run, cancelled = false) {
  if (!run.profilerId || run.profilerFinished) return Promise.resolve();
  if (!run.profilerFinish) run.profilerFinish = request('/profiler/finish', {
    run_id:run.profilerId,
    ...(cancelled ? {cancelled:true} : {browser_run_ms:performance.now()-run.started, save_ms:0}),
  }).then(() => { run.profilerFinished = true; }).finally(() => { run.profilerFinish = null; });
  return run.profilerFinish;
}

export function useDeepAnalysis({gameTree, setCurrentNode, setAnalysisState, currentMaiaModel, gameId}) {
  const maia = useContext(MaiaEngineContext);
  const stockfish = useContext(StockfishEngineContext);
  const runRef = useRef(null);
  const savedGame = useRef({tree:gameTree, id:gameId});
  const [config, setConfig] = useState({targetDepth:stockfish.searchConfig?.default_depth ?? 18});
  const [progress, setProgress] = useState(initialProgress);
  const [fullAnalysis, setFullAnalysis] = useState(gameTree.fullAnalysis || null);

  const stop = useCallback(() => {
    const run = runRef.current;
    if (!run) return;
    run.cancelled = true;
    run.client?.cancel();
    finishProfiler(run, true).catch(() => {});
    stockfish.endBatch(run);
    runRef.current = null;
  }, [stockfish]);

  useEffect(() => {
    savedGame.current = {tree:gameTree, id:gameId};
    setFullAnalysis(gameTree.fullAnalysis || null);
    setProgress(initialProgress);
    return stop;
  }, [gameTree, gameId, stop]);

  const startAnalysis = useCallback(async targetDepth => {
    stop();
    const run = {cancelled:false, client:null, started:performance.now()};
    runRef.current = run;
    stockfish.beginBatch(run);
    const nodes = gameTree.getMainLine();
    setConfig({targetDepth});
    setProgress({...initialProgress, totalMoves:nodes.length, isAnalyzing:true});
    try {
      // Normal analysis pages already have a saved id. Preserve a locally
      // constructed tree too, including its starting FEN and player headers.
      let id = gameId || (savedGame.current.tree === gameTree && savedGame.current.id);
      if (!id) {
        const saved = await storeCustomGame({pgn:gameTree.toPGN()});
        id = saved.game_id;
        gameTree.savedGameId = id;
        savedGame.current = {tree:gameTree, id};
      }
      if (run.cancelled) return;
      if (stockfish.profiling) {
        ({run_id:run.profilerId} = await request('/profiler/start', {game_id:id,
          target_depth:targetDepth, selected_model:currentMaiaModel, total_positions:nodes.length}));
      }
      if (run.cancelled) {
        await finishProfiler(run, true);
        return;
      }
      run.client = new GameAnalysisRun(id, targetDepth, run.profilerId);
      const completed = new Set();
      for await (const event of run.client.events()) {
        if (run.cancelled) break;
        if (event.type === 'progress') {
          setProgress(previous => ({...previous, currentMove:event.message}));
        } else if (event.type === 'position') {
          const node = applyGamePosition(nodes, event.index, event.position, currentMaiaModel, stockfishEvaluation);
          completed.add(event.index);
          setAnalysisState(value => value + 1);
          setCurrentNode(node);
          setProgress(previous => ({...previous, currentMoveIndex:completed.size,
            currentMove:node.san || node.move || `Position ${event.index+1}`}));
        } else if (event.type === 'cancelled') {
          run.cancelled = true;
          setProgress(previous => ({...previous, isAnalyzing:false, isComplete:false, isCancelled:true}));
        } else if (event.type === 'complete') {
          if (!Array.isArray(event.analysis?.positions) || event.analysis.positions.length !== nodes.length) {
            throw new Error('The server returned incomplete game analysis. Please retry.');
          }
          // The terminal event also covers server-cache hits without position
          // events. Keep the complete common result, including fitted ratings.
          event.analysis.positions.forEach((position, index) =>
            applyGamePosition(nodes, index, position, currentMaiaModel, stockfishEvaluation));
          gameTree.fullAnalysis = event.analysis;
          setFullAnalysis(event.analysis);
          setAnalysisState(value => value + 1);
          await finishProfiler(run);
          if (!run.cancelled) setProgress(previous => ({...previous, currentMoveIndex:nodes.length,
            currentMove:'Analysis saved', isAnalyzing:false, isComplete:true}));
        }
      }
    } catch (error) {
      await finishProfiler(run, true).catch(() => {});
      if (!run.cancelled) {
        toast.error(`Game analysis: ${error.message}`, {id:'game-analysis-error'});
        setProgress(previous => ({...previous, isAnalyzing:false, isComplete:false}));
      }
    } finally {
      stockfish.endBatch(run);
      if (runRef.current === run) runRef.current = null;
    }
  }, [gameTree, gameId, stockfish, currentMaiaModel, setAnalysisState, setCurrentNode, stop]);

  const cancelAnalysis = useCallback(() => {
    stop();
    setProgress(previous => ({...previous, isAnalyzing:false, isCancelled:true, isComplete:false}));
  }, [stop]);
  const resetProgress = useCallback(() => { stop(); setProgress(initialProgress); }, [stop]);
  return {progress, config, setConfig, fullAnalysis, startAnalysis, cancelAnalysis, resetProgress,
    isEnginesReady:stockfish.isReady() && maia.status === 'ready'};
}
