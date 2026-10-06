import { useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { collectEngineAnalysisData, generateAnalysisCacheKey } from '@maia/lib/analysis';
import { StockfishEngineContext } from './engines';
import { storeGameAnalysisCache } from './api';

// Save interactive/cancelled position work; a completed full-game result was
// already committed by Python and must never trigger another bulk POST.
export function useAutoSave({game, gameTree, enableAutoSave, analysisState}) {
  const stockfish = useContext(StockfishEngineContext);
  const session = useMemo(() => ({game, tree:gameTree, fullAnalysis:gameTree.fullAnalysis,
    savedKey:generateAnalysisCacheKey(collectEngineAnalysisData(gameTree)), dirty:false, saving:false}),
  [gameTree, game.id, game.type]);
  session.enabled = enableAutoSave;
  const current = useRef(session);
  current.current = session;
  const [autoSave, setAutoSave] = useState({hasUnsavedChanges:false, isSaving:false, status:'saved'});
  const batchRunning = stockfish.isBatchRunning();
  const publish = useCallback(() => {
    if (current.current === session) setAutoSave({hasUnsavedChanges:session.dirty,
      isSaving:session.saving, status:session.saving ? 'saving' : session.dirty ? 'unsaved' : 'saved'});
  }, [session]);

  const saveAnalysis = useCallback(async () => {
    const id = session.game.id || session.tree.savedGameId;
    if (!id || !session.enabled || session.game.type === 'tournament' ||
        !session.dirty || session.saving || stockfish.isBatchRunning()) return;
    const positions = collectEngineAnalysisData(session.tree);
    if (!positions.some(position => position.maia || position.stockfish?.depth >= 12 ||
        Object.keys(position.stockfish?.mate_vec || {}).length)) return;
    const key = generateAnalysisCacheKey(positions);
    if (key === session.savedKey) { session.dirty = false; publish(); return; }
    session.saving = true;
    publish();
    const fullAnalysis = session.tree.fullAnalysis;
    try {
      await storeGameAnalysisCache(id, positions);
      // A request launched before the batch may finish after the canonical
      // result. Keep that newer baseline instead of scheduling a bulk re-save.
      if (session.tree.fullAnalysis !== fullAnalysis) {
        session.fullAnalysis = session.tree.fullAnalysis;
        session.savedKey = generateAnalysisCacheKey(session.fullAnalysis.positions);
      } else {
        session.savedKey = key;
      }
      session.dirty = generateAnalysisCacheKey(collectEngineAnalysisData(session.tree)) !== session.savedKey;
    } catch (error) {
      console.warn('Failed to save interactive analysis:', error);
    } finally {
      session.saving = false;
      publish();
    }
  }, [session, stockfish, publish]);

  useEffect(() => {
    if (batchRunning) return;
    if (session.fullAnalysis !== gameTree.fullAnalysis) {
      session.fullAnalysis = gameTree.fullAnalysis;
      session.savedKey = generateAnalysisCacheKey(collectEngineAnalysisData(gameTree));
      session.dirty = false;
    } else if (analysisState > 0) {
      session.dirty = generateAnalysisCacheKey(collectEngineAnalysisData(gameTree)) !== session.savedKey;
    }
    publish();
  }, [analysisState, gameTree, session, batchRunning, publish]);

  useEffect(() => {
    if (!enableAutoSave) return;
    const timer = setInterval(saveAnalysis, 10000);
    return () => { clearInterval(timer); saveAnalysis(); };
  }, [enableAutoSave, saveAnalysis]);
  return {saveAnalysis, autoSave};
}
