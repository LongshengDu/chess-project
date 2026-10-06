import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defineConfig, normalizePath } from 'vite';
import { transformPlaySource } from './src/adapters/play-transform.js';
import { simplifyHeader } from './src/adapters/header-transform.js';

const local = (path) => normalizePath(fileURLToPath(new URL(path, import.meta.url)));
const upstream = local('../deps/maia-platform-frontend/').replace(/\/$/, '');
const packages = Object.keys(JSON.parse(readFileSync(local('./package.json'))).dependencies);

// External source files resolve their packages from this frontend's lockfile.
const packageAliases = packages.map((name) => ({
  find: new RegExp(`^${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?=/|$)`),
  // The explicit ESM entry keeps toast's default export callable when upstream
  // source is bundled outside its original Next.js package boundary.
  replacement: local(`./node_modules/${name}${name === 'react-hot-toast' ? '/dist/index.mjs' : ''}`),
}));

function maiaAssets() {
  const assets = [
    ['maia-no-bg.png', 'maia-no-bg.png'],
    ['favicon.png', 'favicon.png'],
    ['maia-ios-icon.png', 'maia-ios-icon.png'],
  ];
  function collect(directory) {
    for (const entry of readdirSync(`${upstream}/public/${directory}`, {withFileTypes:true})) {
      const name = `${directory}/${entry.name}`;
      if (entry.isDirectory()) collect(name);
      else assets.push([name, name]);
    }
  }
  collect('assets/pieces');
  collect('assets/sound');
  return {
    name: 'maia-assets',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const asset = assets.find(([name]) => decodeURI(req.url.split('?')[0]) === `/${name}`);
        if (!asset) return next();
        res.setHeader('Content-Type', asset[0].endsWith('.svg') ? 'image/svg+xml' : asset[0].endsWith('.mp3') ? 'audio/mpeg' : 'image/png');
        res.end(readFileSync(`${upstream}/public/${asset[1]}`));
      });
    },
    generateBundle() {
      for (const [fileName, source] of assets) {
        this.emitFile({ type: 'asset', fileName, source: readFileSync(`${upstream}/public/${source}`) });
      }
    },
  };
}

// Redirect both absolute and relative upstream imports, including barrel imports
// such as useAnalysisController's "..". All UI/controller source stays upstream.
const adapters = {
  'hooks': 'hooks.js', 'contexts': 'contexts.js', 'components': 'components.js',
  'components/Analysis': 'analysis-components.js', 'lib': 'lib.js',
  'api': 'api.js', 'api/analysis': 'api.js',
  'contexts/AuthContext': 'auth.jsx', 'contexts/ModalContext': 'auth.jsx',
  'hooks/useLeaderboardStatus': 'auth.jsx',
  'hooks/useAnalysisController/useDeepAnalysis': 'deep-analysis.js',
  'hooks/useAnalysisController/useAutoSave': 'analysis-save.js',
  'contexts/MaiaEngineContext': 'engines.jsx',
  'contexts/StockfishEngineContext': 'engines.jsx',
};
function analysisAdapters() {
  const root = normalizePath(`${upstream}/src`).replace(/\/$/, '');
  const patch = (code, from, to) => {
    if (!code.includes(from)) throw new Error(`Upstream compatibility patch needs review: ${from}`);
    return code.replace(from, to);
  };
  return {
    name:'local-analysis-adapters', enforce:'pre',
    resolveId(source, importer) {
      let resolved = source;
      if (source.startsWith('src/') || source.startsWith('@maia/')) {
        resolved = `${root}/${source.replace(/^(src|@maia)\//, '')}`;
      } else if (source.startsWith('.') && importer) {
        resolved = normalizePath(path.resolve(path.dirname(importer), source));
      }
      const key = normalizePath(resolved).replace(`${root}/`, '').replace(/\.(tsx?|jsx?)$/, '').replace(/\/index$/, '');
      if (adapters[key]) return local(`./src/adapters/${adapters[key]}`);
      if (['next/head','next/link','next/image'].includes(source)) return `\0${source}`;
    },
    load(id) {
      if (id.startsWith('\0next/')) {
        const name = {head:'Head',link:'Link',image:'Image'}[id.slice(6)];
        return `export { ${name} as default } from ${JSON.stringify(local('./src/adapters/next.jsx'))};`;
      }
    },
    transform(code, id) {
      code = code.replace(/\r\n/g, '\n');
      const playCode = transformPlaySource(code,normalizePath(id),local);
      if (playCode !== undefined) return playCode;
      if (normalizePath(id).endsWith('/pages/analysis/[...id].tsx')) {
        // Python accepts full PGN variations; upstream's legacy backend flattened them.
        return patch(
          patch(code, 'normalizeCustomPgnForBackendStore(data)', 'data'),
          '<MovesByRating', '<MovesByRating positionKey={controller.currentNode?.fen}',
        );
      }
      if (normalizePath(id).endsWith('/components/Common/Header.tsx')) {
        return simplifyHeader(code);
      }
      if (normalizePath(id).endsWith('/components/Board/GameplayInterface.tsx')) {
        // Share the viewport among the upstream board and both side panels;
        // its fixed 22rem sidebar plus a 75vh board otherwise crush the controls.
        code = patch(code,
          'mx-auto mt-2 flex w-[90%] flex-row items-start justify-between gap-3',
          'mx-auto mt-2 grid w-[94%] max-w-[1600px] grid-cols-[minmax(14rem,0.9fr)_minmax(0,2fr)_minmax(16rem,1fr)] items-start gap-3');
        code = patch(code,
          'flex h-[75vh] max-h-[75vh] min-h-[75vh] w-[22rem] min-w-[22rem] max-w-[22rem] flex-shrink-0 flex-col overflow-hidden',
          'flex h-[75vh] max-h-[75vh] min-h-[75vh] w-full min-w-0 flex-col overflow-hidden');
        return patch(code,
          'relative flex aspect-square w-full max-w-[75vh] flex-shrink-0',
          'relative flex aspect-square w-full max-w-[75vh] justify-self-center');
      }
      if (normalizePath(id).endsWith('/useAnalysisController/useAnalysisController.ts')) {
        return patch(patch(code,
          'const deepAnalysisController = useDeepAnalysis({',
          'const deepAnalysisController = useDeepAnalysis({ setAnalysisState, currentMaiaModel, gameId: game.id,'),
          '    enableEngineAnalysis,',
          '    enableEngineAnalysis && !deepAnalysisController.progress.isAnalyzing,');
      }
      if (normalizePath(id).endsWith('/useAnalysisController/useEngineAnalysis.ts')) {
        code = code.replaceAll('if (!currentNode || !enabled) return',
          'if (!currentNode || !enabled || maia.profiling) return');
        // Wait for candidate probabilities once instead of starting Stockfish,
        // cancelling it when Maia arrives, and repeating the same search.
        code = patch(code,
          'currentNode.analysis.stockfish?.depth >= targetDepth',
          'stockfish.satisfies(currentNode.analysis.stockfish, targetDepth, currentNode.analysis.maia?.[currentMaiaModel]?.policy, currentNode.mainChild?.move)');
        return patch(patch(code,
          "if (!currentNode || !enabled || maia.profiling) return\n\n    const shouldForceStockfishRerun",
          "if (!currentNode || !enabled || maia.profiling || !currentNode.analysis.maia?.[currentMaiaModel]) return\n\n    const shouldForceStockfishRerun"),
          '      cancelled = true\n      clearTimeout(timeoutId)',
          '      cancelled = true\n      if (!stockfish.isBatchRunning()) stockfish.stopEvaluation()\n      clearTimeout(timeoutId)');
      }
      if (normalizePath(id) === `${root}/lib/analysis.ts`) {
        code = `import { terminalEvaluation } from ${JSON.stringify(local('./src/adapters/analysis-data.js'))};\n` + code;
        code = patch(code,
          'const stockfishEval = reconstructCachedStockfishAnalysis(',
          "const stockfishEval = 'terminal_cp' in stockfish ? terminalEvaluation(stockfish) : reconstructCachedStockfishAnalysis(");
        // Preserve stage coverage across refreshes. Upstream's hosted cache only
        // serializes a single depth, which would lose the per-move distinction.
        return patch(patch(code,
          '        depth: node.analysis.stockfish.depth,',
          '        ...node.analysis.stockfish,\n        depth: node.analysis.stockfish.depth,'),
          '            node.addStockfishAnalysis(stockfishEval)',
          '            node.addStockfishAnalysis({ ...stockfishEval, ...stockfish })');
      }
      if (normalizePath(id) === `${root}/types/node.ts`) {
        code = patch(code, 'if (!stockfishEval || stockfishEval.depth < 12) {',
          'if (!stockfishEval || stockfishEval.depth < 12 || (stockfishEval.root_move_depth_vec?.[move] ?? stockfishEval.depth) < 12) {');
        return patch(patch(code, 'const shouldReplaceSameDepth =',
          'const shouldReplaceSameDepth = stockfishEval.complete === true ||'),
          'if (existingStockfish.depth > stockfishEval.depth) {',
          'if (existingStockfish.depth > stockfishEval.depth && !(stockfishEval.complete === true && existingStockfish.complete === false)) {');
      }
      if (normalizePath(id).endsWith('/components/Analysis/AnalysisConfigModal.tsx')) {
        code = `import { StockfishEngineContext } from ${JSON.stringify(local('./src/adapters/engines.jsx'))};\n` + code;
        code = patch(code, '  const depthOptions = [',
          '  const { searchConfig } = React.useContext(StockfishEngineContext)\n  const bounded = searchConfig?.strategy === "bounded"\n  const budget = (depth) => searchConfig?.budgets[depth]\n  const depthOptions = [');
        for (const [depth, name, description] of [[12,'Fast','Quick surface-level analysis'], [15,'Balanced','Deeper analysis with good speed'], [18,'Deep','Thorough analysis with slower speed']]) {
          code = patch(code, `label: '${name} (d${depth})'`, `label: bounded ? '${name} (up to d${depth})' : '${name} (d${depth})'`);
          code = patch(code, `description: '${description}'`, `description: bounded ? \`Up to \${budget(${depth})} \${budget(${depth}) === 1 ? 'second' : 'seconds'} of search per position\` : \`Depth target with a \${searchConfig?.max_budgets[${depth}]}-second search limit\``);
        }
        code = patch(code, '  ]\n\n  const handleConfirm',
          '  ].filter(option => searchConfig?.budgets[option.value] !== undefined)\n\n  useEffect(() => {\n    if (searchConfig && searchConfig.budgets[selectedDepth] === undefined) setSelectedDepth(searchConfig.default_depth)\n  }, [searchConfig, selectedDepth])\n\n  const handleConfirm');
        return patch(code,
          'Higher depths provide more accurate analysis but take longer to\n              complete. You can cancel the analysis at any time. Analysis will\n              persist even after you close the tab,',
          '{bounded ? "Search stops at the depth target or time budget. The actual depth reached is shown; difficult positions may finish below the target. " : "Search targets the selected depth; exceeding the maximum search time stops the run. "}\n              Completed analysis is saved locally. Keep this tab open while analysis runs.');
      }
      if (normalizePath(id).endsWith('/components/Analysis/Highlight.tsx')) {
        code = `import { depthLabel } from ${JSON.stringify(local('./src/adapters/search-policy.js'))};\n` + code;
        code = patch(code, 'const stockfishDepthLabel = stockfishDepth ? `d${stockfishDepth}` : null',
          'const stockfishDepthLabel = depthLabel(moveEvaluation?.stockfish)');
        code = patch(code, '    const mateEntries = Object.entries(stockfish.mate_vec ?? {})',
          '    if (stockfish.best_move) {\n      const mate = stockfish.mate_vec?.[stockfish.best_move];\n      if (mate !== undefined) return formatMateDisplay(mate);\n      const cp = stockfish.cp_vec[stockfish.best_move];\n      return `${cp > 0 ? "+" : ""}${(cp / 100).toFixed(2)}`;\n    }\n    const mateEntries = Object.entries(stockfish.mate_vec ?? {})');
        return patch(patch(code, 'SF 17: Engine Moves', 'SF: Engine Moves'), "'Stockfish 17'", "'Stockfish'");
      }
      if (normalizePath(id).endsWith('/useAnalysisController/utils.ts')) {
        return patch(code, '[...moves].sort((a, b) => {',
          '[...moves].sort((a, b) => {\n    if (stockfish.engine_moves?.length) {\n      const rank = (move) => { const index = stockfish.engine_moves.indexOf(move); return index < 0 ? 999 : index; };\n      if (rank(a) !== rank(b)) return rank(a) - rank(b);\n    }');
      }
      if (normalizePath(id).endsWith('/useAnalysisController/useBoardDescription.ts')) {
        code = patch(code, 'const stockfishEvals = moveEvaluation.stockfish.cp_vec',
          'const stockfishEvals = Object.fromEntries(Object.entries(moveEvaluation.stockfish.cp_vec).filter(([move]) => (moveEvaluation.stockfish.root_move_depth_vec?.[move] ?? moveEvaluation.stockfish.depth) >= 12))');
        return patch(code, '    moveEvaluation?.stockfish?.depth,', '    moveEvaluation?.stockfish,');
      }
    },
  };
}

export default defineConfig({
  plugins: [analysisAdapters(), maiaAssets()],
  esbuild: { jsx: 'automatic' },
  build: {
    rolldownOptions: {
      onwarn(warning, warn) {
        // This is an entirely client-rendered app; RSC directives have no role.
        if (warning.code === 'MODULE_LEVEL_DIRECTIVE' && warning.message.includes('use client')) return;
        warn(warning);
      },
    },
  },
  resolve: {
    alias: [
      { find: 'next/router', replacement: local('./src/adapters/router.jsx') },
      { find: 'posthog-js', replacement: local('./src/adapters/telemetry.js') },
      { find: /^src\//, replacement: `${upstream}/src/` },
      { find: /^@maia\//, replacement: `${upstream}/src/` },
      ...packageAliases,
    ],
  },
  server: {
    fs: { allow: ['..'] },
    proxy: { '/api': process.env.MAIA_BACKEND_URL || 'http://127.0.0.1:5000' },
  },
});
