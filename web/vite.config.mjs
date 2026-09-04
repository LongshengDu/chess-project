import { fileURLToPath } from 'node:url';

import { defineConfig } from 'vite';

const lilaLib = fileURLToPath(new URL('../deps/lichess-lila/ui/lib/src/', import.meta.url));
const webModules = fileURLToPath(new URL('./node_modules/', import.meta.url));

export default defineConfig({
  resolve: {
    alias: [
      {
        find: /^@lichess-org\/chessground\/assets\/(.+)$/,
        replacement: `${webModules}/@lichess-org/chessground/assets/$1`,
      },
      {
        find: /^@lichess-org\/chessground\/(.+)$/,
        replacement: `${webModules}/@lichess-org/chessground/dist/$1.js`,
      },
      {
        find: /^@lichess-org\/chessground$/,
        replacement: `${webModules}/@lichess-org/chessground/dist/chessground.js`,
      },
      { find: /^@badrap\/result$/, replacement: `${webModules}/@badrap/result/dist/mjs/index.mjs` },
      { find: /^chessops\/(.+)$/, replacement: `${webModules}/chessops/dist/esm/$1.js` },
      { find: /^chessops$/, replacement: `${webModules}/chessops/dist/esm/index.js` },
      { find: /^snabbdom$/, replacement: `${webModules}/snabbdom/build/index.js` },
      { find: /^@\/game$/, replacement: `${lilaLib}/game/interfaces.ts` },
      { find: /^@\/view$/, replacement: `${lilaLib}/view/snabbdom.ts` },
      { find: /^@\//, replacement: `${lilaLib}/` },
    ],
  },
  server: {
    fs: {
      allow: ['..'],
    },
  },
});
