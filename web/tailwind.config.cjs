const source = require('node:fs').readFileSync(
  require.resolve('../deps/maia-platform-frontend/tailwind.config.js'), 'utf8',
);
const config = { exports: {} };
// Resolve build plugins from this frontend without installing the upstream app.
require('node:vm').runInNewContext(source, { module: config, require });
const upstream = config.exports;
module.exports = {
  ...upstream,
  content: [
    './vite.config.mjs',
    './src/**/*.{js,jsx,ts,tsx}',
    '../deps/maia-platform-frontend/src/**/*.{js,jsx,ts,tsx}',
  ],
  plugins: [require('@tailwindcss/typography')],
};
