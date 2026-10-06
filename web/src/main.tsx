import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './upstream-app';
import '../../deps/maia-platform-frontend/src/styles/tailwind.css';
import '../../deps/maia-platform-frontend/src/styles/themes.css';
import 'chessground/assets/chessground.base.css';
import 'chessground/assets/chessground.brown.css';
import 'chessground/assets/chessground.cburnett.css';

ReactDOM.createRoot(document.getElementById('root')!).render(<App />);
