// This is the actual upstream page, not a reconstruction of its UI.
import React, { useContext, useEffect, useState } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import AnalysisPage from '@maia/pages/analysis/[...id].tsx';
import CustomLinkPage from '@maia/pages/analysis/custom.tsx';
import PlayMaiaPage from '@maia/pages/play/maia.tsx';
import { Header } from '@maia/components/Common/Header';
import { Footer } from '@maia/components/Common/Footer';
import { SettingsProvider } from '@maia/contexts/SettingsContext';
import { SoundProvider } from '@maia/contexts/SoundContext';
import { WindowSizeContextProvider } from '@maia/contexts/WindowSizeContext';
import { TourProvider } from '@maia/contexts/TourContext';
import { AnalysisListContextProvider } from '@maia/contexts/AnalysisListContext';
import { RouterProvider, useRouter } from './adapters/router';
import { EngineProvider } from './adapters/engines';
import { request } from './adapters/api';
import { ModalContext, ModalContextProvider } from './adapters/auth';

function PlayLanding() {
  const {setPlaySetupModalProps} = useContext(ModalContext);
  useEffect(() => {setPlaySetupModalProps({playType:'againstMaia'});},[setPlaySetupModalProps]);
  return <div className="m-12 text-center text-primary">
    <button className="rounded border border-white/10 bg-white/5 px-6 py-3" onClick={() => setPlaySetupModalProps({playType:'againstMaia'})}>Play Maia</button>
  </div>;
}

function Page() {
  const router = useRouter();
  const {playConfig,configError,reloadPlayConfig} = useContext(ModalContext);
  const [error, setError] = useState(null);
  useEffect(() => {
    const showError = event => setError(event.detail);
    const showPlayError = event => toast.error(event.detail,{id:'local-play-error'});
    window.addEventListener('local-analysis-error', showError);
    window.addEventListener('local-play-error',showPlayError);
    return () => {
      window.removeEventListener('local-analysis-error', showError);
      window.removeEventListener('local-play-error',showPlayError);
    };
  }, []);
  useEffect(() => setError(null),[router.pathname]);
  useEffect(() => {
    if (router.query.id || router.pathname === '/analysis/custom' || router.pathname.startsWith('/play')) return;
    let active = true;
    request('/bootstrap').then(({ game_id }) => {
      if (active) router.replace(`/analysis/${game_id}/custom`);
    }).catch(e => setError(e.message));
    return () => { active = false; };
  }, [router]);
  if (error) return <div className="m-12 text-center text-primary" role="alert"><p>{error}</p>
    <a className="underline" href="/analysis">Open saved analyses</a></div>;
  if (router.pathname === '/play') return <PlayLanding />;
  if (router.pathname === '/play/maia') {
    if (configError) return <div className="m-12 text-center text-primary" role="alert"><p>{configError}</p><button className="underline" onClick={reloadPlayConfig}>Retry</button></div>;
    return playConfig ? <PlayMaiaPage /> : <p className="m-12 text-center text-primary">Loading game settings…</p>;
  }
  if (router.pathname === '/analysis/custom') return <CustomLinkPage />;
  return router.query.id ? <AnalysisPage /> : <p className="m-12 text-center text-primary">Opening analysis…</p>;
}

export default function App() {
  return <RouterProvider><SettingsProvider><SoundProvider><WindowSizeContextProvider>
    <TourProvider><EngineProvider><ModalContextProvider><AnalysisListContextProvider>
      <div className="font-sans app-container">
        <Header />
        <div className="content-container">
          <div className="pointer-events-none fixed inset-0" style={{background:'radial-gradient(circle at center, rgba(239,68,68,0.06) 0%, transparent 70%)'}} />
          <Page />
        </div>
        <Footer />
      </div>
      <Toaster position="bottom-right" />
    </AnalysisListContextProvider></ModalContextProvider></EngineProvider></TourProvider>
  </WindowSizeContextProvider></SoundProvider></SettingsProvider></RouterProvider>;
}
