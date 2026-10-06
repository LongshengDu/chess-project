import { createContext, useCallback, useEffect, useMemo, useState } from 'react';
import { toast } from 'react-hot-toast';
import { PlaySetupModal } from '@maia/components/Common/PlaySetupModal';
import { fetchPlayConfig } from './play-api.js';
// Local analysis and play share no hosted account or authentication.
export const AuthContext = createContext({
  user: { displayName: 'Local player' },
  connectLichess: () => toast('Local play does not require sign in.'),
  logout: () => {},
});
export const ModalContext = createContext({
  setPlaySetupModalProps: () => {},
});
export function ModalContextProvider({children}) {
  const [playSetupModalProps,setPlaySetupModalProps] = useState();
  const [playConfig,setPlayConfig] = useState(null);
  const [configError,setConfigError] = useState(null);
  const reloadPlayConfig = useCallback(() => {
    setConfigError(null);
    fetchPlayConfig().then(setPlayConfig).catch(error => setConfigError(error.message));
  },[]);
  useEffect(reloadPlayConfig,[reloadPlayConfig]);
  const defaultMaiaVersion = playConfig ? `maia_kdd_${playConfig.player_rating}` : undefined;
  const value = useMemo(() => ({playSetupModalProps,setPlaySetupModalProps,
    playConfig,configError,reloadPlayConfig,defaultMaiaVersion}),
    [playSetupModalProps,playConfig,configError,reloadPlayConfig,defaultMaiaVersion]);
  return <ModalContext.Provider value={value}>
    {children}
    {playSetupModalProps && playConfig && <PlaySetupModal
      maiaVersion={defaultMaiaVersion} {...playSetupModalProps} playType="againstMaia" />}
    {playSetupModalProps && !playConfig && <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70">
      <div className="rounded border border-white/10 bg-backdrop p-6 text-primary" role={configError ? 'alert' : 'status'}>
        <p>{configError || 'Loading game settings…'}</p>
        {configError && <button className="mt-3 underline" onClick={reloadPlayConfig}>Retry</button>}
        <button className="ml-4 underline" onClick={() => setPlaySetupModalProps(undefined)}>Close</button>
      </div>
    </div>}
  </ModalContext.Provider>;
}
export const useLeaderboardStatus = () => ({ status: null, loading: false });
