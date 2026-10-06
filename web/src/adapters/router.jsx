import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { hrefString, isLocalRoute, queryForUrl } from './routes.js';

const Context = createContext(null);
const listeners = new Map();
const events = {
  on: (name, fn) => { if (!listeners.has(name)) listeners.set(name,new Set()); listeners.get(name).add(fn); },
  off: (name, fn) => listeners.get(name)?.delete(fn),
};
const emit = (name,...args) => listeners.get(name)?.forEach(fn => fn(...args));
export const useRouter = () => useContext(Context);
export function RouterProvider({ children }) {
  const [location, setLocation] = useState(() => ({internal:window.location.pathname+window.location.search,
    displayed:window.location.pathname+window.location.search}));
  useEffect(() => {
    const update = () => {
      const target = window.location.pathname+window.location.search;
      emit('routeChangeStart',target);
      setLocation({internal:target,displayed:target});
      emit('routeChangeComplete',target);
    };
    window.addEventListener('popstate', update);
    return () => window.removeEventListener('popstate', update);
  }, []);
  const navigate = useCallback(async (href, as, replace) => {
      const target = hrefString(href,window.location.origin);
      const displayed = as ? hrefString(as,window.location.origin) : target;
      if (target === '/401') {
        window.dispatchEvent(new CustomEvent('local-analysis-error',{detail:'The local game could not be loaded. Please try again.'}));
        return false;
      }
      if (!isLocalRoute(target)) {
        window.location.assign(`https://www.maiachess.com${target}`);
        return true;
      }
      emit('routeChangeStart',displayed);
      window.history[replace ? 'replaceState' : 'pushState']({}, '', displayed);
      // Next's `as` URL hides the session ID; keep it in the live router query.
      // Refresh/back intentionally starts a fresh game, like the upstream page.
      setLocation({internal:target,displayed});
      emit('routeChangeComplete',displayed);
      return true;
  },[]);
  const push = useCallback((href,as) => navigate(href,as,false),[navigate]);
  const replace = useCallback((href,as) => navigate(href,as,true),[navigate]);
  const router = useMemo(() => {
    const url = new URL(location.internal, window.location.origin);
    return {
      pathname: url.pathname, asPath: location.displayed, isReady: true, events,
      query: queryForUrl(url),push,replace,
      back: () => window.history.back(), prefetch: async () => {},
    };
  }, [location,push,replace]);
  return <Context.Provider value={router}>{children}</Context.Provider>;
}
