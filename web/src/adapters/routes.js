export function hrefString(href, origin = 'http://localhost') {
  if (typeof href === 'string') return href;
  const url = new URL(href.pathname || '/',origin);
  for (const [key,value] of Object.entries(href.query || {})) {
    if (value === null || value === undefined) continue;
    for (const item of Array.isArray(value) ? value : [value]) url.searchParams.append(key,String(item));
  }
  return url.pathname+url.search+(href.hash || '');
}

export function isLocalRoute(path) {
  const pathname = path.split(/[?#]/,1)[0];
  return pathname === '/' || pathname === '/analysis' || pathname.startsWith('/analysis/')
    || pathname === '/play' || pathname === '/play/maia';
}

export function queryForUrl(url) {
  const parts = url.pathname.split('/').filter(Boolean);
  return {...Object.fromEntries(url.searchParams),
    ...(parts[0] === 'analysis' && parts.length >= 3 ? {id:parts.slice(1)} : {})};
}
