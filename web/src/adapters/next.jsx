import React, { forwardRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { useRouter } from './router';
import { hrefString, isLocalRoute } from './routes.js';

export function Head({children}) {
  const nodes = React.Children.toArray(children);
  const title = nodes.find(node => React.isValidElement(node) && node.type === 'title');
  const titleText = title ? React.Children.toArray(title.props.children).join('') : null;
  useEffect(() => { if (titleText !== null) document.title = titleText; },[titleText]);
  return createPortal(nodes.filter(node => !React.isValidElement(node) || node.type !== 'title'),document.head);
}
export const Image = ({ priority, fill, unoptimized, quality, layout, objectFit, ...props }) =>
  <img {...props} style={{...props.style, ...(fill ? {position:'absolute',width:'100%',height:'100%',objectFit:objectFit || 'cover'} : {})}} />;
export const Link = forwardRef(function Link({ href, children, onClick, prefetch, replace, scroll, shallow, passHref, ...props }, ref) {
  const router = useRouter();
  const path = hrefString(href,window.location.origin);
  const local = isLocalRoute(path);
  const target = path.startsWith('/') && !local ? `https://www.maiachess.com${path}` : path;
  return <a {...props} ref={ref} href={target} onClick={event => {
    onClick?.(event);
    if (local && !event.defaultPrevented && !event.ctrlKey && !event.metaKey && !event.shiftKey && event.button === 0 && props.target !== '_blank') {
      event.preventDefault();
      router[replace ? 'replace' : 'push'](path);
    }
  }}>{children}</a>;
});
