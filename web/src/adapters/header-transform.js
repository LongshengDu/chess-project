// Keep the pinned upstream Header's markup and styling, while exposing only
// the two local features. Every edit has an exact, unique source anchor.
function replaceOnce(code, from, to) {
  const first = code.indexOf(from);
  if (first < 0 || code.indexOf(from, first + from.length) >= 0) {
    throw new Error(`Upstream Header compatibility patch needs review: ${from}`);
  }
  return code.slice(0, first) + to + code.slice(first + from.length);
}

function removeBetween(code, from, until) {
  const first = code.indexOf(from);
  const last = code.indexOf(until, first + from.length);
  if (first < 0 || last < 0 || code.indexOf(from, first + from.length) >= 0
      || code.indexOf(until, last + until.length) >= 0) {
    throw new Error(`Upstream Header compatibility patch needs review: ${from}`);
  }
  return code.slice(0, first) + code.slice(last);
}

export function simplifyHeader(source) {
  let code = source.replace(/\r\n/g, '\n');

  // Remove hosted-profile construction, then unrelated navigation blocks.
  code = removeBetween(code, '  const userInfo = ', '  const desktopLayout = (');
  code = removeBetween(code,
    '          <div\n            className="relative"\n            onMouseEnter={() => setShowPlayDropdown(true)}',
    '          <Link\n            href="/analysis"');
  code = replaceOnce(code, '          <Link\n            href="/analysis"',
    `          <button
            onClick={() => startGame('againstMaia')}
            className={\`px-2 py-1 transition-all duration-200 hover:!text-primary \${router.pathname.startsWith('/play') ? '!text-primary' : '!text-primary/80'}\`}
          >
            Play Maia
          </button>
          <Link
            href="/analysis"`);
  code = removeBetween(code,
    '          <Link\n            href="/puzzles"',
    '        </div>\n      </div>\n      <div className="hidden flex-row items-center gap-3 md:flex">');
  code = removeBetween(code,
    '      <div className="hidden flex-row items-center gap-3 md:flex">',
    '    </div>\n  )\n\n  const mobileLayout = (');
  code = removeBetween(code,
    '            <div className="flex flex-col items-start justify-center gap-3">\n              <button>PLAY</button>',
    '            <Link href="/analysis"');
  code = replaceOnce(code, '            <Link href="/analysis"',
    `            <button
              className="text-left"
              onClick={() => startGame('againstMaia')}
            >
              Play Maia
            </button>
            <Link href="/analysis"`);
  code = removeBetween(code,
    '            <Link href="/puzzles"',
    '          </div>\n          <div className="flex w-full flex-row items-center gap-3 px-12">');
  code = removeBetween(code,
    '          <div className="flex w-full flex-row items-center gap-3 px-12">',
    '        </div>\n      )}');

  // Both direct actions close the mobile menu before opening game setup.
  code = replaceOnce(code,
    '      setPlaySetupModalProps({ playType: playType })',
    '      setShowMenu(false)\n      setPlaySetupModalProps({ playType: playType })');

  // These imports/state values existed solely for the removed hosted controls.
  for (const line of [
    "import { DiscordIcon } from './Icons'\n",
    "import { AuthContext } from 'src/contexts/AuthContext'\n",
    "import { LeaderboardNavBadge } from '../Leaderboard/LeaderboardNavBadge'\n",
    "import { useLeaderboardStatus } from 'src/hooks/useLeaderboardStatus'\n",
    "import { motion, AnimatePresence } from 'framer-motion'\n",
    '  const [showPlayDropdown, setShowPlayDropdown] = useState(false)\n',
    '  const [showMoreDropdown, setShowMoreDropdown] = useState(false)\n',
    '  const [showProfileDropdown, setShowProfileDropdown] = useState(false)\n',
    '  const { user, connectLichess, logout } = useContext(AuthContext)\n',
    '  const isCompactDesktopNav = !isMobile && width > 0 && width < 1300\n',
  ]) code = replaceOnce(code, line, '');
  code = replaceOnce(code,
    '  const { isMobile, width } = useContext(WindowSizeContext)',
    '  const { isMobile } = useContext(WindowSizeContext)');
  code = removeBetween(code,
    '  // Get leaderboard status for the logged in user\n',
    '  const router = useRouter()');
  return code;
}
