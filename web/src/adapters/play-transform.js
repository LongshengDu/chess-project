// Small compatibility patches keep the actual upstream modal, page and controls.
// Every anchor is checked so an upstream update cannot silently break local play.
export function transformPlaySource(code, id, local) {
  const patch = (from,to) => {
    if (!code.includes(from)) throw new Error(`Upstream play compatibility patch needs review: ${from}`);
    code = code.replace(from,to);
  };
  if (id.endsWith('/pages/play/maia.tsx')) {
    patch('const { setPlaySetupModalProps } = useContext(ModalContext)',
      'const { setPlaySetupModalProps, defaultMaiaVersion } = useContext(ModalContext)');
    patch("maiaVersion || 'maia_kdd_1100'",'maiaVersion || defaultMaiaVersion');
    patch("sampleMoves: sampleMoves == 'true'","sampleMoves: sampleMoves !== 'false'");
    patch('          undefined,\n        )','          undefined,\n          playGameConfig.startFen,\n        )');
    patch('          <PlayMaia\n            id={id as string}', '          <PlayMaia\n            key={id as string}\n            id={id as string}');
    patch("        router.push('/401')", "        window.dispatchEvent(new CustomEvent('local-analysis-error', {detail: e instanceof Error ? e.message : 'Could not start the local game.'}))");
    return code;
  }
  if (id.endsWith('/components/Common/PlaySetupModal.tsx')) {
    patch('props.sampleMoves || true','props.sampleMoves ?? true');
    const start = code.indexOf('const maiaOptions = [');
    const end = code.indexOf('\n]',start);
    if (start < 0 || end < 0) throw new Error('Upstream Maia option list needs review.');
    code = code.slice(0,start)+"const maiaOptions = Array.from({length:21},(_,i) => `maia_kdd_${600+i*100}`)"+code.slice(end+2);
    patch('  const { setPlaySetupModalProps } = useContext(ModalContext)',
      '  const availableMaiaOptions = [...new Set([...maiaOptions, ...(props.maiaVersion ? [props.maiaVersion] : [])])].sort((a,b) => Number(a.split("_").at(-1))-Number(b.split("_").at(-1)))\n  const { setPlaySetupModalProps } = useContext(ModalContext)');
    code = code.replaceAll('maiaOptions.map(', 'availableMaiaOptions.map(');
    return code;
  }
  if (id.endsWith('/hooks/usePlayController/useVsMaiaController.ts')) {
    patch("            : controller.whiteClock) / 1000", "            : controller.whiteClock) / 1000 - (controller.moveList.length > 1 && controller.lastMoveTime > 0 ? (Date.now()-controller.lastMoveTime)/1000 : 0)");
    patch('              simulateMaiaTime ? initialClock : 0,', '              initialClock,');
    patch('              simulateMaiaTime ? maiaClock : 0,',
      '              Math.max(0,maiaClock),\n              controller.game.id,');
    patch('        const nextMove = maiaMoves[\'top_move\']',
      `        if (canceled) return
        if (!maiaMoves.top_move) {
          if (maiaMoves.termination?.type === 'time') {
            controller.expireOnTime(controller.player === 'white' ? 'black' : 'white')
            return
          }
          throw new Error('Maia did not return a legal move.')
        }
        const nextMove = maiaMoves['top_move']`);
    patch('          const delayMs = Math.max(moveDelay * 1000, minimumDelayMs)',
      `          const remainingMs = controller.timeControl !== 'unlimited' && controller.moveList.length > 1
            ? Math.max(0,(controller.player === 'white' ? controller.blackClock : controller.whiteClock)
                - (controller.lastMoveTime > 0 ? Date.now()-controller.lastMoveTime : 0)-20)
            : Infinity
          const delayMs = Math.min(Math.max(moveDelay * 1000, minimumDelayMs),remainingMs)`);
    code = code.replaceAll("jitter: 'full',", "jitter: 'full', numOfAttempts: 3,");
    patch('    makeMaiaMove()', "    makeMaiaMove().catch(error => { if (!canceled) window.dispatchEvent(new CustomEvent('local-play-error', {detail:error.message})) })");
    patch('    submitFn()', "    submitFn().catch(error => window.dispatchEvent(new CustomEvent('local-play-error', {detail:`Could not save game: ${error.message}`})))");
    patch('        safeUpdateRating(response.player_elo, updateRating)',
      '        if (response.player_elo != null) safeUpdateRating(response.player_elo, updateRating)');
    return code;
  }
  if (id.endsWith('/hooks/usePlayController/usePlayController.ts')) {
    patch('  const initialClockValue = baseMinutes * 60 * 1000',
      '  const initialClockValue = (baseMinutes > 0 ? baseMinutes * 60 : incrementSeconds) * 1000');
    patch('      if (moveList.length < 2 && !forceClockUpdate) {\n        return 0',
      '      if (moveList.length < 2 && !forceClockUpdate) {\n        if (moveList.length === 1) setLastMoveTime(Date.now())\n        return 0');
    patch('        overrideTime === undefined ? now - lastMoveTime : overrideTime',
      '        overrideTime === undefined ? (lastMoveTime > 0 ? Math.max(0,now - lastMoveTime) : 0) : overrideTime');
    patch('    setResigned,', '    setResigned,\n    expireOnTime,');
    return code;
  }
  if (id.endsWith('/components/Common/ExportGame.tsx')) {
    code = "import { PlayControllerContext } from 'src/contexts/PlayControllerContext'\n"+code;
    patch('  const controller = useContext(TreeControllerContext)',
      '  const controller = useContext(TreeControllerContext)\n  const playController = useContext(PlayControllerContext)');
    patch("tree.setHeader('Site', 'https://maiachess.com/')",
      `for (const key of ['Site', 'WhiteElo', 'BlackElo', 'TimeControl']) {
      const value = gameTree.getHeader(key)
      if (value !== undefined) tree.setHeader(key, value)
    }
    if (type === 'play') {
      const rating = playController.maiaVersion.replace('maia_kdd_', '')
      const [minutes, increment] = playController.timeControl.split('+').map(Number)
      tree.setHeader('WhiteElo', rating)
      tree.setHeader('BlackElo', rating)
      tree.setHeader('Site', 'lichess.org')
      tree.setHeader('TimeControl', playController.timeControl === 'unlimited' ? '300' : \`\${minutes * 60}+\${increment}\`)
    } else if (!tree.getHeader('Site')) {
      tree.setHeader('Site', window.location.origin)
    }`);
    patch('    game.id,', '    game.id,\n    playController.maiaVersion,\n    playController.timeControl,');
    return code;
  }
  if (id.endsWith('/components/Play/PlayControls.tsx')) {
    code = `import { useRouter } from 'next/router'\nimport toast from 'react-hot-toast'\nimport { savePlayedGameForAnalysis } from ${JSON.stringify(local('./src/adapters/play-api.js'))}\n`+code;
    patch('  const [showResignConfirm, setShowResignConfirm] = useState(false)',
      `  const [showResignConfirm, setShowResignConfirm] = useState(false)
  const [savingAnalysis, setSavingAnalysis] = useState(false)
  const router = useRouter()
  const analyzeGame = async () => {
    if (savingAnalysis) return
    setSavingAnalysis(true)
    try {
      const {game_id} = await savePlayedGameForAnalysis(game)
      await router.push('/analysis/'+game_id+'/custom')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Could not save game for analysis.')
    } finally { setSavingAnalysis(false) }
  }`);
    patch("              onClick={() => {\n                window.open(`/analysis/${game.id}/play`, '_blank')\n              }}",
      '              onClick={analyzeGame}\n              disabled={savingAnalysis}');
    patch('              ANALYZE GAME', "              {savingAnalysis ? 'SAVING GAME…' : 'ANALYZE GAME'}");
    return code;
  }
  if (id.endsWith('/components/Common/StatsDisplay.tsx')) {
    patch('useState<number | undefined>(0)','useState<number | undefined>(undefined)');
    patch("${cachedRating === undefined && 'opacity-0'} ${stats.rating === undefined && 'opacity-50'}",'');
    patch('{cachedRating || 0}',"{stats.rating == null ? 'Unrated' : cachedRating}");
    return code;
  }
  if (id.endsWith('/contexts/AnalysisListContext.tsx')) {
    patch('    if (user?.lichessId) {\n      const playRequest',
      '    if (user) {\n      const playRequest');
    code = code.replaceAll('  }, [user?.lichessId])','  }, [user?.lichessId, router.asPath])');
    patch("              player_color: 'white' | 'black'",
      "              player_color: 'white' | 'black'\n              custom_name?: string\n              is_favorited?: boolean");
    patch("            const raw = game.maia_name.replace('_kdd_', ' ')",
      `            if (type === 'custom') {
              return {id:game.game_id, label:game.custom_name || 'Custom Game',
                result:game.result || '*', type, custom_name:game.custom_name,
                is_favorited:game.is_favorited || false}
            }
            const raw = game.maia_name.replace('_kdd_', ' ')`);
    return code;
  }
}
