# Bounded Stockfish search: full-game comparison

The same example.pgn completed in **204.477s**, versus **902.775s**: **4.42× faster**, **77.4% less elapsed time**.

Both runs used the actual Analyze Entire Game button, cold result caches, Maia3 79M on CUDA, all 21 ratings, Maia 1500 candidates, Stockfish 19 with 8 threads and 512 MB hash. These are individual desktop runs, not repeated controlled trials.

## Search policy

The bounded run targets d18 and shares a 6-second native-search budget per position. It screens all legal moves up to d6 with at most 0.3s, prioritizes a single best engine line (40% of the budget), probes the played move and Maia candidates (40%, shared), then searches four alternatives with the remaining time. Unused time flows to later work. Every native search receives both depth and movetime; it stops at whichever limit is reached first.

## Slowest original positions

| Position before | Before s | After s | Best depth | Played depth | All candidates at target? |
| --- | ---: | ---: | ---: | ---: | --- |
| 17... g6 | 339.287 | 6.165 | 15 | 18 | False |
| 19... Ncxe4 | 128.769 | 6.134 | 15 | 13 | False |
| 19. Nxe4 | 56.145 | 6.131 | 18 | 18 | False |
| 13... Bg6 | 36.560 | 6.116 | 18 | 18 | False |
| 16... gxh4 | 33.920 | 6.058 | 18 | 18 | False |
| 18. Bxg6 | 25.945 | 6.130 | 18 | 18 | False |
| 18... e4 | 24.864 | 6.119 | 12 | 15 | False |
| 17. Rxh4 | 24.825 | 6.183 | 14 | 15 | False |
| 15... Nc5 | 22.823 | 6.116 | 16 | 13 | False |
| 12... h6 | 20.833 | 6.118 | 18 | 18 | False |

## Evaluation tradeoffs

- Best move agrees on 35 of 51 nonterminal positions.
- Median absolute best-score change: 9 centipawns, excluding mate scores.
- Median absolute played-move score change: 9.5 centipawns; maximum 503, excluding mate scores.
- Achieved best-line depth: [12, 18]; played-move depth: [13, 18].
- All selected candidates reached d18 in 27 nonterminal positions. All legal moves have a recorded score: True.
- The uncapped run is a reference, not ground truth. Scores can change with search order, depth, time and multithreading. The bounded policy trades some depth for predictable waiting time.
- Existing deeper studies are retained. Time-limited completion is saved with actual per-move depths and a time-limit label. It cannot satisfy an exhaustive request.

## Sources

- [Stockfish UCI documentation](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html): combined limits stop at the first limit; MultiPV 1 gives the best performance; threads should match available CPU cores.
- [python-chess engine limits](https://python-chess.readthedocs.io/en/latest/engine.html#chess.engine.Limit): native time and depth parameters.

Full data: [position comparison](comparison.csv), [component report](REPORT.md), [all searches](searches.csv), [saved evaluations](saved-analysis.json).
