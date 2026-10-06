---
name: chess-player-rating
description: Design, implement, review, and evaluate player-rating fitting methods in this chess project's analysis/player_rating package. Use for a new estimator or changes to its mathematics, evidence, uncertainty, scale behavior, diagnostics, or method selection; ordinary game analysis uses chess-analysis.
---

# Player-rating fitting methods

Apply the [project instructions](../project-development/SKILL.md). Read the current [estimator contract](../../../analysis/README.md#player-rating-estimator-interface), [abstract interface](../../../analysis/player_rating/interface.py), and [service](../../../analysis/player_rating/service.py) before implementing a method. Use [chess-analysis](../chess-analysis/SKILL.md) for shared-pipeline changes. This skill defines requirements for method development; it does not itself request an experiment, full-game run, coaching report, or change of default.

## Define the mathematical model

- Estimate strength demonstrated in one game, distinct from the player's long-run account rating and the commercial estimate used for comparison.
- Prefer a universal, intuitive model whose calculation can be explained from a shared accuracy curve, observed accuracy, and explicitly defined variability. Explain what each additional input contributes. Do not add special rules for particular games, players, moves, or known reference errors.
- Keep shared-curve intersections, average accuracy, and variance visible as explanatory quantities. 
- Match the observed and expected statistics: aggregation, quality perspective, position selection, and weighting must describe the same quantity. Distinguish arithmetic mean move accuracy from Lichess full-game accuracy; a change to the aggregation must also change the reference calculation consistently.
- Specify how full legal-move Maia probabilities become expected quality. If exploring a probability cutoff or another selection rule, document and expose it as a method argument; do not silently switch to only the most probable move or freeze an old experiment's cutoff.
- Legally forced moves do not demonstrate decision quality. State their treatment and distinguish them from positions with several legal choices but one likely move. An expected Maia accuracy is a mean, not a ceiling on an individual's performance.
- Define the source and units of uncertainty. Played-move sample variance, conditional Maia move-choice variance, model discrepancy, and uncertainty in a rating are different quantities. Do not use high observed move-to-move variability alone as a justification for broad rating uncertainty.
- Handle missing intersections and rating boundaries through the declared model. Distinguish measured Maia anchors from extrapolated support and prior assumptions; extrapolated curves are not new human evidence. Avoid discontinuous edge-case fixes added to repair named benchmark results.

## Account ratings, ordering, and scales

- Actual Elo may provide declared context, but the method should estimate played strength rather than merely return account strength. Explain its influence and measure sensitivity to plausible account-rating changes; do not import a historical perturbation size or tolerance as an unexplained constant.
- Preserve the within-game ordering implied by the shared curve and observed accuracies through the model's mathematics. State when clipping, rounding, missing information, or flat regions can create ties. Do not use reference labels to reorder predictions afterward.
- Evaluate White/Black ordering, ties, above/below-actual performance, and absolute numerical error separately. A reference tie requires a suitably small fitted gap under the requested evaluation tolerance; a small fitted gap does not imply the reference must be tied. A nonzero reference ordering can be respected even by a small fitted difference.
- Maia's native coordinates are Lichess Blitz. Let the shared service interpret `Site`, `TimeControl`, and explicit scale overrides, normalize inputs once, and convert results once. Do not repeat conversion inside `Rating.fit`.
- Equivalent account inputs expressed on another supported site/time-control scale should produce approximately equivalent converted estimates. Check priors, support, uncertainty, figures, and rounding as well as point estimates. Never fit conversion coefficients to these test-game references.

## Keep evaluation games out of model development data

- **Never use the games collection as a training, calibration, or population corpus.** This includes unlabeled positions, Maia policies, accuracy curves, moments, covariance estimates, normalizers, priors, and any precomputed asset derived from them.
- Removing commercial labels, anonymizing cases, hashing their contents, or leaving out the target game does not permit using the other test games as fitting inputs. Each estimate may use that game's own numeric evidence and declared inputs only.
- Commercial estimate headers belong to evaluation after predictions. They must not reach the estimator, an upstream preprocessing stage, coefficient fitting, or an output-correction rule.
- Derive the method and justify its parameters before comparing references. Requested comparisons can expose a flawed assumption or motivate an explicitly requested minor adjustment; repeated parameter search against the same cases is not a new algorithm or independent validation.
- Do not restore deleted benchmark assets or withdrawn methods. Any proposal requiring learned external information needs genuinely separate development data and clear provenance; the test collection cannot be substituted for it.

## Implement a drop-in method

- Add `analysis/player_rating/<METHOD>.py` defining its own concrete `Rating(PlayerRating)` class with synchronous `fit(evidence)`. The filename is the method identity selected by `ANALYSIS.PLAYER_RATING.METHOD`; no registry or backend/coach branch should be required.
- Keep the estimator stateless and its constructor inexpensive. Fit from the supplied numeric evidence without loading PGNs, other games, reference labels, caches, engines, or report files. The common service owns those integrations and both applications use it.
- Keep mathematical arguments in the method module. Expose effective arguments through `parameters` and increment the method version when its calculation changes so saved numerical fits can be invalidated. Do not add global YAML sigma, prior, or interval settings for one method.
- Use the validated evidence schema without reinterpreting existing fields. A genuinely new evidence requirement needs an explicit shared schema/collector change, not a hidden engine call or an application-specific shortcut.
- Return the standard two-player result, including `players`, `prior`, `interval_scope`, `rating_range`, and `central_interval`, with finite JSON values and coherent availability/count metadata. A side with no usable observations must not receive an invented estimate.
- Declare uncertainty honestly. Point-only methods set intervals and uncertainty to `None`; never borrow another method's posterior. A chosen central display probability is not a calibrated error bound and narrowing it does not improve estimation accuracy.
- Keep experimental method selection separate from changing the production default. Follow the requested scope and explain a promotion when one is requested; do not automatically replace the default because a test-table metric improved.

## Validate and compare

- Put experiments and meaningful regressions in `tests/analysis/`, following [chess-tests](../chess-tests/SKILL.md). Start with synthetic mathematical cases and the standard loader/service, before any requested game collection run.
- Check the derivation numerically and cover relevant invariants: ordering, side exchange where the model is symmetric, account-rating sensitivity, scale consistency, boundaries, forced/empty/flat evidence, and the declared interval semantics.
- Verify that changing reference labels or unrelated games cannot affect a prediction. Check that method/version/argument changes invalidate fits while compatible engine evidence remains reusable.
- Use each game's own validated cached measurements for a rating-only experiment. Read current PGN context when running comparisons; do not compare stale actual ratings or reference headers. Recompute numerical fits when a rerun is requested instead of merely redrawing or relabeling a saved fit.
- Discover requested inputs rather than hardcoding an old game count. Report the evaluated cohort, rating scale, missing-intersection cases, and separate error/ordering/sensitivity results when relevant. Do not impose an old MAE target or hide excluded games.
- Keep source PGNs, non-rating analysis, and cache locations intact during rating-only work. Do not run engines, a coaching agent, or every game merely because the skill was loaded.

## Integrate outputs and explain the method

- Application fitting must refresh the saved fit and rating SVGs even when engine evidence is cached; pure `fit(evidence)` remains free of file output. Replace known generated plots without deleting unrelated user diagrams.
- Keep the shared accuracy curve beside a combined White/Black result panel. Mark intersections, extrapolation, final estimates, and applicable uncertainty parameters distinctly. Show a compact separate prior plot when a prior exists; a point-only result must not be depicted as a posterior.
- Retain consistent axes and declared rating scales across comparable figures. Use current rendering conventions rather than freezing an earlier experiment's display bounds in the algorithm.
- Add the method's module row and interface explanation to [analysis/README.md](../../../analysis/README.md). Keep method documentation there rather than adding a README inside `player_rating/`.
- For a method being documented for adoption, produce a concise standalone technical paper following [chess-docs](../chess-docs/SKILL.md): design reasoning, equations, assumptions, self-contained example, limitations, and explanatory SVGs. Do not use private source links, run instructions, contextless commercial comparisons, or artificial prose line breaks as substitutes for the mathematical explanation.
