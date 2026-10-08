#!/usr/bin/env python3
"""Convert all four online chess rating scales with one global smooth model.

Python 3.10+, standard library only. Codes and full names both work:
lb = Lichess Blitz, lr = Lichess Rapid,
cb = Chess.com Blitz, cr = Chess.com Rapid.

Python:
    convert(1600, "lr", "cr")     # about 1233; emerges from the regional fit
    convert(800, "lb")            # about lr=1046, cb=231, cr=449
    valid_ranges()                # exact inclusive source bounds
    reference_table()             # full LB400..2800 table, rounded to 10

CLI:
    python -m analysis.elo_convert 1600 lr --to cr
    python -m analysis.elo_convert 800 lb --round-to 10
    python -m analysis.elo_convert 1200 cr --json
    python -m analysis.elo_convert --ranges
    python -m analysis.elo_convert --table
    python -m analysis.elo_convert 3200 lb --to cr --extrapolate

Project API:
    scale_from_headers({"Site": "Chess.com", "TimeControl": "600+0"})
    resolve_scale(headers)         # explicit PGN provenance, or an error
    convert(3200, "lb", "cr", extrapolate=True)
    convert_with_metadata(3200, "lb", "cr", extrapolate=True)
    derivative(3200, "cr", extrapolate=True)  # density-change Jacobian

Conversion is strict by default. Explicit extrapolation evaluates the same
analytic curves beyond the fitted LB400..2800 domain, without clipping or
inventing a new tail. Its output is marked as extrapolated in metadata.

MODEL: Unanchored regional-trend fit, revised 2026-10-06.
No finite-rating equality constraints or specially weighted example points.
The user's approximate examples are NOT forced correspondences. In
particular, there is no constraint at LB800 or LR1600, and no special
fit target of CB200, CR400, CR1200, or CR1235.

For B = Lichess Blitz and s in lr/cb/cr, the single analytic equation is
    f_s(B) = F_s + a_s*150*softplus((B-2000)/150)
             + sum(A_sj*sigmoid((B-c_j)/100), j=1..16),
    c_j = 100 + 150*(j-1).
Here softplus(z)=log(1+exp(z)) and sigmoid(z)=1/(1+exp(-z)).
Positive a_s and nonnegative A_sj guarantee strict monotonicity for every
finite real B. The function is infinitely differentiable and approaches
its lower bound F_s as B tends to minus infinity. No floor value is fitted
as a finite point. The structural high-asymptote condition
    sum(A_sj)=b_s-F_s+2000*a_s
makes f_s(B)-(a_s*B+b_s) tend to zero at high input. This constrains an
entire limiting line, not the value at any finite rating.

The preferred references are our earlier low-rating trend over LB400..1000,
ChessGoals in the middle, and ChessRatings.org's LB-input lines at the high
end. Our earlier low curve includes manual adjustments, so it receives
less weight. It is a broad prior, not a table to reproduce exactly.
Old values at or within one point of a clipped floor are excluded.
The fitting process uses constrained weighted least squares, amplitude
ridge 0.0001, and a squared-second-derivative penalty of strength 60.
Selected settings: centers spaced 150, sigmoid width 100, softplus width150.
The broad basis was selected for smoother slopes and regional agreement,
not proximity to the user's illustrative numbers. Its moderate curvature
penalty discourages the old model's artificial narrow slope changes.

Reproducible objective, with fitted amplitudes A in rating points:
    sum(w_i*((f(B_i)-y_i)/100)**2)
    + 0.0001*sum((A_j/100)**2)
    + sum((60*f''(r_k))**2).
Reference weights and numerical grids:
    low: 0.5*sigmoid((980-B)/70), B400..1000 by10, floor cases excluded;
    middle: sigmoid((B-1000)/80)*sigmoid((1950-B)/110),
            92 uniformly spaced points from1090.616977995768 to2000;
    high: sigmoid((B-2010)/100), B2000..2800 by10;
    curvature: r_k=400..2800 by10.
These near-uniform curve samples approximate a reference-curve loss.
They are not independent player observations or measured confidence weights.
No runtime segments, interpolation knots, clipping, or source switches are
used. Stable numerical branches implement the same analytic equation.

Sources retrieved 2026-10-06:
    https://chessgoals.com/rating-comparison/ (Updated July2026)
    https://chessratings.org/
    https://chessratings.org/assets/index-DicX3iC2.js
The CG targets use continuous inversion of its CB-based cubic equations,
only within the covered LB range. No CG data are invented below LB1090.617.
The high reference is the site's three LB-input lines; other site pairwise
regressions need not be consistent with this common-coordinate conversion.

All 12 directional conversions use the SAME LB coordinate, fitted on400..2800:
    source rating -> inverse source curve -> target curve.
Thus round trips and conversion paths agree to floating-point precision.
Supported source intervals are the images of that reference domain, not
entire platform ranges. valid_ranges() gives their exact inclusive bounds.
Integer inputs: LB400..2800, LR645..2839, CB104..2814, CR146..2929.
For example, CB100 is below this model's CB minimum (about103.6768).
Table entries are rounded to10; rounded endpoints can fall outside the
exact input ranges. Inversion always uses full-precision curves.

These are approximate user-directed regional scale correspondences.
Reference-curve agreement and numerical precision are not estimates of
prediction accuracy for individual players. The old low prior and chosen
high reference remain modeling assumptions requiring separate validation.
"""

import argparse
from collections.abc import Mapping
import json
import math
import re
import sys


NAMES = {
    "lb": "Lichess Blitz",
    "lr": "Lichess Rapid",
    "cb": "Chess.com Blitz",
    "cr": "Chess.com Rapid",
}
SCALE_IDENTIFIERS = {
    'lb': 'lichess_blitz',
    'lr': 'lichess_rapid',
    'cb': 'chess_com_blitz',
    'cr': 'chess_com_rapid',
}
LB_MIN, LB_MAX = 400.0, 2800.0
MODEL_VERSION = "2026-10-06-unanchored"
_EPS = 1e-9
_CENTERS = tuple(float(j) for j in range(100, 2401, 150))
_WIDTH = 100.0
_SOFTPLUS_CENTER = 2000.0
_SOFTPLUS_WIDTH = 150.0

# Full-precision fitted coefficients, generated from the selected model.
# Each entry contains: structural floor, high slope/intercept, amplitudes.
_CURVES = {'lr': {'floor': 400.0,
        'slope': 0.9237272889380987,
        'intercept': 252.46563024413263,
        'amplitudes': (0.0,
                       129.7372147255868,
                       210.87135021924811,
                       124.95707825707427,
                       165.71035408539748,
                       150.59045253466343,
                       85.51674443383112,
                       148.80218027037247,
                       163.986348588047,
                       131.5504328239941,
                       127.60258506171249,
                       118.89672086665418,
                       59.669722013349556,
                       51.538750726531,
                       17.342658845006973,
                       13.147614668860928)},
 'cb': {'floor': 100.0,
        'slope': 1.3372721097513234,
        'intercept': -930.4533289873989,
        'amplitudes': (0.0,
                       0.0,
                       0.0,
                       0.0,
                       0.0,
                       306.45606267182495,
                       66.25019147248628,
                       178.53932328795824,
                       282.71486138957584,
                       221.44715441234553,
                       182.56661352377995,
                       350.71065858918735,
                       1.4694276554718638e-12,
                       1.141827127341823e-12,
                       18.041781841753007,
                       37.36424332648343)},
 'cr': {'floor': 100.0,
        'slope': 1.3104810776515827,
        'intercept': -740.5926519828513,
        'amplitudes': (6.559540065215127e-12,
                       3.223789884078529e-12,
                       4.780276740812086,
                       211.96222296773104,
                       1.2219918376347571e-12,
                       315.407967642744,
                       194.1581039561952,
                       139.50540698275563,
                       244.19500966657702,
                       172.6009846830089,
                       179.2335245597942,
                       180.20747555723386,
                       45.376522037974226,
                       23.42196355808736,
                       55.55439652465099,
                       13.965648442738999)}}


def normalize_scale(name: str) -> str:
    """Return a supported scale code from its code or case-insensitive name."""
    if isinstance(name, str):
        key = "".join(c for c in name.lower() if c.isalnum())
        for code, label in NAMES.items():
            if key in (code, "".join(c for c in label.lower() if c.isalnum())):
                return code
    raise ValueError(f"Unknown rating system {name!r}; use lb, lr, cb, or cr.")


class UnsupportedRatingScale(ValueError):
    """A PGN does not identify one of the four modeled rating populations."""

    def __init__(self, reason: str, message: str, context: dict):
        super().__init__(message)
        self.reason = reason
        self.context = dict(context)


def resolve_scale(headers: Mapping, override: str | None = None) -> dict:
    """Resolve rating population from Site and TimeControl, never Link.

    PGN numeric time controls are seconds, optionally plus increment seconds.
    The platform category uses initial+40*increment. Words blitz/rapid are
    also accepted. Unsupported categories, ambiguous/missing sites, and
    unknown/multiperiod controls raise UnsupportedRatingScale explicitly.
    An explicit supported override is authoritative and recorded as such.

    Chess.com thresholds and its40-move increment convention are documented at
    https://support.chess.com/en/articles/8705367-why-are-there-different-ratings-in-live-chess
    Lichess categories use the supplied30/180/480/1500-second boundaries.
    """
    if not isinstance(headers, Mapping):
        raise TypeError('PGN headers must be a mapping.')
    context = {'site': headers.get('Site'), 'time_control': headers.get('TimeControl'),
               'platform': None, 'time_class': None, 'estimated_seconds': None}
    if override is not None:
        scale = normalize_scale(override)
        return {**context, 'scale': scale, 'name': NAMES[scale], 'source': 'override',
                'platform': 'lichess' if scale[0] == 'l' else 'chess.com',
                'time_class': 'blitz' if scale[1] == 'b' else 'rapid'}
    site = str(context['site'] or '').casefold()
    platforms = [platform for token, platform in (('lichess.org', 'lichess'), ('chess.com', 'chess.com'))
                 if token in site]
    if len(platforms) != 1:
        reason = 'ambiguous_site' if platforms else 'unknown_site'
        raise UnsupportedRatingScale(reason, 'Site must identify exactly one of lichess.org or Chess.com.', context)
    platform = context['platform'] = platforms[0]
    control = str(context['time_control'] or '').strip().casefold()
    if not control or control in ('?', '-'):
        raise UnsupportedRatingScale('missing_time_control', 'TimeControl is missing or unknown.', context)
    words = {'ultrabullet', 'bullet', 'blitz', 'rapid', 'classical', 'daily', 'correspondence'}
    if control in words:
        category = control
    else:
        numeric = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:\+\s*(\d+(?:\.\d+)?))?', control)
        if numeric is None:
            raise UnsupportedRatingScale('unsupported_time_control', 'TimeControl must be PGN seconds[+increment] or a named category; multiperiod controls are unsupported.', context)
        initial = float(numeric.group(1))
        increment = float(numeric.group(2) or 0.)
        duration = initial+40.*increment
        if not math.isfinite(duration) or duration <= 0:
            raise UnsupportedRatingScale('unsupported_time_control', 'TimeControl must specify a positive finite duration.', context)
        context['estimated_seconds'] = duration
        if platform == 'lichess':
            category = next((name for upper, name in ((30, 'ultrabullet'), (180, 'bullet'),
                                                       (480, 'blitz'), (1500, 'rapid')) if duration < upper), 'classical')
        else:
            category = 'bullet' if duration < 180 else 'blitz' if duration < 600 else 'rapid'
    context['time_class'] = category
    if category not in ('blitz', 'rapid'):
        raise UnsupportedRatingScale('unsupported_time_class', f'{platform} {category} has no supported rating conversion; only Blitz and Rapid are modeled.', context)
    scale = ('l' if platform == 'lichess' else 'c')+category[0]
    return {**context, 'scale': scale, 'name': NAMES[scale], 'source': 'pgn_headers'}


def scale_from_headers(headers: Mapping) -> str:
    """Return the PGN's supported scale; never silently assume a population."""
    return resolve_scale(headers)['scale']


def _sigmoid(z: float) -> float:
    # Both branches evaluate the same analytic function without overflow.
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def _forward(lb: float, scale: str) -> float:
    if scale == "lb":
        return lb
    p = _CURVES[scale]
    baseline = p["floor"] + p["slope"] * _SOFTPLUS_WIDTH * _softplus(
        (lb - _SOFTPLUS_CENTER) / _SOFTPLUS_WIDTH
    )
    return baseline + math.fsum(
        amplitude * _sigmoid((lb - center) / _WIDTH)
        for amplitude, center in zip(p["amplitudes"], _CENTERS)
    )


def valid_ranges() -> dict[str, tuple[float, float]]:
    """Return inclusive, unrounded model bounds for all four source scales."""
    return {s: (_forward(LB_MIN, s), _forward(LB_MAX, s)) for s in NAMES}


def _checked_rating(rating: float, source: str, *, extrapolate: bool = False) -> float:
    if type(extrapolate) is not bool:
        raise ValueError('extrapolate must be explicitly True or False.')
    try:
        if isinstance(rating, bool):
            raise ValueError
        value = float(rating)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Rating must be a finite number.") from exc
    if not math.isfinite(value):
        raise ValueError("Rating must be a finite number.")
    if extrapolate:
        if source != 'lb' and value <= _CURVES[source]['floor']:
            raise ValueError(f'{NAMES[source]} input must exceed the analytic lower asymptote {_CURVES[source]["floor"]:g}; no finite Lichess Blitz rating maps to that floor or below.')
        return value
    low, high = valid_ranges()[source]
    if value < low - _EPS or value > high + _EPS:
        raise ValueError(
            f"{NAMES[source]} input {value:g} is outside the model's inclusive "
            f"range [{low:.15g}, {high:.15g}], corresponding to "
            "Lichess Blitz 400..2800. No extrapolation is performed."
        )
    # Absorb only floating-point roundoff at a supported endpoint.
    return min(high, max(low, value))


def _to_lb(value: float, source: str, *, extrapolate: bool = False) -> float:
    if source == "lb":
        return value
    low, high = LB_MIN, LB_MAX
    if value == _forward(low, source):
        return low
    if value == _forward(high, source):
        return high
    if extrapolate:
        span = LB_MAX-LB_MIN
        while _forward(low, source) > value:
            high, low = low, low-span
            span *= 2.
        span = LB_MAX-LB_MIN
        while _forward(high, source) < value:
            low, high = high, min(sys.float_info.max, high+span)
            span *= 2.
            if high == sys.float_info.max and _forward(high, source) < value:
                raise ValueError('Rating inversion exceeds finite floating-point coordinates.')
    for _ in range(128):
        midpoint = low+(high-low)/2.0
        if midpoint in (low, high):
            break
        if _forward(midpoint, source) < value:
            low = midpoint
        else:
            high = midpoint
    return low+(high-low)/2.0


def convert(rating: float, source: str, target: str = "all", *, extrapolate: bool = False) -> float | dict[str, float]:
    """Convert supported rating scales; 'all' returns all four named code keys.

    Names are case-insensitive. Full names, spaces and punctuation work.
    Invalid systems and nonfinite inputs raise ValueError. Out-of-range inputs
    require extrapolate=True, which uses the same analytic equation outside its
    fitted domain. Full precision is retained; no rating/floor clipping occurs.
    """
    return convert_with_metadata(rating, source, target, extrapolate=extrapolate)['value']


def convert_with_metadata(rating: float, source: str, target: str = 'all', *, extrapolate: bool = False) -> dict:
    """Convert with explicit fitted-domain status and unchanged model provenance."""
    source = normalize_scale(source)
    all_targets = isinstance(target, str) and target.strip().lower() == "all"
    targets = list(NAMES) if all_targets else [normalize_scale(target)]
    value = _checked_rating(rating, source, extrapolate=extrapolate)
    lb = _to_lb(value, source, extrapolate=extrapolate)
    results = {s: value if s == source else _forward(lb, s) for s in targets}
    if not all(math.isfinite(result) for result in results.values()):
        raise ValueError('Converted rating exceeds finite floating-point values.')
    inside = LB_MIN-_EPS <= lb <= LB_MAX+_EPS
    return {'value': results if all_targets else results[targets[0]],
            'source': source, 'target': 'all' if all_targets else targets[0],
            'canonical_lb': lb, 'in_model_range': inside, 'extrapolated': not inside,
            'source_range': list(valid_ranges()[source]), 'canonical_range': [LB_MIN, LB_MAX],
            'model_version': MODEL_VERSION,
            'provenance': 'User-supplied unanchored regional-trend coefficients; low-rating prior, ChessGoals middle region and ChessRatings.org high region; approximate population-scale correspondence, not individual prediction accuracy.',
            'extrapolation': 'same_analytic_formula' if not inside else None}


def derivative(lb: float, target: str, *, extrapolate: bool = False) -> float:
    """Return d(target rating)/d(Lichess Blitz), for pushforward densities.

    The same strict/explicit-extrapolation convention as convert applies.
    """
    target = normalize_scale(target)
    lb = _checked_rating(lb, 'lb', extrapolate=extrapolate)
    if target == 'lb':
        return 1.
    parameters = _CURVES[target]
    baseline = parameters['slope']*_sigmoid((lb-_SOFTPLUS_CENTER)/_SOFTPLUS_WIDTH)
    return baseline+math.fsum(amplitude*_sigmoid((lb-center)/_WIDTH)*_sigmoid(-(lb-center)/_WIDTH)/_WIDTH
                              for amplitude, center in zip(parameters['amplitudes'], _CENTERS))


def _rounded(value: float, step: float) -> float:
    scaled = value / step
    if not math.isfinite(scaled):
        return value  # Finer than representable floating-point precision.
    return step * math.floor(scaled + 0.5)


def reference_table() -> list[tuple[int, ...]]:
    """Full LB400..2800 table by100; columns LB,LR,CB,CR, rounded to10."""
    return [
        tuple(int(_rounded(_forward(float(lb), s), 10.0)) for s in NAMES)
        for lb in range(int(LB_MIN), int(LB_MAX) + 1, 100)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rating", type=float, nargs="?", help="Source rating")
    parser.add_argument("source", nargs="?", help="lb, lr, cb, cr, or full name")
    parser.add_argument("--to", default="all", help="Target system (default: all)")
    parser.add_argument("--round-to", type=float, metavar="STEP",
                        help="Round displayed results, e.g. 1 or10; ties upward")
    parser.add_argument("--json", action="store_true", help="Print JSON")
    parser.add_argument("--extrapolate", action="store_true", help="Explicitly extend the same analytic model beyond its fitted range")
    parser.add_argument("--metadata", action="store_true", help="Print conversion result with model and extrapolation metadata as JSON")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--table", action="store_true", help="Show the complete table")
    modes.add_argument("--ranges", action="store_true", help="Show supported ranges")
    args = parser.parse_args(argv)
    if args.round_to is not None and (
        not math.isfinite(args.round_to) or args.round_to <= 0.0
    ):
        parser.error("--round-to must be a positive finite number.")
    if args.table or args.ranges:
        if args.rating is not None or args.source is not None:
            parser.error("Use --table or --ranges without a rating or source.")
        if args.to != "all" or args.round_to is not None or args.extrapolate or args.metadata:
            parser.error("--to, --round-to, --extrapolate and --metadata apply only to conversions.")
        if args.table:
            rows = reference_table()
            if args.json:
                print(json.dumps([dict(zip(NAMES, row)) for row in rows], indent=2))
            else:
                print(" | ".join(f"{name:>16}" for name in NAMES.values()))
                for row in rows:
                    print(" | ".join(f"{value:16d}" for value in row))
                print("Rounded to10; inversion uses the unrounded curves.")
        elif args.json:
            print(json.dumps(valid_ranges(), indent=2))
        else:
            for s, (low, high) in valid_ranges().items():
                print(f"{s}  {NAMES[s]:16}  {low:.12f} .. {high:.12f}")
            print("Inclusive model ranges. Use --ranges --json for full precision.")
        return 0
    if args.rating is None and args.source is None:
        parser.print_help()
        return 0
    if args.rating is None or args.source is None:
        parser.error("Provide rating and source, e.g. 1600 lr --to cr.")
    try:
        metadata = convert_with_metadata(args.rating, args.source, args.to, extrapolate=args.extrapolate)
        result = metadata['value']
    except ValueError as exc:
        parser.error(str(exc))

    def displayed(value):
        if isinstance(value, dict):
            return {s: displayed(v) for s, v in value.items()}
        return value if args.round_to is None else _rounded(value, args.round_to)

    result = displayed(result)
    if args.metadata:
        print(json.dumps({**metadata, 'value': result}, indent=2, allow_nan=False))
    elif args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    elif isinstance(result, dict):
        for s, value in result.items():
            print(f"{NAMES[s]:16}: {value:.6f}")
    else:
        print(f"{result:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
