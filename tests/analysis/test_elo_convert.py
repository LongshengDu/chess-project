"""Scale inference, unchanged coefficients, analytic conversion and Jacobians."""
from contextlib import redirect_stdout
from hashlib import sha256
from io import StringIO
import json
import unittest

from analysis import elo_convert as converter


class ConversionTests(unittest.TestCase):
    def test_supplied_coefficients_are_unchanged(self):
        self.assertEqual(sha256(json.dumps(converter._CURVES, sort_keys=True).encode()).hexdigest(),
                         '1a1a1c40fa6acadd01eac145b627ee7ca270545c9618b32dd25633dfcd4b91ab')

    def test_codes_and_names_are_normalized(self):
        for code, name in converter.NAMES.items():
            for value in (code.upper(), name.upper(), name.replace(' ', '_'), ' '+name+' '):
                self.assertEqual(converter.normalize_scale(value), code)
        for value in (None, '', 'Lichess Bullet', 'rapid', True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                converter.normalize_scale(value)

    def test_all_paths_round_trip_including_strict_boundaries(self):
        for lb in (400., 800., 1600., 2600., 2800.):
            values = converter.convert(lb, 'lb')
            for source, value in values.items():
                self.assertAlmostEqual(converter.convert(value, source, 'lb'), lb, places=8)
                for target, expected in values.items():
                    with self.subTest(lb=lb, source=source, target=target):
                        self.assertAlmostEqual(converter.convert(value, source, target), expected, places=8)

    def test_each_scale_is_strict_by_default(self):
        for source, (lower, upper) in converter.valid_ranges().items():
            for value in (lower-1e-5, upper+1e-5):
                with self.subTest(source=source, value=value), self.assertRaises(ValueError):
                    converter.convert(value, source)
            self.assertFalse(converter.convert_with_metadata(lower, source)['extrapolated'])
            self.assertFalse(converter.convert_with_metadata(upper, source)['extrapolated'])

    def test_explicit_extrapolation_is_analytic_and_round_trips_canonical_support(self):
        for lb in (0., 200., 400., 2800., 3000., 3200.):
            values = converter.convert(lb, 'lb', extrapolate=True)
            for source, value in values.items():
                for target, expected in values.items():
                    with self.subTest(lb=lb, source=source, target=target):
                        metadata = converter.convert_with_metadata(value, source, target, extrapolate=True)
                        self.assertAlmostEqual(metadata['value'], expected, places=7)
                        self.assertAlmostEqual(metadata['canonical_lb'], lb, places=7)
                        self.assertEqual(metadata['extrapolated'], lb < 400 or lb > 2800)
                        self.assertEqual(metadata['in_model_range'], 400 <= lb <= 2800)
                        self.assertEqual(metadata['model_version'], converter.MODEL_VERSION)
        self.assertGreater(converter.convert(3200., 'lb', 'cr', extrapolate=True),
                           converter.convert(2800., 'lb', 'cr'))

    def test_extrapolation_rejects_unreachable_floors_and_nonfinite_values(self):
        for source, parameters in converter._CURVES.items():
            for value in (parameters['floor']-1, parameters['floor']):
                with self.subTest(source=source, value=value), self.assertRaisesRegex(ValueError, 'lower asymptote'):
                    converter.convert(value, source, extrapolate=True)
        for value in (True, None, float('nan'), float('inf'), 'not a rating'):
            for extrapolate in (False, True):
                with self.subTest(value=value, extrapolate=extrapolate), self.assertRaises(ValueError):
                    converter.convert(value, 'lb', extrapolate=extrapolate)
        with self.assertRaises(ValueError):
            converter.convert(1600., 'lb', extrapolate=1)

    def test_derivatives_match_finite_differences_on_extended_support(self):
        for lb in (0., 400., 1600., 2800., 3200.):
            for target in converter.NAMES:
                step = .001
                numerical = (converter.convert(lb+step, 'lb', target, extrapolate=True)
                             -converter.convert(lb-step, 'lb', target, extrapolate=True))/(2*step)
                slope = converter.derivative(lb, target, extrapolate=True)
                self.assertGreater(slope, 0.)
                self.assertAlmostEqual(slope, numerical, places=7)
        with self.assertRaises(ValueError):
            converter.derivative(3200., 'cr')

    def test_cli_supports_explicit_extrapolation_metadata(self):
        stream = StringIO()
        with redirect_stdout(stream):
            self.assertEqual(converter.main(['3200', 'lb', '--to', 'cr', '--extrapolate', '--metadata']), 0)
        result = json.loads(stream.getvalue())
        self.assertTrue(result['extrapolated'])
        self.assertEqual(result['canonical_lb'], 3200.)
        self.assertEqual(result['value'], converter.convert(3200., 'lb', 'cr', extrapolate=True))


class ScaleInferenceTests(unittest.TestCase):
    def test_site_is_case_insensitive_and_authoritative_over_link(self):
        for site, link, expected in (('https://LICHESS.ORG/abc', 'https://chess.com/game', 'lb'),
                                     ('CHESS.COM', 'https://lichess.org/game', 'cb')):
            headers = {'Site': site, 'TimeControl': 'BLITZ', 'Link': link}
            self.assertEqual(converter.scale_from_headers(headers), expected)
            self.assertEqual(converter.resolve_scale(headers)['source'], 'pgn_headers')
        with self.assertRaises(converter.UnsupportedRatingScale) as error:
            converter.scale_from_headers({'Site': '?', 'Link': 'https://lichess.org/game', 'TimeControl': 'blitz'})
        self.assertEqual(error.exception.reason, 'unknown_site')

    def test_lichess_all_supplied_category_boundaries(self):
        cases = ((29, 'ultrabullet'), (30, 'bullet'), (179, 'bullet'), (180, 'blitz'),
                 (479, 'blitz'), (480, 'rapid'), (1499, 'rapid'), (1500, 'classical'))
        for seconds, category in cases:
            headers = {'Site': 'lichess.org', 'TimeControl': str(seconds)}
            if category in ('blitz', 'rapid'):
                result = converter.resolve_scale(headers)
                self.assertEqual(result['scale'], 'l'+category[0])
                self.assertEqual(result['estimated_seconds'], seconds)
            else:
                with self.assertRaises(converter.UnsupportedRatingScale) as error:
                    converter.resolve_scale(headers)
                self.assertEqual(error.exception.reason, 'unsupported_time_class')
                self.assertEqual(error.exception.context['time_class'], category)

    def test_chesscom_categories_include_increment_and_do_not_invent_classical(self):
        for control, expected, seconds in (('180', 'cb', 180), ('599+0', 'cb', 599),
                                           ('600+0', 'cr', 600), ('120+12', 'cr', 600),
                                           ('300+5', 'cb', 500), ('3600', 'cr', 3600)):
            result = converter.resolve_scale({'Site': 'Chess.com', 'TimeControl': control})
            self.assertEqual(result['scale'], expected)
            self.assertEqual(result['estimated_seconds'], seconds)
        with self.assertRaises(converter.UnsupportedRatingScale) as error:
            converter.resolve_scale({'Site': 'Chess.com', 'TimeControl': '179'})
        self.assertEqual(error.exception.context['time_class'], 'bullet')
        self.assertEqual(converter.scale_from_headers({'Site': 'lichess.org', 'TimeControl': '300+5'}), 'lr')

    def test_named_time_controls_and_explicit_override(self):
        for site, prefix in (('lichess.org', 'l'), ('Chess.com', 'c')):
            for category in ('blitz', 'rapid'):
                self.assertEqual(converter.scale_from_headers({'Site': site, 'TimeControl': category.upper()}), prefix+category[0])
        result = converter.resolve_scale({'Site': '?', 'TimeControl': 'bullet'}, override='Lichess Blitz')
        self.assertEqual(result['scale'], 'lb')
        self.assertEqual(result['source'], 'override')
        self.assertEqual(result['time_control'], 'bullet')

    def test_unsupported_or_ambiguous_metadata_fails_explicitly(self):
        cases = (
            ({'Site': 'lichess.org and Chess.com', 'TimeControl': '300'}, 'ambiguous_site'),
            ({'Site': 'example.org', 'TimeControl': 'rapid'}, 'unknown_site'),
            ({'Site': 'Chess.com'}, 'missing_time_control'),
            ({'Site': 'Chess.com', 'TimeControl': '?'}, 'missing_time_control'),
            ({'Site': 'lichess.org', 'TimeControl': '40/7200:3600'}, 'unsupported_time_control'),
            ({'Site': 'Chess.com', 'TimeControl': '1/86400'}, 'unsupported_time_control'),
            ({'Site': 'Chess.com', 'TimeControl': '3|2'}, 'unsupported_time_control'),
            ({'Site': 'Chess.com', 'TimeControl': '0+0'}, 'unsupported_time_control'),
            ({'Site': 'lichess.org', 'TimeControl': 'classical'}, 'unsupported_time_class'),
            ({'Site': 'Chess.com', 'TimeControl': 'bullet'}, 'unsupported_time_class'),
            ({'Site': 'Chess.com', 'TimeControl': 'daily'}, 'unsupported_time_class'),
        )
        for headers, reason in cases:
            with self.subTest(headers=headers), self.assertRaises(converter.UnsupportedRatingScale) as error:
                converter.resolve_scale(headers)
            self.assertEqual(error.exception.reason, reason)


if __name__ == '__main__':
    unittest.main()
