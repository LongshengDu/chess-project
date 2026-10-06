"""Check diagnostic maps against their candidate APIs without writing figures."""
import unittest

from tests.analysis.retired_population import REASON as WITHDRAWN_TEST_REASON

from tests.analysis import curve_anchor_candidates, curve_hierarchical_candidates, curve_regularized_candidates
from tests.analysis.intuitive_curve_diagnostics import METHODS, build_figure, prepare_plot
from tests.analysis.rating_evidence_fixture import evidence_fixture


@unittest.skip(WITHDRAWN_TEST_REASON)
class IntuitiveDiagnosticTests(unittest.TestCase):
    def test_displayed_common_maps_equal_candidate_predictions_for_both_players(self):
        evidence = evidence_fixture()
        actual = {'White': 1400., 'Black': 1600.}
        data = prepare_plot(evidence, actual, tuple(METHODS))
        expected = {}
        for module in (curve_anchor_candidates, curve_hierarchical_candidates, curve_regularized_candidates):
            expected.update(module.predict(evidence, actual))
        for mapping in data['maps']:
            for side in ('White', 'Black'):
                self.assertEqual(mapping.point(data['observed'][side]), expected[mapping.method][side])

    def test_fixed_axes_and_missing_crossing_are_explicit(self):
        data = prepare_plot(evidence_fixture(), {'White': 1400., 'Black': 1600.})
        data['observed']['White'] = 100.
        figure = build_figure('synthetic', data)
        try:
            self.assertEqual(figure.axes[0].get_xlim(), (200., 3000.))
            self.assertEqual(figure.axes[0].get_ylim(), (50., 100.))
            self.assertEqual(figure.axes[1].get_xlim(), (50., 100.))
            texts = [cell.get_text().get_text() for axis in figure.axes for table in axis.tables
                     for cell in table.get_celld().values()]
            self.assertIn('No crossing\nabove 0–3200 curve', texts)
        finally:
            figure.clear()


if __name__ == '__main__':
    unittest.main()
