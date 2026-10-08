"""Validation and atomic publication of coaching reports."""
from __future__ import annotations

import re

from .agent_budget import CoachingLimitError
from .settings import CONFIG


def clean_report(answer):
    if not isinstance(answer, str):
        return ''
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.S | re.I).strip()
    fence = re.fullmatch(r'```(?:markdown|md)?\s*\n(.*?)\n```', answer, re.S | re.I)
    return fence[1].strip() if fence else answer


def check_sections(answer):
    # Check actual Markdown headings, allowing numbering, emphasis and normal synonyms.
    headings = [re.sub(r'[^a-z ]', '', h.lower()).strip()
                for h in re.findall(r'^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$', answer, re.M)]
    groups = {
        'performance': ('performance', 'rating', 'elo'),
        'main pattern': ('main pattern', 'recurring', 'key weakness', 'main weakness', 'key lesson'),
        'improvement plan': ('improvement', 'training plan', 'practice plan', 'next step'),
        'exercises': ('exercise', 'practice position', 'training position'),
    }
    # Best/Worst decisions are conditional on the chess evidence. A heading
    # check cannot determine suitability; never force invented praise or errors.
    missing = [name for name, aliases in groups.items()
               if not any(alias in heading for heading in headings for alias in aliases)]
    if missing:
        raise ValueError('Add report sections covering: ' + ', '.join(missing) + '. Equivalent headings are accepted.')


def validate_report(library, answer):
    """Accept reports with locally checked decision coverage and board images."""
    from analysis.game.summary import decision_rows
    decisions = decision_rows(library.analysis, library.side)
    investigated = [row for row in decisions if row['ply'] in library.investigated]
    needed = min(3, len(decisions))
    missing_stages = {row['stage'] for row in decisions} - {row['stage'] for row in investigated}
    if len(investigated) < needed or missing_stages:
        raise ValueError(f'Investigate at least {needed} selected-player decisions with explore_candidate or compare_played_vs_candidate, covering {sorted(missing_stages)}. Assess likely human replies and verify tactical claims before writing.')
    if not isinstance(answer, str) or len(answer) < 250:
        raise ValueError('Return a complete Markdown coaching report.')
    library.diagrams.validate(answer)
    check_sections(answer)
    return True


class ReportGate:
    def __init__(self, library, *, report_name='coaching', max_attempts=CONFIG['COACH']['REPORT_ATTEMPTS'], validator=None, render=None):
        self.library, self.report_name = library, report_name
        self.validator = validator or (lambda answer: validate_report(library, answer))
        self.render = render or (lambda report: report)
        self.attempts = 0
        self.max_attempts = max_attempts
        self.last_error = None

    def validate(self, answer):
        self.attempts += 1
        report = self.render(clean_report(answer))
        if report:
            name = self.report_name + '.draft.md'
            (self.library.directory / name).write_text(report + '\n', encoding='utf-8')
        progress = getattr(self.library, 'progress', None)
        if progress is not None:
            progress.stage('Checking the report against the investigated decisions and required sections.')
        try:
            self.validator(report)
        except ValueError as exc:
            self.last_error = str(exc)
            if progress is not None:
                progress.emit('The draft needs a correction before it can be published.')
                progress.idle()
            raise
        self.last_error = None
        return True

    def check_retry(self):
        if self.last_error and self.attempts >= self.max_attempts:
            raise CoachingLimitError(f'Report could not pass validation after {self.attempts} drafts: {self.last_error} Draft saved separately; no further model calls.')

    def publish(self, answer):
        report = self.render(clean_report(answer))
        self.validator(report)
        name = self.report_name + '.md'
        target = self.library.directory / name
        temporary = target.with_suffix('.md.tmp')
        temporary.write_text(report + '\n', encoding='utf-8')
        temporary.replace(target)
        return report
