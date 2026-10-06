"""Local tool schemas reject malformed requests before chess work executes."""
import unittest

from coach.tools_schema import chess_tool, validate_tool_arguments


class ChessToolTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        @chess_tool
        def explore(ply: int, line: list[str] | None = None) -> str:
            """Explore a legal continuation.

            Args:
                ply: One-based game ply.
                line: Optional UCI continuation.
            """
            self.calls.append((ply, line))
            return 'checked'
        self.tool = explore

    def test_required_and_nullable_arguments_match_callable(self):
        schema = self.tool.input_schema
        self.assertEqual(schema['required'], ['ply'])
        self.assertEqual(schema['properties']['ply']['description'], 'One-based game ply.')
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(self.tool(ply=2), 'checked')
        self.assertEqual(self.tool(ply=3, line=['e7e5']), 'checked')
        self.assertEqual(self.calls, [(2, None), (3, ['e7e5'])])

    def test_malformed_tool_arguments_do_not_execute(self):
        for args in ({}, {'ply': True}, {'ply': '2'}, {'ply': 2, 'unknown': 1},
                     {'ply': 2, 'line': [1]}, {'ply': 2, 'line': 'e7e5'}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.tool(**args)
        with self.assertRaises(ValueError):
            validate_tool_arguments(self.tool, [{'ply': 2}])
        self.assertEqual(self.calls, [])

    def test_declared_numeric_bounds_are_enforced_before_execution(self):
        self.tool.input_schema['properties']['ply'].update(minimum=1, maximum=12)
        for ply in (0, 13):
            with self.subTest(ply=ply), self.assertRaises(ValueError):
                self.tool(ply=ply)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.tool(ply=12), 'checked')

    def test_nonfinite_numbers_are_not_valid_json_tool_arguments(self):
        @chess_tool
        def measure(value: float) -> float:
            self.calls.append(value)
            return value

        for value in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                measure(value=value)
        self.assertEqual(self.calls, [])


if __name__ == '__main__':
    unittest.main()
