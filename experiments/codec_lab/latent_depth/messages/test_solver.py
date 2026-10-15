"""Schedule and soundness boundaries for fixed-input message depth."""
import unittest

from solver import Graph, stages, status


class Messages(unittest.TestCase):
    def test_synchronous_chain_has_no_inplace_shortcut(self):
        graph = Graph(3, (1, 3, 6))
        sequence = stages(graph, 6)
        self.assertEqual(sequence[1].one, (1, 3, 6))
        self.assertEqual(sequence[2].one, (1, 2, 6))
        self.assertEqual(sequence[3].one, (1, 2, 6))
        self.assertEqual(sequence[4].one, (1, 2, 4))
        self.assertEqual([status(graph, s) for s in sequence[:5]], ["UNKNOWN"] * 5)
        self.assertEqual(status(graph, sequence[5]), "FEASIBLE")
        self.assertEqual(sequence[5], sequence[6])
        reordered = Graph(3, tuple(reversed(graph.columns)))
        self.assertEqual([status(reordered, s) for s in stages(reordered, 6)],
                         [status(graph, s) for s in sequence])

    def test_more_rounds_do_not_choose_a_cycle_assignment(self):
        graph = Graph(3, (3, 6, 5))
        sequence = stages(graph, 32)
        self.assertEqual(sequence[0], sequence[-1])
        self.assertEqual(status(graph, sequence[-1]), "UNKNOWN")

    def test_hall_deficit_is_unknown_not_a_feasibility_claim(self):
        graph = Graph(3, (3, 3, 3))
        sequence = stages(graph, 32)
        self.assertEqual(sequence[0], sequence[-1])
        self.assertEqual(status(graph, sequence[-1]), "UNKNOWN")

    def test_local_capacity_contradiction_is_certified(self):
        graph = Graph(2, (1, 1))
        sequence = stages(graph, 2)
        self.assertEqual(status(graph, sequence[1]), "UNKNOWN")
        self.assertEqual(status(graph, sequence[2]), "INFEASIBLE")

    def test_repeating_original_input_cannot_extend_radius(self):
        graph = Graph(3, (1, 3, 6))
        for mode in ["input_only_onepass", "repeat_original_no_feedback", "target_only_no_capacity_messages"]:
            sequence = stages(graph, 32, mode)
            self.assertEqual(sequence[1], sequence[-1])
            self.assertEqual(status(graph, sequence[-1]), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
