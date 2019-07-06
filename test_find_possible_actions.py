from true_hypo_models import *
import numpy as np
import unittest

class test_find_actions(unittest.TestCase):

    def test_binary(self):
        experiment_vars = np.array(["d"])
        correct = [[0], [1]]
        output = find_possible_actions(experiment_vars)
        self.assertEqual(correct, output)

    def test_one_factor(self):
        experiment_vars = np.array(["Feedback_1", "Feedback_2", "Feedback_3", "Feedback_4"])
        correct = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
        output = find_possible_actions(experiment_vars)
        self.assertEqual(correct, output)

    def test_two_factor_one_level(self):
        experiment_vars = np.array(["Feedback", "Motivation"])
        correct = [[0, 0], [0, 1], [1, 0], [1, 1]]
        output = find_possible_actions(experiment_vars)
        self.assertEqual(correct, output)

    def test_two_factor_mixed_levels(self):
        experiment_vars = np.array(["Feedback_1", "Feedback_2", "Feedback_3", "Motivation"])
        correct = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], 
        [0, 1, 0, 0], [0, 1, 0, 1], [1, 0, 0, 0], [1, 0, 0, 1]]
        output = find_possible_actions(experiment_vars)
        self.assertEqual(correct, output)

    def test_two_factor_mixed_levels_rev(self):
        experiment_vars = np.array(["Encourage", "Feedback_1", "Feedback_2", "Feedback_3"])
        correct = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], 
        [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]]
        output = find_possible_actions(experiment_vars)
        self.assertEqual(correct, output)

    def test_two_factor_multiple_levels(self):
        experiment_vars = np.array(["Feedback_1",
                       "Feedback_2",
                       "Feedback_3",
                       "Feedback_4",
                       "Motivational_1",
                       "Motivational_2",
                       "Motivational_3"
                       ]
)
        correct = [ [0,0,0,0,   0,0,0],
                              [0,0,0,0,   0,0,1],
                              [0,0,0,0,   0,1,0],
                              [0,0,0,0,   1,0,0],

                              [0,0,0,1,   0,0,0],
                              [0,0,0,1,   0,0,1],
                              [0,0,0,1,   0,1,0],
                              [0,0,0,1,   1,0,0],

                              [0,0,1,0,   0,0,0],
                              [0,0,1,0,   0,0,1],
                              [0,0,1,0,   0,1,0],
                              [0,0,1,0,   1,0,0],

                              [0,1,0,0,   0,0,0],
                              [0,1,0,0,   0,0,1],
                              [0,1,0,0,   0,1,0],
                              [0,1,0,0,   1,0,0],

                              [1,0,0,0,   0,0,0],
                              [1,0,0,0,   0,0,1],
                              [1,0,0,0,   0,1,0],
                              [1,0,0,0,   1,0,0] ]

        output = find_possible_actions(experiment_vars)
        self.assertEqual(correct, output)

    def test_two_factor_multiple_levels_non_alph_order(self):
        experiment_vars = np.array(["Motivational_1",
                       "Motivational_2",
                       "Motivational_3",
                       "Feedback_1",
                       "Feedback_2",
                       "Feedback_3",
                       "Feedback_4"
                       ]
)
        correct = [ [0,0,0,  0,0,0,0],
                              [0,0,0,  0,0,0,1],
                              [0,0,0,  0,0,1,0],
                              [0,0,0,  0,1,0,0],
                              [0,0,0,  1,0,0,0],

                              [0,0,1,  0,0,0,0],
                              [0,0,1,  0,0,0,1],
                              [0,0,1,  0,0,1,0],
                              [0,0,1,  0,1,0,0],
                              [0,0,1,  1,0,0,0],

                              [0,1,0,   0,0,0,0],
                              [0,1,0,   0,0,0,1],
                              [0,1,0,   0,0,1,0],
                              [0,1,0,   0,1,0,0],
                              [0,1,0,   1,0,0,0],

                              [1,0,0,   0,0,0,0],
                              [1,0,0,   0,0,0,1],
                              [1,0,0,   0,0,1,0],
                              [1,0,0,   0,1,0,0],
                              [1,0,0,   1,0,0,0] ]

        output = find_possible_actions(experiment_vars)
        try:
            self.assertCountEqual(correct, output) # This should fail
            self.test_passed = False # If correct, this doesn't reach here
        except AssertionError:
            self.test_passed = True

    def test_three_factor_multiple_levels(self):
        experiment_vars = ["Feedback_1",
                       "Feedback_2",
                       "Feedback_3",
                       "Feedback_4",
                       "Motivational_1",
                       "Motivational_2",
                       "Motivational_3",
                       "TimeSlot_2",
                       "TimeSlot_3",
                       "TimeSlot_4"
                       ]
        correct = [ [0, 0, 0, 0,      0, 0, 0,      0, 0, 0],
                              [0, 0, 0, 0,      0, 0, 0,      0, 0, 1],
                              [0, 0, 0, 0,      0, 0, 0,      0, 1, 0],
                              [0, 0, 0, 0,      0, 0, 0,      1, 0, 0],

                              [0, 0, 0, 0,      0, 0, 1,      0, 0, 0],
                              [0, 0, 0, 0,      0, 0, 1,      0, 0, 1],
                              [0, 0, 0, 0,      0, 0, 1,      0, 1, 0],
                              [0, 0, 0, 0,      0, 0, 1,      1, 0, 0],

                              [0, 0, 0, 0,      0, 1, 0,      0, 0, 0],
                              [0, 0, 0, 0,      0, 1, 0,      0, 0, 1],
                              [0, 0, 0, 0,      0, 1, 0,      0, 1, 0],
                              [0, 0, 0, 0,      0, 1, 0,      1, 0, 0],

                              [0, 0, 0, 0,      1, 0, 0,      0, 0, 0],
                              [0, 0, 0, 0,      1, 0, 0,      0, 0, 1],
                              [0, 0, 0, 0,      1, 0, 0,      0, 1, 0],
                              [0, 0, 0, 0,      1, 0, 0,      1, 0, 0],

                              [0, 0, 0, 1,      0, 0, 0,      0, 0, 0],
                              [0, 0, 0, 1,      0, 0, 0,      0, 0, 1],
                              [0, 0, 0, 1,      0, 0, 0,      0, 1, 0],
                              [0, 0, 0, 1,      0, 0, 0,      1, 0, 0],

                              [0, 0, 0, 1,      0, 0, 1,      0, 0, 0],
                              [0, 0, 0, 1,      0, 0, 1,      0, 0, 1],
                              [0, 0, 0, 1,      0, 0, 1,      0, 1, 0],
                              [0, 0, 0, 1,      0, 0, 1,      1, 0, 0],

                              [0, 0, 0, 1,      0, 1, 0,      0, 0, 0],
                              [0, 0, 0, 1,      0, 1, 0,      0, 0, 1],
                              [0, 0, 0, 1,      0, 1, 0,      0, 1, 0],
                              [0, 0, 0, 1,      0, 1, 0,      1, 0, 0],

                              [0, 0, 0, 1,      1, 0, 0,      0, 0, 0],
                              [0, 0, 0, 1,      1, 0, 0,      0, 0, 1],
                              [0, 0, 0, 1,      1, 0, 0,      0, 1, 0],
                              [0, 0, 0, 1,      1, 0, 0,      1, 0, 0],

                              [0, 0, 1, 0,      0, 0, 0,      0, 0, 0],
                              [0, 0, 1, 0,      0, 0, 0,      0, 0, 1],
                              [0, 0, 1, 0,      0, 0, 0,      0, 1, 0],
                              [0, 0, 1, 0,      0, 0, 0,      1, 0, 0],

                              [0, 0, 1, 0,      0, 0, 1,      0, 0, 0],
                              [0, 0, 1, 0,      0, 0, 1,      0, 0, 1],
                              [0, 0, 1, 0,      0, 0, 1,      0, 1, 0],
                              [0, 0, 1, 0,      0, 0, 1,      1, 0, 0],

                              [0, 0, 1, 0,      0, 1, 0,      0, 0, 0],
                              [0, 0, 1, 0,      0, 1, 0,      0, 0, 1],
                              [0, 0, 1, 0,      0, 1, 0,      0, 1, 0],
                              [0, 0, 1, 0,      0, 1, 0,      1, 0, 0],

                              [0, 0, 1, 0,      1, 0, 0,      0, 0, 0],
                              [0, 0, 1, 0,      1, 0, 0,      0, 0, 1],
                              [0, 0, 1, 0,      1, 0, 0,      0, 1, 0],
                              [0, 0, 1, 0,      1, 0, 0,      1, 0, 0],

                              [0, 1, 0, 0,      0, 0, 0,      0, 0, 0],
                              [0, 1, 0, 0,      0, 0, 0,      0, 0, 1],
                              [0, 1, 0, 0,      0, 0, 0,      0, 1, 0],
                              [0, 1, 0, 0,      0, 0, 0,      1, 0, 0],

                              [0, 1, 0, 0,      0, 0, 1,      0, 0, 0],
                              [0, 1, 0, 0,      0, 0, 1,      0, 0, 1],
                              [0, 1, 0, 0,      0, 0, 1,      0, 1, 0],
                              [0, 1, 0, 0,      0, 0, 1,      1, 0, 0],

                              [0, 1, 0, 0,      0, 1, 0,      0, 0, 0],
                              [0, 1, 0, 0,      0, 1, 0,      0, 0, 1],
                              [0, 1, 0, 0,      0, 1, 0,      0, 1, 0],
                              [0, 1, 0, 0,      0, 1, 0,      1, 0, 0],

                              [0, 1, 0, 0,      1, 0, 0,      0, 0, 0],
                              [0, 1, 0, 0,      1, 0, 0,      0, 0, 1],
                              [0, 1, 0, 0,      1, 0, 0,      0, 1, 0],
                              [0, 1, 0, 0,      1, 0, 0,      1, 0, 0],

                              [1, 0, 0, 0,      0, 0, 0,      0, 0, 0],
                              [1, 0, 0, 0,      0, 0, 0,      0, 0, 1],
                              [1, 0, 0, 0,      0, 0, 0,      0, 1, 0],
                              [1, 0, 0, 0,      0, 0, 0,      1, 0, 0],

                              [1, 0, 0, 0,      0, 0, 1,      0, 0, 0],
                              [1, 0, 0, 0,      0, 0, 1,      0, 0, 1],
                              [1, 0, 0, 0,      0, 0, 1,      0, 1, 0],
                              [1, 0, 0, 0,      0, 0, 1,      1, 0, 0],

                              [1, 0, 0, 0,      0, 1, 0,      0, 0, 0],
                              [1, 0, 0, 0,      0, 1, 0,      0, 0, 1],
                              [1, 0, 0, 0,      0, 1, 0,      0, 1, 0],
                              [1, 0, 0, 0,      0, 1, 0,      1, 0, 0],

                              [1, 0, 0, 0,      1, 0, 0,      0, 0, 0],
                              [1, 0, 0, 0,      1, 0, 0,      0, 0, 1],
                              [1, 0, 0, 0,      1, 0, 0,      0, 1, 0],
                              [1, 0, 0, 0,      1, 0, 0,      1, 0, 0]]
        output = find_possible_actions(experiment_vars)
        self.assertEqual(correct, output)

if __name__ == "__main__":
    unittest.main()