from datetime import datetime, timedelta
import random
import unittest

import numpy as np
import pandas as pd
from scipy import stats

from nlp4if import lexcat


# the tests need synthetic data, so we create that here
# let there be three categores: pos, neg, neu
cats = ['pos', 'neg', 'neu']
# let there be three groups, with the following binomial params:
params = {
    'target': {'pos': 0.01, 'neg': 0.03, 'neu': 0.05},
    'authoritarian': {'pos': 0.08, 'neg': 0.04, 'neu': 0.06},
    'control': {'pos': 0.05, 'neg': 0.01, 'neu': 0.08},
}
# let there be two months (60 days) of data
dates = [datetime(2021, 4, 1) + timedelta(days=i) for i in range(61)]
# then lets sample some data
data = []
# for each day, randomly sample a number of documents uniformly in [30, 60]
number_of_docs = list(range(30, 61))
# for each document, randomly sample a sentence length uniformly in [80, 120]
sentence_lengths = list(range(80, 121))
# control randomization
random.seed(42)
np.random.seed(42)
for group in params:
    for date in dates:
        n_docs = random.choice(number_of_docs)
        for _ in range(n_docs):
            x = {
                'date': date,
                'group': group,
            }
            for cat in cats:
                n = random.choice(sentence_lengths)
                theta = params[group][cat]
                c = stats.binom.rvs((n,), theta)
                freq = c / n
                x[cat] = freq
            data.append(x)
data = pd.DataFrame(data)
target = data[data.group == 'target']
authoritarian = data[data.group == 'authoritarian']
control = data[data.group == 'control']


class TestGetIndicators(unittest.TestCase):

    def test_indicators_take_correct_values(self):
        target = np.array([
            [1, 5, 3],  # 3
            [2, 1, 3],  # 2
        ])
        authoritarian = np.array([
            [3, 5],  # 4
            [6, 8],  # 7
        ])
        control = np.array([
            [2, 2],  # 2
            [5, 7],  # 6
        ])
        indicators = lexcat.get_indicators(target, authoritarian, control)
        expected = np.array([1, 0])
        self.assertTrue(np.array_equal(expected, indicators))


class TestMeanCI(unittest.TestCase):

    def test_mean_and_ci_correct(self):
        scores = np.array(list(range(100)))
        mean, low, high = lexcat.mean_ci(scores, alpha=0.02)
        self.assertEqual(49.5, mean)
        self.assertEqual(0.99, low)
        self.assertEqual(98.01, high)


class TestScore(unittest.TestCase):

    def test_score_returns_expected_results(self):
        mean, low, high = lexcat.score(
            target,
            authoritarian,
            control,
            cats,
            100)
        # it is without noise, so we get no real CI
        self.assertEqual(0.67, round(mean, 2))
        self.assertEqual(0.67, round(low, 2))
        self.assertEqual(0.67, round(high, 2))


class TestSampleScores(unittest.TestCase):

    # NOTE: tested indirectly as part of "score" above
    pass


class TestSample(unittest.TestCase):

    def test_correct_shape_returned(self):
        sample = lexcat.sample(df=target, cat='pos', n_samples=10)
        self.assertEqual((10, len(target)), sample.shape)
