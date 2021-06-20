""""Lexical category bias.

https://www.aclweb.org/anthology/2020.nlp4if-1.2.pdf
"""
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from recombinator.optimal_block_length import optimal_block_length
from recombinator.tapered_block_bootstrap import tapered_block_bootstrap
from tqdm import tqdm


def get_indicators(target_samples: np.array,
                   authoritarian_samples: np.array,
                   control_samples: np.array) -> np.array:
    """Calculate indicator score variables from samples.

    This, like the sampling methods below, calculates over all observations
    for a single category.

    Args:
      target_samples: numpy.array, shape (n_observations, n_samples), for the
        target source.
      authoritarian_samples: numpy.array, shape (n_observations, n_samples),
        for the authoritarian source(s).
      control_samples: numpy.array, shape (n_observations, n_samples), for the
        control source(s).

    Returns:
      numpy.array, shape (n_samples,).
    """
    # take the mean frequency over the observations
    target_means = target_samples.mean(axis=1)
    authoritarian_means = authoritarian_samples.mean(axis=1)
    control_means = control_samples.mean(axis=1)

    # determine the indicators of bias in the direction of the authoritarian
    # source(s)
    target_diff = control_means - target_means
    authoritarian_diff = control_means - authoritarian_means
    indicators = target_diff * authoritarian_diff > 0

    # (n_samples,)
    return indicators


def mean_ci(scores: np.array, alpha: float = 0.05) \
        -> Tuple[float, float, float]:
    left = np.percentile(scores, alpha/2*100)
    right = np.percentile(scores, 100-alpha/2*100)
    return float(scores.mean()), left, right


# NOTE: this is the function for consumers to use, the rest break this down
def score(target: pd.DataFrame,
          authoritarian: pd.DataFrame,
          control: pd.DataFrame,
          cats: List[str],
          n_samples: int,
          alpha: float = 0.05) -> Tuple[float, float, float]:
    """Calculate lexical category bias score.

    Args:
      target: pandas.DataFrame, must have a `date` column, and columns for each
        category in `cats`.
      authoritarian: pandas.DataFrame, must have a `date` column, and columns
        for each category in `cats`.
      control: pandas.DataFrame, must have a `date` column, and columns for each
        category in `cats`.
      cats: List of strings, the names of each category to be analyzed.
      n_samples: Int, the number of bootstrap samples to take.
      alpha: Float, significance level for CI.

    Returns:
      mean: Float.
      low: Float, lower bound of CI.
      high: Float, upper bound of CI.
    """
    # make sure the dfs are sorted by date
    target = target.sort_values(by='date', ascending=True)
    authoritarian = authoritarian.sort_values(by='date', ascending=True)
    control = control.sort_values(by='date', ascending=True)

    # sample and score
    scores = sample_scores(target, authoritarian, control, cats, n_samples)

    # determine CI
    mean, low, high = mean_ci(scores, alpha)

    return mean, low, high


def sample_scores(target: pd.DataFrame,
                  authoritarian: pd.DataFrame,
                  control: pd.DataFrame,
                  cats: List[str],
                  n_samples: int) -> np.array:
    scores = []

    with tqdm(total=len(cats)) as pbar:
        # NOTE: the reason we do this by category is to control memory usage
        for cat in cats:
            pbar.set_description('sampling...')
            target_samples = sample(target, cat, n_samples)
            authoritarian_samples = sample(authoritarian, cat, n_samples)
            control_samples = sample(control, cat, n_samples)

            pbar.set_description('getting indicators...')
            indicators = get_indicators(
                target_samples, authoritarian_samples, control_samples)
            # (1, n_samples)
            indicators = np.expand_dims(indicators, axis=0)
            scores.append(indicators)

            pbar.update()

    # (n_cats, n_samples)
    not_reduced = np.concatenate(scores, axis=0)
    # (n_samples)
    scores = not_reduced.mean(axis=0)

    # scale to be in [-1, 1] and not [0, 1]
    scores = -1 + 2 * scores

    return scores


def sample(df: pd.DataFrame, cat: str, n_samples: int) -> np.array:
    """Perform tapered block bootstrap for a category.

    Args:
      df: pandas.DataFrame, the data to sample.
      cat: String, the column name representing the category to sample.
      n_samples: Int, the number of bootstrap samples.

    Returns:
      numpy.array: of shape (n_samples, n_observations).
    """
    # get optimal block size
    b_star = optimal_block_length(df[cat].values)
    b_star = math.ceil(b_star[0].b_star_cb)

    # (n_samples, n_observations)
    samples = tapered_block_bootstrap(df[cat].values,
                                      block_length=b_star,
                                      replications=n_samples)

    return samples
