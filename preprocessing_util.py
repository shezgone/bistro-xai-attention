"""
Utility: aggregate daily BISTRO forecast samples to monthly values.
Reconstructed from usage patterns in bistro_runner_30var.py.
"""
import numpy as np


def aggregate_daily_forecast_to_monthly(
    samples: np.ndarray,
    ground_truth: np.ndarray,
    last_input: float,
    steps_per_period: int = 32,
    expected_periods: int = 12,
):
    """
    Aggregate daily-resolution forecast samples to monthly granularity.

    Parameters
    ----------
    samples : (N_samples, T_daily) array of forecast samples
    ground_truth : (T_daily,) array of actual daily values
    last_input : last observed value in context (for de-trending if needed)
    steps_per_period : number of daily steps per monthly period (patch size)
    expected_periods : number of months to output

    Returns
    -------
    preds : (expected_periods,) median monthly predictions
    gts : (expected_periods,) monthly ground truth (mean of each chunk)
    ci : (expected_periods, 2) confidence intervals [lo, hi] at 5th/95th percentile
    """
    n_steps = steps_per_period * expected_periods

    # Truncate to expected length
    samples_trunc = samples[:, :n_steps]
    gt_trunc = ground_truth[:n_steps]

    # Reshape to (N_samples, expected_periods, steps_per_period)
    n_samples = samples_trunc.shape[0]
    samples_monthly = samples_trunc.reshape(n_samples, expected_periods, steps_per_period)
    gt_monthly = gt_trunc.reshape(expected_periods, steps_per_period)

    # Aggregate: mean within each period
    samples_agg = samples_monthly.mean(axis=2)  # (N_samples, expected_periods)
    gt_agg = gt_monthly.mean(axis=1)             # (expected_periods,)

    # Median prediction and CI
    preds = np.median(samples_agg, axis=0)       # (expected_periods,)
    ci_lo = np.percentile(samples_agg, 5, axis=0)
    ci_hi = np.percentile(samples_agg, 95, axis=0)
    ci = np.stack([ci_lo, ci_hi], axis=1)        # (expected_periods, 2)

    return preds, gt_agg, ci
