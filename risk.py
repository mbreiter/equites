import numpy as np
import pandas as pd


def risk_prefs(horizon, aversion, initial_dollars, target_dollars, l, mu_bl1, mu_bl2, cov_bl1):

    if horizon is None:
        horizon = 10

    exposures = (0.05, 0.20)

    alpha = 0.05

    return_target = (target_dollars / initial_dollars) ** (1 / (2 * horizon)) - 1

    safe_target = float(((mu_bl1 + mu_bl2) / 2).mean())

    # set the variances for the first period estimates
    vars = pd.DataFrame(np.diag(cov_bl1), index=cov_bl1.index)

    risk_mul, turn_mul = l, 1

    if horizon <= 1:
        # select the 12 assets with the lowest variances
        cardinality = np.where(vars.rank(ascending=True) > len(mu_bl1) * 1/3, 1, 0).ravel()
        exposures = (0.05, 0.15)
        risk_mul *= 2
        turn_mul *= 0.25
        alpha = 0.20

    elif horizon <= 5:
        cardinality = np.where(pd.DataFrame(np.divide(mu_bl1.values, vars.values).ravel()).rank() > len(mu_bl1) * 1 / 3, 1, 0).ravel()
        risk_mul *= 0.75
        turn_mul *= 1
        alpha = 0.10

    else:
        cardinality = np.where(mu_bl1.rank() > len(mu_bl1) * 1 / 3, 1, 0).ravel()
        exposures = (0.02, 0.35)
        risk_mul *= 0.25
        turn_mul *= 2

    if return_target > safe_target:
        risk_mul *= 0.5

    risk_mul *= aversion

    return alpha, return_target, (risk_mul, turn_mul), exposures, list(cardinality)
