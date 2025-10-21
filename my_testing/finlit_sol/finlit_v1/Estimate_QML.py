
import numpy as np
from scipy.optimize import minimize
from model_finlit_multi import FinLitMulti

def estimate_qml(config):
    model = FinLitMulti(
        data_path=config["data_path"],
        Kx=config.get("Kx",3),
        bins_assets=config.get("bins_assets",4),
        bins_income=config.get("bins_income",3),
        bins_lit=config.get("bins_lit",3),
        bins_age=config.get("bins_age",3),
        smooth=config.get("smooth",0.5),
        beta=config.get("beta",0.95),
        rf=config.get("rf",0.01),
        quad_n=config.get("quad_n",5),
    )
    model.load_and_bin()

    # θ = (sigma, F0, phi, mu_R, sigma_R)
    theta0 = np.array([
        config.get("sigma0",3.0),
        config.get("F0_0",  500.0),   # in currency units (because A,Y are in levels)
        config.get("phi0",    250.0), # cost reduction per +1 literacy z
        config.get("muR0",     0.06),
        config.get("sigR0",    0.18),
    ], float)

    bounds = [
        (0.5, 8.0),      # sigma
        (0.0, 5000.0),   # F0
        (-2000.0, 2000.0), # phi
        (-0.2, 0.3),     # mu_R
        (0.02, 0.8)      # sig_R
    ]

    def obj(th):
        return - model.loglik_qml(th, model.P_hat)

    res = minimize(obj, theta0, method="L-BFGS-B",
                   bounds=bounds,
                   options={"maxiter": config.get("maxiter", 120), "ftol": 1e-6})
    return res, model
