"""Joint covariance of per-class effects fitted on the same rows and clusters (issue #13)."""

import numpy as np
from sklearn.linear_model import LinearRegression

from eucalyptus_impact.causal.dml import EffectEstimate, dml_plr, joint_cluster_cov
from eucalyptus_impact.data.synthetic import EUC, NATIVE
from eucalyptus_impact.dynamic import Components, project, scenario_contrasts
from eucalyptus_impact.real import lookup as LK


def _moment_estimates(rng, rho, n_groups=150, per_group=20, theta=(0.3, -0.2)):
    """Two residual-on-residual estimates whose errors have correlation `rho` by construction.

    Each class has its own treatment residual d_c; the outcome residuals are y_c = theta_c d_c +
    e_c, with cluster shocks and row noise in (e_1, e_2) correlated at rho, and d_1 = d_2. Then
    theta_hat_c - theta_c = sum(d e_c) / sum(d^2), so corr(theta_hat_1, theta_hat_2) = rho.
    """
    n = n_groups * per_group
    groups = np.repeat(np.arange(n_groups), per_group)
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    u = (rng.standard_normal((n_groups, 2)) @ chol.T)[groups]
    v = rng.standard_normal((n, 2)) @ chol.T
    # The cluster shocks load on the treatment residual, so the score is clustered too.
    d = rng.standard_normal(n)
    out = []
    for k in range(2):
        e = d * u[:, k] + v[:, k]
        y = theta[k] * d + e
        jac = float(np.mean(d**2))
        est = float(np.mean(d * y) / jac)
        diag = {"res_y": y, "res_d": d, "groups": groups, "d_error_var": 0.0, "jac": jac}
        out.append(EffectEstimate(f"c{k}", est, np.nan, "test", n, diag=diag))
    return out


def test_joint_cov_recovers_a_known_correlation():
    rng = np.random.default_rng(0)
    for rho in (0.6, -0.5):
        draws, covs = [], []
        for _ in range(200):
            a, b = _moment_estimates(rng, rho)
            draws.append([a.estimate, b.estimate])
            covs.append(joint_cluster_cov([a, b]))
        V = np.mean(covs, axis=0)
        corr = V[0, 1] / np.sqrt(V[0, 0] * V[1, 1])
        assert abs(corr - rho) < 0.06, (rho, corr)
        # The estimated covariance matches the sampling covariance of the estimates.
        emp = np.cov(np.array(draws).T)
        assert np.allclose(V, emp, rtol=0.25, atol=0), (V, emp)
        # And the SE of the difference is the joint one, not the independent (hypot) one.
        se_diff = np.sqrt(V[0, 0] + V[1, 1] - 2 * V[0, 1])
        emp_sd = np.std(np.diff(np.array(draws), axis=1))
        assert abs(se_diff / emp_sd - 1) < 0.15
        hypot = np.hypot(np.sqrt(V[0, 0]), np.sqrt(V[1, 1]))
        # Treating the fits as independent is too wide when rho > 0 and too narrow when rho < 0.
        assert hypot > 1.3 * emp_sd if rho > 0 else hypot < 0.9 * emp_sd


def test_joint_cov_diagonal_matches_cluster_se_of_dml_fits():
    rng = np.random.default_rng(1)
    n = 2000
    groups = np.repeat(np.arange(40), n // 40)
    X = rng.standard_normal((n, 3))
    f = rng.dirichlet([2, 2, 2], n)
    y = 0.4 * f[:, 0] - 0.3 * f[:, 1] + X[:, 0] + rng.standard_normal(n)
    ests = [
        dml_plr(
            y,
            f[:, c],
            np.column_stack([X, np.delete(f, c, axis=1)]),
            groups,
            n_folds=4,
            learner_factory=lambda s: LinearRegression(),
            name=f"c{c}",
        )
        for c in range(2)
    ]
    V = joint_cluster_cov(ests)
    assert np.allclose(np.sqrt(np.diag(V)), [e.se for e in ests])
    assert np.allclose(V, V.T)


def test_lookup_contrast_uses_the_stored_covariance():
    se_e, se_n = 0.01, 0.02
    for rho in (0.8, 0.0, -0.8):
        cov = rho * se_e * se_n
        eff = {
            "cover_effects": {"f_eucalyptus": [-0.02, se_e], "f_native_broadleaf": [-0.05, se_n]},
            "cover_cov": {
                "classes": ["f_eucalyptus", "f_native_broadleaf"],
                "matrix": [[se_e**2, cov], [cov, se_n**2]],
            },
        }
        d, se = LK.cover_contrast(eff, "f_eucalyptus", "f_native_broadleaf")
        assert np.isclose(d, 0.03)
        assert np.isclose(se, np.sqrt(se_e**2 + se_n**2 - 2 * cov))
        c = LK.eucalyptus_contribution(eff, 0.5, "native_broadleaf")
        assert np.isclose(c["ci"][1] - c["estimate"], 1.96 * se * 0.5)


def test_brief_contrast_interval_is_the_joint_one():
    from eucalyptus_impact.i18n_gl import num
    from eucalyptus_impact.real.brief import _native_contrast_txt

    e = EffectEstimate("e", 0.0, 0.01, "DML-PLR", 100)
    nb = EffectEstimate("n", -0.0225, 0.0144, "DML-PLR", 100)
    fire = {"cover_effects": {"f_eucalyptus": e, "f_native_broadleaf": nb}}
    for rho, zero in ((0.0, "inclúe o cero"), (0.8, "exclúe o cero")):
        c = rho * 0.01 * 0.0144
        fire["cover_cov"] = {
            "classes": ["f_eucalyptus", "f_native_broadleaf"],
            "matrix": [[0.01**2, c], [c, 0.0144**2]],
        }
        se = np.sqrt(0.01**2 + 0.0144**2 - 2 * c)
        txt = _native_contrast_txt(fire)
        ci = f"{num((0.0225 - 1.96 * se) * 10, 2)} a {num((0.0225 + 1.96 * se) * 10, 2)}"
        assert ci in txt, (ci, txt)
        assert zero in txt
        assert "aproximado" not in txt


def _components(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return Components(
        conv_rate=np.zeros(n),
        fire_boost=0.0,
        source=np.array([0, 0.2, 0.2, 0.4, 0.2, 0.0]),
        burn_loss=np.array([0, 0.6, 0.5, 0, 0, 0]),
        p_base=rng.uniform(0.01, 0.08, n),
        theta=np.array([0.03, 0.0, -0.02, 0.0, 0, 0]),
        burn_frac=0.4,
        weather=np.array([0.5, 1.0, 1.5]),
    )


def test_project_draws_theta_jointly():
    comp = _components()
    f0 = np.random.default_rng(1).dirichlet([3, 2, 2, 2, 2, 0.5], len(comp.p_base))
    se = np.array([0.02, 0.0, 0.02, 0.0, 0.0, 0.0])

    def band(theta_cov):
        kw = dict(cell_area_ha=100, n_sims=40, seed=3, theta_cov=theta_cov)
        trajs = {
            "BAU": project(f0, comp, 8, "bau", **kw),
            "Restore": project(f0, comp, 8, "restore", np.ones(len(f0)), **kw),
        }
        r = scenario_contrasts(trajs).iloc[0]
        return r["d_burned_p95"] - r["d_burned_p05"]

    indep = np.diag(se**2)
    joint = indep.copy()
    joint[EUC, NATIVE] = joint[NATIVE, EUC] = 0.95 * se[EUC] * se[NATIVE]
    # Restoration moves eucalyptus to native, so its contrast depends on theta_native -
    # theta_euc; a strong positive correlation must narrow the band a lot.
    assert band(joint) < 0.5 * band(indep)
    # theta_se is the independent special case, with identical draws.
    kw = dict(cell_area_ha=100, n_sims=5, seed=3)
    a = project(f0, comp, 4, "bau", theta_se=se, **kw)
    b = project(f0, comp, 4, "bau", theta_cov=indep, **kw)
    assert np.allclose(a["expected_burned_ha"], b["expected_burned_ha"])
