"""
This module provides functions for simulating and analyzing multi-armed bandit algorithms
in the context of agricultural sequential decision-making, including model-based bandits
(epsilon-greedy, UCB, ViOlin) and utilities for fitting yield response models.
"""

import numpy as np 
import scipy.optimize
from scipy.optimize import curve_fit, minimize
from scipy.stats import mode

# 4 models
def mitscherlich(x, A, b, d):
    """Mitscherlich yield response function."""
    return d + A * (1 - np.exp(-b * x))

def quadratic_plateau(x, a, b, c, x0):
    """Quadratic-plateau yield response: quadratic up to x0, then flat."""
    x = np.array(x)
    y = np.where(x <= x0, a + b * x + c * x ** 2, a + b * x0 + c * x0 ** 2)
    return y

def michaelis_menten(x, a, b, d):
    """Michaelis-Menten yield response function."""
    return d + a * x / (b + x)

def logistic(x, A, B, C, d):
    """Logistic yield response function."""
    return d + A / (1 + np.exp(-B * (x - C)))

def profit(x, y, p_y, p_x):
    """
    Calculate profit given input x, yield y, and prices.
    Args:
        x (float): Input level.
        y (float): Yield.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
    Returns:
        float: Profit.
    """
    # Profit is revenue minus cost.
    return p_y * y - p_x * x

def generate_yield(x, model, params, noise_sd=0.5):
    """
    Generate noisy yield data from a specified model and parameters.
    Args:
        x (float or np.ndarray): Input variable(s).
        model (str): Model name ('mitscherlich', 'quadratic_plateau', etc.).
        params (dict): Model parameters.
        noise_sd (float): Standard deviation of Gaussian noise.
    Returns:
        np.ndarray: Simulated yield(s) with noise.
    """
    x = np.atleast_1d(x)
    # Select the correct yield function based on model name
    if model == "mitscherlich":
        y = mitscherlich(x, **params)
    elif model == "quadratic_plateau":
        y = quadratic_plateau(x, **params)
    elif model == "michaelis_menten":
        y = michaelis_menten(x, **params)
    elif model == "logistic":
        y = logistic(x, **params)
    else:
        raise ValueError("Unknown model!")
    # Add Gaussian noise to simulate real-world variability
    return y + np.random.normal(0, noise_sd, size=x.shape)

# Fitting the models to data
def fit_model(x, y, model, params0):
    """
    Fit a yield response model to data using non-linear least squares.
    Args:
        x (np.ndarray): Input data.
        y (np.ndarray): Observed yields.
        model (str): Model name.
        params0 (dict): Initial parameter guesses.
    Returns:
        tuple: (Optimal parameters, Covariance matrix)
    """
    # Select the correct model and set up initial guesses and bounds
    if model == "mitscherlich":
        fun = lambda x, A, b, d: mitscherlich(x, A, b, d)
        p0 = tuple(params0[k] for k in ["A", "b", "d"])
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    elif model == "quadratic_plateau":
        fun = lambda x, a, b, c, x0: quadratic_plateau(x, a, b, c, x0)
        p0 = tuple(params0[k] for k in ["a", "b", "c", "x0"])
        bounds = ([-np.inf, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])
    elif model == "michaelis_menten":
        fun = lambda x, a, b, d: michaelis_menten(x, a, b, d)
        p0 = tuple(params0[k] for k in ["a", "b", "d"])
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
    elif model == "logistic":
        fun = lambda x, A, B, C, d: logistic(x, A, B, C, d)
        p0 = tuple(params0[k] for k in ["A", "B", "C", "d"])
        bounds = ([0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf])
    else:
        raise ValueError("Unknown model!")
    # Use scipy's curve_fit to estimate parameters
    popt, pcov = curve_fit(fun, x, y, p0=p0, bounds=bounds, maxfev=10000)
    return popt, pcov

# Epsilon-greedy bandit function
def epsilon_greedy_bandit(
    model, params_true, params0, x_grid, T, p_y, p_x, noise_sd=0.5, epsilon_sched=None, min_fit=5
):
    """
    Epsilon-greedy bandit algorithm for sequential decision-making.
    Args:
        model (str): Yield model name.
        params_true (dict): True model parameters.
        params0 (dict): Initial parameter guesses.
        x_grid (np.ndarray): Discrete set of possible actions.
        T (int): Time horizon.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        noise_sd (float): Noise standard deviation.
        epsilon_sched (callable): Function for epsilon schedule.
        min_fit (int): Minimum samples before exploitation.
    Returns:
        tuple: (X, Y, profits, param_history)
    """
    X, Y, profits = [], [], []
    param_history = []
    epsilon_sched = epsilon_sched or (lambda t: t**(-0.5))
    popt = np.array(list(params0.values())) # Set initial parameters as guess
    np.random.seed(42)
    for t in range(T):
        # With probability epsilon, explore; otherwise, exploit best known arm
        epsilon = epsilon_sched(t+1)
        explore = (len(X) < min_fit) or (np.random.rand() < epsilon)
        if explore:
            x_t = np.random.choice(x_grid)
        else:
            try:
                popt, pcov = fit_model(np.array(X), np.array(Y), model, params0)
                fine_grid = np.linspace(x_grid.min(), x_grid.max(), 200)
                if model == "mitscherlich":
                    y_pred = mitscherlich(fine_grid, *popt)
                elif model == "quadratic_plateau":
                    y_pred = quadratic_plateau(fine_grid, *popt)
                elif model == "michaelis_menten":
                    y_pred = michaelis_menten(fine_grid, *popt)
                elif model == "logistic":
                    y_pred = logistic(fine_grid, *popt)
                pi_pred = profit(fine_grid, y_pred, p_y, p_x)
                x_star = fine_grid[np.argmax(pi_pred)]
                x_t = x_grid[np.argmin(np.abs(x_grid - x_star))]
            except Exception:
                x_t = np.random.choice(x_grid)
                print("EXCEPTION FOUND EPS")
        param_history.append(popt.copy()) # Append the updated params. Use the previous rounds if not updated
        y_t = generate_yield(np.array([x_t]), model, params_true, noise_sd=noise_sd)[0]
        pi_t = profit(x_t, y_t, p_y, p_x)
        X.append(x_t)
        Y.append(y_t)
        profits.append(pi_t)
    return np.array(X), np.array(Y), np.array(profits), param_history

# UCB bandit function
def ucb_bandit(
    model, params_true, params0, x_grid, T, p_y, p_x, noise_sd=0.5, alpha=1.0, min_fit=5
):
    """
    Upper Confidence Bound (UCB) bandit algorithm for sequential decision-making.
    Args:
        model (str): Yield model name.
        params_true (dict): True model parameters.
        params0 (dict): Initial parameter guesses.
        x_grid (np.ndarray): Discrete set of possible actions.
        T (int): Time horizon.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        noise_sd (float): Noise standard deviation.
        alpha (float): UCB exploration parameter.
        min_fit (int): Minimum samples before exploitation.
    Returns:
        tuple: (X, Y, profits, param_history)
    """
    X, Y, profits = [], [], []
    param_history = []
    np.random.seed(42)
    # Each model needs its own gradient function for UCB; here are basic versions:
    def grad_fun(x, popt):
        """
        Approximates gradient with respect to parameters for uncertainty calculation.
        Args:
            x (float): Input value.
            popt (np.ndarray): Model parameters.
        Returns:
            np.ndarray: Gradient vector.
        """
        h = 1e-4
        grads = []
        for i in range(len(popt)):
            p1 = np.array(popt)
            p2 = np.array(popt)
            p1[i] += h
            p2[i] -= h
            if model == "mitscherlich":
                f1 = mitscherlich(np.array([x]), *p1)[0]
                f2 = mitscherlich(np.array([x]), *p2)[0]
            elif model == "quadratic_plateau":
                f1 = quadratic_plateau(np.array([x]), *p1)[0]
                f2 = quadratic_plateau(np.array([x]), *p2)[0]
            elif model == "michaelis_menten":
                f1 = michaelis_menten(np.array([x]), *p1)[0]
                f2 = michaelis_menten(np.array([x]), *p2)[0]
            elif model == "logistic":
                f1 = logistic(np.array([x]), *p1)[0]
                f2 = logistic(np.array([x]), *p2)[0]
            grads.append((f1 - f2) / (2 * h))
        return np.array(grads)
    popt = np.array(list(params0.values())) # Set initial parameters as guess
    for t in range(T):
        # For first min_fit rounds, explore randomly; then exploit UCB
        if len(X) < min_fit:
            x_t = np.random.choice(x_grid)
        else:
            try:
                popt, pcov = fit_model(np.array(X), np.array(Y), model, params0)
                profit_preds, unc_preds = [], []
                for xi in x_grid:
                    if model == "mitscherlich":
                        y_hat = mitscherlich(np.array([xi]), *popt)[0]
                    elif model == "quadratic_plateau":
                        y_hat = quadratic_plateau(np.array([xi]), *popt)[0]
                    elif model == "michaelis_menten":
                        y_hat = michaelis_menten(np.array([xi]), *popt)[0]
                    elif model == "logistic":
                        y_hat = logistic(np.array([xi]), *popt)[0]
                    pi_hat = profit(xi, y_hat, p_y, p_x)
                    grad_i = grad_fun(xi, popt)
                    unc = np.sqrt(np.dot(grad_i, np.dot(pcov, grad_i)))
                    profit_preds.append(pi_hat)
                    unc_preds.append(unc)
                ucb = np.array(profit_preds) + alpha * np.array(unc_preds)
                x_t = x_grid[np.argmax(ucb)]
            except Exception:
                x_t = np.random.choice(x_grid)
                print("EXCEPTION FOUND UCB ")
        param_history.append(popt.copy()) # Append the updated params. Use the previous rounds if not updated
        y_t = generate_yield(np.array([x_t]), model, params_true, noise_sd=noise_sd)[0]
        pi_t = profit(x_t, y_t, p_y, p_x)
        X.append(x_t)
        Y.append(y_t)
        profits.append(pi_t)
    return np.array(X), np.array(Y), np.array(profits), param_history

def finite_diff_grad_hess(fun, x, h=1e-2):
    """
    Finite difference approximation for gradient and Hessian of a function.
    Args:
        fun (callable): Function to differentiate.
        x (float): Point at which to evaluate.
        h (float): Step size.
    Returns:
        tuple: (gradient, hessian)
    """
    f_xph = fun(x + h)
    f_xmh = fun(x - h)
    f_x = fun(x)
    grad = (f_xph - f_xmh) / (2 * h)
    hess = (f_xph - 2 * f_x + f_xmh) / (h ** 2)
    return grad, hess

def violin_bandit_curvature(
    model, params_true, params0, x_grid, T, p_y, p_x,
    noise_sd=0.5, kappa1=2.0, kappa2=640.0, min_fit=5
):
    """
    ViOlin bandit algorithm with curvature-aware score for exploration.
    Args:
        model (str): Yield model name.
        params_true (dict): True model parameters.
        params0 (dict): Initial parameter guesses.
        x_grid (np.ndarray): Discrete set of possible actions.
        T (int): Time horizon.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        noise_sd (float): Noise standard deviation.
        kappa1 (float): Gradient bonus parameter.
        kappa2 (float): Curvature bonus parameter.
        min_fit (int): Minimum samples before exploitation.
    Returns:
        tuple: (X, Y, profits, param_history)
    """
    X, Y, profits, param_history = [], [], [], []
    rng = np.random.default_rng(42)
    # ------- helper: analytic ∂/∂x and ∂²/∂x² for each model ----------
    def f_yield(x, θ):
        if model == "mitscherlich":
            A, b, d = θ ; return A*(1-np.exp(-b*(x-d)))
        if model == "quadratic_plateau":
            a,b,c,x0 = θ ; return np.where(x<=x0, a+b*x+c*x**2, a+b*x0+c*x0**2)
        if model == "michaelis_menten":
            a,b,d = θ ; return a*(x-d)/(b+x-d)
        if model == "logistic":
            A,B,C,d = θ ; return A/(1+np.exp(-B*(x-C))) + d
    def df_dx(x, θ):
        if model == "mitscherlich":
            A,b,d = θ ; return A*b*np.exp(-b*(x-d))
        if model == "quadratic_plateau":
            a,b,c,x0 = θ ; return np.where(x<=x0, b+2*c*x, 0.0)
        if model == "michaelis_menten":
            a,b,d = θ
            u = x-d ; return a*b/(b+u)**2
        if model == "logistic":
            A,B,C,_ = θ
            ez = np.exp(-B*(x-C)) ; return A*B*ez/(1+ez)**2
    def d2f_dx2(x, θ):
        if model == "mitscherlich":
            A,b,d = θ ; return -A*b**2*np.exp(-b*(x-d))
        if model == "quadratic_plateau":
            a,b,c,x0 = θ ; return np.where(x<=x0, 2*c, 0.0)
        if model == "michaelis_menten":
            a,b,d = θ
            u = x-d ; return -2*a*b/(b+u)**3
        if model == "logistic":
            A,B,C,_ = θ
            ez = np.exp(-B*(x-C))
            return A*B**2*ez*(ez-1)/(1+ez)**3
    # -------- initialise with one random arm ----------
    x0 = rng.choice(x_grid)
    y0 = generate_yield(x0, model, params_true, noise_sd)[0]
    X, Y, profits = [x0], [y0], [profit(x0, y0, p_y, p_x)]
    θ_hat = np.array(list(params0.values()))
    param_history.append(θ_hat.copy())
    for t in range(1, T):
        # For first min_fit rounds, explore randomly; then exploit curvature-aware score
        if len(X) < min_fit:
            x_t = rng.choice(x_grid)          # pure exploration
        else:
            θ_hat, _ = fit_model(np.array(X), np.array(Y), model, params0)
            μ = f_yield(x_grid, θ_hat)
            grad = df_dx(x_grid, θ_hat)
            curv = d2f_dx2(x_grid, θ_hat)
            profit_pred = p_y * μ - p_x * x_grid
            score = (profit_pred
                     + kappa1 * np.abs(grad)
                     + kappa2 * np.abs(curv)) #score function that takes curvature into account
            x_t = x_grid[np.argmax(score)]
        param_history.append(θ_hat.copy())
        y_t = generate_yield(x_t, model, params_true, noise_sd)[0]
        X.append(x_t)
        Y.append(y_t)
        profits.append(profit(x_t, y_t, p_y, p_x))
    return np.array(X), np.array(Y), np.array(profits), param_history

def true_profit_maximizer(params, model, p_y, p_x, x_grid):
    """
    Find the input level that maximizes true profit for a given model and parameters.
    Args:
        params (dict): True model parameters.
        model (str): Model name.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        x_grid (np.ndarray): Discrete set of possible actions.
    Returns:
        tuple: (x_star, pi_star) where x_star is the optimal input and pi_star is the max profit.
    """
    # Compute yield for each input level in the grid
    if model == "mitscherlich":
        y_grid = mitscherlich(x_grid, **params)
    elif model == "quadratic_plateau":
        y_grid = quadratic_plateau(x_grid, **params)
    elif model == "michaelis_menten":
        y_grid = michaelis_menten(x_grid, **params)
    elif model == "logistic":
        y_grid = logistic(x_grid, **params)
    else:
        raise ValueError("Unknown model!")
    # Calculate profit for each input
    pi_grid = p_y * y_grid - p_x * x_grid
    x_star = x_grid[np.argmax(pi_grid)]
    pi_star = np.max(pi_grid)
    return x_star, pi_star

def run_bandit_replicates(
    bandit_func, model, params_true, params0, x_grid, T, p_y, p_x, noise_sd=0.5, reps=30, **bandit_kwargs
):
    """
    Run multiple replicates of a bandit algorithm and collect statistics.
    Args:
        bandit_func (callable): Bandit function to run.
        model (str): Model name.
        params_true (dict): True model parameters.
        params0 (dict): Initial parameter guesses.
        x_grid (np.ndarray): Discrete set of possible actions.
        T (int): Time horizon.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        noise_sd (float): Noise standard deviation.
        reps (int): Number of replicates.
        **bandit_kwargs: Additional arguments for the bandit function.
    Returns:
        tuple: (avg_profit, avg_cumprofit, avg_regret, avg_arm, profits_mat, regret_mat, arms_mat, avg_param)
    """
    profits_mat = np.zeros((reps, T))
    regret_mat = np.zeros((reps, T))
    arms_mat = np.zeros((reps, T))
    params_mat = np.zeros((reps, T, len(params0)))
    # Find the optimal profit for regret calculation
    _, pi_star = true_profit_maximizer(params_true, model, p_y, p_x, x_grid)
    for r in range(reps):
        # Run one replicate of the bandit algorithm
        X, Y, profits, param_history = bandit_func(
            model, params_true, params0, x_grid, T, p_y, p_x, noise_sd=noise_sd, **bandit_kwargs
        )
        profits_mat[r, :] = profits
        arms_mat[r, :] = X
        cumulative_profit = np.cumsum(profits)
        optimal_cumprofit = np.cumsum(np.repeat(pi_star, T))
        regret_mat[r, :] = optimal_cumprofit - cumulative_profit
        params_mat[r, :] = param_history
    avg_profit = profits_mat.mean(axis=0)
    avg_cumprofit = np.cumsum(avg_profit)
    avg_regret = regret_mat.mean(axis=0)
    avg_arm = mode(arms_mat, axis=0).mode[0]
    avg_param = params_mat.mean(axis=0)
    return (avg_profit, avg_cumprofit, avg_regret, avg_arm,
            profits_mat, regret_mat, arms_mat, avg_param)

def epsilon_greedy_bandit_misspecified(
    model_true, params_true, model_fit, params0, x_grid, T, p_y, p_x, noise_sd=0.5, epsilon_sched=None, min_fit=5
):
    """
    Epsilon-greedy bandit algorithm for the case where the fitted model is misspecified.
    Args:
        model_true (str): True data-generating model name.
        params_true (dict): True model parameters.
        model_fit (str): Model used for fitting/learning.
        params0 (dict): Initial parameter guesses.
        x_grid (np.ndarray): Discrete set of possible actions.
        T (int): Time horizon.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        noise_sd (float): Noise standard deviation.
        epsilon_sched (callable): Function for epsilon schedule.
        min_fit (int): Minimum samples before exploitation.
    Returns:
        tuple: (X, Y, profits, param_history)
    """
    X, Y, profits = [], [], []
    param_history = []
    epsilon_sched = epsilon_sched or (lambda t: t**(-0.5))
    popt = np.array(list(params0.values())) # Set initial parameters as guess
    np.random.seed(42)
    for t in range(T):
        # With probability epsilon, explore; otherwise, exploit best known arm (using misspecified model)
        epsilon = epsilon_sched(t+1)
        explore = (len(X) < min_fit) or (np.random.rand() < epsilon)
        if explore:
            x_t = np.random.choice(x_grid)
        else:
            try:
                popt, pcov = fit_model(np.array(X), np.array(Y), model_fit, params0)
                fine_grid = np.linspace(x_grid.min(), x_grid.max(), 200)
                if model_fit == "mitscherlich":
                    y_pred = mitscherlich(fine_grid, *popt)
                elif model_fit == "quadratic_plateau":
                    y_pred = quadratic_plateau(fine_grid, *popt)
                elif model_fit == "michaelis_menten":
                    y_pred = michaelis_menten(fine_grid, *popt)
                elif model_fit == "logistic":
                    y_pred = logistic(fine_grid, *popt)
                pi_pred = profit(fine_grid, y_pred, p_y, p_x)
                x_star = fine_grid[np.argmax(pi_pred)]
                x_t = x_grid[np.argmin(np.abs(x_grid - x_star))]
            except Exception:
                x_t = np.random.choice(x_grid)
                print("EXCEPTION FOUND EPS")
        # Data generated by the true model
        param_history.append(popt.copy()) 
        y_t = generate_yield(np.array([x_t]), model_true, params_true, noise_sd=noise_sd)[0]
        pi_t = profit(x_t, y_t, p_y, p_x)
        X.append(x_t)
        Y.append(y_t)
        profits.append(pi_t)
    return np.array(X), np.array(Y), np.array(profits), param_history

def ucb_bandit_misspecified(
    model_true, params_true, model_fit, params0, x_grid, T, p_y, p_x, noise_sd=0.5, alpha=1.0, min_fit=5
):
    """
    UCB bandit algorithm for the case where the fitted model is misspecified.
    Args:
        model_true (str): True data-generating model name.
        params_true (dict): True model parameters.
        model_fit (str): Model used for fitting/learning.
        params0 (dict): Initial parameter guesses.
        x_grid (np.ndarray): Discrete set of possible actions.
        T (int): Time horizon.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        noise_sd (float): Noise standard deviation.
        alpha (float): UCB exploration parameter.
        min_fit (int): Minimum samples before exploitation.
    Returns:
        tuple: (X, Y, profits, param_history)
    """
    X, Y, profits = [], [], []
    param_history = []
    np.random.seed(42)
    def grad_fun(x, popt):
        """
        Approximates gradient with respect to parameters for uncertainty calculation (misspecified model).
        Args:
            x (float): Input value.
            popt (np.ndarray): Model parameters.
        Returns:
            np.ndarray: Gradient vector.
        """
        h = 1e-4
        grads = []
        for i in range(len(popt)):
            p1 = np.array(popt)
            p2 = np.array(popt)
            p1[i] += h
            p2[i] -= h
            if model_fit == "mitscherlich":
                f1 = mitscherlich(np.array([x]), *p1)[0]
                f2 = mitscherlich(np.array([x]), *p2)[0]
            elif model_fit == "quadratic_plateau":
                f1 = quadratic_plateau(np.array([x]), *p1)[0]
                f2 = quadratic_plateau(np.array([x]), *p2)[0]
            elif model_fit == "michaelis_menten":
                f1 = michaelis_menten(np.array([x]), *p1)[0]
                f2 = michaelis_menten(np.array([x]), *p2)[0]
            elif model_fit == "logistic":
                f1 = logistic(np.array([x]), *p1)[0]
                f2 = logistic(np.array([x]), *p2)[0]
            grads.append((f1 - f2) / (2 * h))
        return np.array(grads)
    popt = np.array(list(params0.values())) # Set initial parameters as guess
    for t in range(T):
        # For first min_fit rounds, explore randomly; then exploit UCB (using misspecified model)
        if len(X) < min_fit:
            x_t = np.random.choice(x_grid)
        else:
            try:
                popt, pcov = fit_model(np.array(X), np.array(Y), model_fit, params0)
                profit_preds, unc_preds = [], []
                for xi in x_grid:
                    if model_fit == "mitscherlich":
                        y_hat = mitscherlich(np.array([xi]), *popt)[0]
                    elif model_fit == "quadratic_plateau":
                        y_hat = quadratic_plateau(np.array([xi]), *popt)[0]
                    elif model_fit == "michaelis_menten":
                        y_hat = michaelis_menten(np.array([xi]), *popt)[0]
                    elif model_fit == "logistic":
                        y_hat = logistic(np.array([xi]), *popt)[0]
                    pi_hat = profit(xi, y_hat, p_y, p_x)
                    grad_i = grad_fun(xi, popt)
                    unc = np.sqrt(np.dot(grad_i, np.dot(pcov, grad_i)))
                    profit_preds.append(pi_hat)
                    unc_preds.append(unc)
                ucb = np.array(profit_preds) + alpha * np.array(unc_preds)
                x_t = x_grid[np.argmax(ucb)]
            except Exception:
                x_t = np.random.choice(x_grid)
                print("EXCEPTION FOUND UCB")
        param_history.append(popt.copy()) # Append the updated params. Use the previous rounds if not updated
        y_t = generate_yield(np.array([x_t]), model_true, params_true, noise_sd=noise_sd)[0]
        pi_t = profit(x_t, y_t, p_y, p_x)
        X.append(x_t)
        Y.append(y_t)
        profits.append(pi_t)
    return np.array(X), np.array(Y), np.array(profits), param_history

def violin_bandit_curvature_misspecified(
        model_true, params_true,
        model_fit,  params0,
        x_grid, T,
        p_y, p_x,
        noise_sd      = 0.5,
        kappa_grad    = 2.0,          # κ1  (gradient bonus)
        kappa_hess    = 640.0,         # κ2  (curvature bonus)   <-- tune!
        min_fit       = 5,
        fd_h          = 1e-2,
        seed          = 42):
    """
    ViOlin with curvature–aware score   (misspecified version).

    score(x) =  \hatΠ(x)
              + κ₁ |∂Π/∂x|                (encourages steep ascent directions)
              + κ₂ max(0, -∂²Π/∂x²)       (prefers locally concave peaks)

    κ₁, κ₂ are hyper–parameters ***you*** control.
    """
    """
    ViOlin bandit algorithm with curvature-aware score for the misspecified model case.
    Args:
        model_true (str): True data-generating model name.
        params_true (dict): True model parameters.
        model_fit (str): Model used for fitting/learning.
        params0 (dict): Initial parameter guesses.
        x_grid (np.ndarray): Discrete set of possible actions.
        T (int): Time horizon.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        noise_sd (float): Noise standard deviation.
        kappa_grad (float): Gradient bonus parameter.
        kappa_hess (float): Curvature bonus parameter.
        min_fit (int): Minimum samples before exploitation.
        fd_h (float): Step size for finite difference.
        seed (int): Random seed.
    Returns:
        tuple: (X, Y, profits, param_history)
    """
    rng = np.random.default_rng(seed)
    X, Y, profits, param_history = [], [], [], []

    # Map model names to functions
    f_map = {
        "mitscherlich":       lambda x,*p: mitscherlich(x,*p),
        "quadratic_plateau":  lambda x,*p: quadratic_plateau(x,*p),
        "michaelis_menten":   lambda x,*p: michaelis_menten(x,*p),
        "logistic":           lambda x,*p: logistic(x,*p),
    }
    f_true  = f_map[model_true]
    f_fit   = f_map[model_fit]

    # Start with one random observation
    x0 = rng.choice(x_grid)
    y0 = generate_yield(np.array([x0]), model_true, params_true, noise_sd=noise_sd)[0]
    X.append(x0);  Y.append(y0)
    profits.append(profit(x0, y0, p_y, p_x))
    popt = np.array(list(params0.values()))
    prev_popt = popt.copy()
    param_history.append(popt.copy())

    # Main loop
    for t in range(1, T):

        # Fit misspecified model if enough data, else use previous fit
        if len(X) >= min_fit:
            try:
                prev_popt = popt.copy()  # save previous parameters
                popt, _ = fit_model(np.array(X), np.array(Y), model_fit, params0)
                not_enough_data = False
            except Exception:
                popt = prev_popt
        else:
            popt = prev_popt
            not_enough_data = True

        # Decide which arm to play
        if not_enough_data:
            # Not enough data or fit failed: explore randomly
            x_t = rng.choice(x_grid)
            grad_t = hess_t = None
        else:
            score = []

            # Compute optimism score for each arm
            for x in x_grid:
                y_hat = f_fit(np.array([x]), *popt)[0]
                pi_hat = profit(x, y_hat, p_y, p_x)

                # local 1-D profit function for FD derivative
                local_fun = lambda z: profit(
                        z,
                        f_fit(np.array([z]), *popt)[0],
                        p_y, p_x
                    )
                g, hess = finite_diff_grad_hess(local_fun, x, h=fd_h)

                # ViOlin-style optimism score
                s = pi_hat \
                    + kappa_grad * abs(g) \
                    + kappa_hess * max(0.0, -hess)   # concavity bonus
                score.append(s)
            x_t = x_grid[int(np.argmax(score))]      # tie-break leftmost
            grad_t, hess_t = finite_diff_grad_hess(
                                lambda z: profit(z,
                                                 f_fit(np.array([z]), *popt)[0],
                                                 p_y, p_x),
                                x_t, h=fd_h)
        param_history.append(popt.copy())

        # Pull arm and observe reward
        y_t  = generate_yield(np.array([x_t]), model_true, params_true, noise_sd=noise_sd)[0]
        pi_t = profit(x_t, y_t, p_y, p_x)
        X.append(x_t);  Y.append(y_t);   profits.append(pi_t)

        # (You can store (grad_t,hess_t) in a separate structure if wanted)
    return np.array(X), np.array(Y), np.array(profits), param_history

def run_bandit_replicates_misspecified(
    bandit_func,
    model_true,           # True data-generating model (string)
    params_true,          # True parameters (dict)
    model_fit,            # Model used for fitting/learning (string)
    params0,              # Initial guess for fit
    x_grid,               # Grid of available actions/arms
    T,                    # Time horizon
    p_y,                  # Price per unit yield
    p_x,                  # Price per unit input
    noise_sd=0.5,         # Noise level in reward
    reps=30,              # Number of replicates
    **bandit_kwargs
):
    """
    Run multiple replicates of a bandit algorithm with potentially misspecified models and collect statistics.
    Args:
        bandit_func (callable): Bandit function to run (must have interface: model_true, params_true, model_fit, params0, ...).
        model_true (str): True data-generating model name.
        params_true (dict): True model parameters.
        model_fit (str): Model used for fitting/learning.
        params0 (dict): Initial parameter guesses.
        x_grid (np.ndarray): Discrete set of possible actions.
        T (int): Time horizon.
        p_y (float): Price per unit yield.
        p_x (float): Price per unit input.
        noise_sd (float): Noise standard deviation.
        reps (int): Number of replicates.
        **bandit_kwargs: Additional arguments for the bandit function.
    Returns:
        tuple: (avg_profit, avg_cumprofit, avg_regret, avg_arm, profits_mat, regret_mat, arms_mat, avg_param)
    """
    profits_mat = np.zeros((reps, T))
    regret_mat = np.zeros((reps, T))
    arms_mat = np.zeros((reps, T))
    params_mat = np.zeros((reps, T, len(params0)))

    # Oracle profit with respect to the true model
    _, pi_star = true_profit_maximizer(params_true, model_true, p_y, p_x, x_grid)
    for r in range(reps):
        
        # Note the function signature here: model_true, params_true, model_fit, params0, ...
        X, Y, profits, param_history = bandit_func(
            model_true, params_true, model_fit, params0,
            x_grid, T, p_y, p_x, noise_sd=noise_sd, **bandit_kwargs
        )
        profits_mat[r, :] = profits
        arms_mat[r, :] = X
        cumulative_profit = np.cumsum(profits)
        optimal_cumprofit = np.cumsum(np.repeat(pi_star, T))
        regret_mat[r, :] = optimal_cumprofit - cumulative_profit
        params_mat[r, :] = param_history
    avg_profit = profits_mat.mean(axis=0)
    avg_cumprofit = np.cumsum(avg_profit)
    avg_regret = regret_mat.mean(axis=0)
    avg_arm = mode(arms_mat, axis=0).mode[0]
    avg_param = params_mat.mean(axis=0)

    return (avg_profit, avg_cumprofit, avg_regret, avg_arm,
            profits_mat, regret_mat, arms_mat, avg_param)