import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
import copy

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11


class MultiHorizonBacktester:
    def __init__(self, returns_path='data/log_returns_clean.csv', commission=0.001):
        self.returns = pd.read_csv(returns_path, parse_dates=['Date']).set_index('Date')
        ignored = ['RGBITR', 'UWGN']
        self.returns = self.returns.drop(columns=[c for c in ignored if c in self.returns.columns])

        self.asset_names = self.returns.columns
        self.N = len(self.asset_names)
        self.commission = commission

        self.target_daily_return = 0.21 / 250.0

    def _markowitz_objective(self, weights, cov_matrix, prev_weights):
        variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        turnover_penalty = self.commission * np.sum(np.abs(weights - prev_weights))
        return float(variance + turnover_penalty)

    def _cvar_objective(self, params, returns_S, returns_L, prev_weights, omega=0.5, alpha=0.95):
        w = params[:self.N]
        zeta_S = params[self.N]
        zeta_L = params[self.N + 1]
        u = params[self.N + 2: self.N + 2 + len(returns_S)]
        v = params[self.N + 2 + len(returns_S):]

        cvar_S = zeta_S + (1.0 / ((1.0 - alpha) * len(returns_S))) * np.sum(u)
        cvar_L = zeta_L + (1.0 / ((1.0 - alpha) * len(returns_L))) * np.sum(v)

        risk_part = omega * cvar_S + (1.0 - omega) * cvar_L
        turnover_penalty = self.commission * np.sum(np.abs(w - prev_weights))
        return float(risk_part + turnover_penalty)

    def optimize_markowitz(self, cov_matrix, mean_returns, prev_weights):
        init_w = copy.copy(prev_weights)
        bounds = tuple((0.0, 1.0) for _ in range(self.N))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

        if np.max(mean_returns) >= self.target_daily_return:
            constraints.append({'type': 'ineq', 'fun': lambda w: np.dot(w, mean_returns) - self.target_daily_return})
        else:
            constraints.append({'type': 'ineq', 'fun': lambda w: np.dot(w, mean_returns) - np.median(mean_returns)})

        res = minimize(self._markowitz_objective, init_w, args=(cov_matrix, prev_weights),
                       method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-7})

        if res.success:
            return res.x
        else:
            res_fallback = minimize(self._markowitz_objective, init_w, args=(cov_matrix, prev_weights),
                                    method='SLSQP', bounds=bounds, constraints=[constraints[0]],
                                    options={'ftol': 1e-7})
            return res_fallback.x

    def optimize_cvar(self, returns_S, returns_L, mean_returns, prev_weights):
        T_S, T_L = len(returns_S), len(returns_L)
        num_vars = self.N + 2 + T_S + T_L

        init_params = np.zeros(num_vars)
        init_params[:self.N] = np.copy(prev_weights)
        init_params[self.N] = 0.02
        init_params[self.N + 1] = 0.02

        bounds = ([(0.0, 1.0)] * self.N) + [(-1.0, 1.0), (-1.0, 1.0)] + ([(0.0, None)] * (T_S + T_L))

        constraints = [
            {'type': 'eq', 'fun': lambda p: np.sum(p[:self.N]) - 1.0},
            {'type': 'ineq',
             'fun': lambda p: p[self.N + 2: self.N + 2 + T_S] - (-np.dot(returns_S, p[:self.N])) + p[self.N]},
            {'type': 'ineq', 'fun': lambda p: p[self.N + 2 + T_S:] - (-np.dot(returns_L, p[:self.N])) + p[self.N + 1]}
        ]

        if np.max(mean_returns) >= self.target_daily_return:
            constraints.append(
                {'type': 'ineq', 'fun': lambda p: np.dot(p[:self.N], mean_returns) - self.target_daily_return})
        else:
            constraints.append(
                {'type': 'ineq', 'fun': lambda p: np.dot(p[:self.N], mean_returns) - np.median(mean_returns)})

        res = minimize(self._cvar_objective, init_params, args=(returns_S, returns_L, prev_weights),
                       method='SLSQP', bounds=bounds, constraints=constraints, options={'ftol': 1e-7, 'maxiter': 500})

        if res.success:
            return res.x[:self.N]
        else:
            res_fallback = minimize(self._cvar_objective, init_params, args=(returns_S, returns_L, prev_weights),
                                    method='SLSQP', bounds=bounds, constraints=constraints[:3],
                                    options={'ftol': 1e-7, 'maxiter': 500})
            return res_fallback.x[:self.N]

    def run_backtest(self):
        test_start_date = pd.to_datetime('2024-01-01')
        rebalancing_month = 6
        out_of_sample_returns = self.returns.loc[test_start_date:]
        test_days = out_of_sample_returns.index

        cap_mpt = [1.0]
        cap_cvar = [1.0]

        rebal_pt = test_days[0]
        print(f"Стартовая оптимизация портфелей на дате: {rebal_pt.strftime('%Y-%m-%d')}")

        start_S = rebal_pt - pd.DateOffset(months=4)
        hist_S = self.returns.loc[start_S:rebal_pt].to_numpy()
        start_L = rebal_pt - pd.DateOffset(months=16)
        end_L = rebal_pt - pd.DateOffset(months=4)
        hist_L = self.returns.loc[start_L:end_L].to_numpy()

        mean_returns = self.returns.loc[start_S:rebal_pt].mean().to_numpy()
        cov_matrix_S = self.returns.loc[start_S:rebal_pt].cov().to_numpy()

        w_start = np.ones(self.N) / self.N
        w_mpt = self.optimize_markowitz(cov_matrix_S, mean_returns, w_start)
        w_cvar = self.optimize_cvar(hist_S, hist_L, mean_returns, w_start)

        cap_mpt[0] *= (1.0 - self.commission * np.sum(np.abs(w_mpt - w_start)))
        cap_cvar[0] *= (1.0 - self.commission * np.sum(np.abs(w_cvar - w_start)))

        last_rebal_year_period = (rebal_pt.year, 1 if rebal_pt.month <= 6 else 2)

        for i in range(1, len(test_days)):
            current_date = test_days[i]
            current_year_period = (current_date.year, (current_date.month - 1) // rebalancing_month + 1)

            if current_year_period != last_rebal_year_period:
                rebal_pt = current_date
                print(f"Плановая {rebalancing_month}-месячная ребалансировка: {rebal_pt.strftime('%Y-%m-%d')}")

                start_S = rebal_pt - pd.DateOffset(months=4)
                hist_S = self.returns.loc[start_S:rebal_pt].to_numpy()
                start_L = rebal_pt - pd.DateOffset(months=16)
                end_L = rebal_pt - pd.DateOffset(months=4)
                hist_L = self.returns.loc[start_L:end_L].to_numpy()

                mean_returns = self.returns.loc[start_S:rebal_pt].mean().to_numpy()
                cov_matrix_S = self.returns.loc[start_S:rebal_pt].cov().to_numpy()

                w_mpt_new = self.optimize_markowitz(cov_matrix_S, mean_returns, w_mpt)
                w_cvar_new = self.optimize_cvar(hist_S, hist_L, mean_returns, w_cvar)

                cost_mpt = self.commission * np.sum(np.abs(w_mpt_new - w_mpt))
                cost_cvar = self.commission * np.sum(np.abs(w_cvar_new - w_cvar))

                cap_mpt[-1] *= (1.0 - cost_mpt)
                cap_cvar[-1] *= (1.0 - cost_cvar)

                w_mpt, w_cvar = w_mpt_new, w_cvar_new
                last_rebal_year_period = current_year_period


            day_ret = out_of_sample_returns.loc[current_date].to_numpy()
            p_ret_mpt = np.dot(w_mpt, day_ret)
            p_ret_cvar = np.dot(w_cvar, day_ret)

            cap_mpt.append(cap_mpt[-1] * np.exp(p_ret_mpt))
            cap_cvar.append(cap_cvar[-1] * np.exp(p_ret_cvar))

        results_df = pd.DataFrame(index=test_days)
        results_df['Markowitz'] = cap_mpt
        results_df['Robust_CVaR'] = cap_cvar

        plt.figure(figsize=(10, 5.5))
        plt.plot(results_df.index, results_df['Markowitz'], color='#d62728', linewidth=2,
                 label='Классический Марковиц (Mean-Variance)')
        plt.plot(results_df.index, results_df['Robust_CVaR'], color='#2ca02c', linewidth=2,
                 label='Двухмасштабная Робастная модель (Minimum CVaR)')
        plt.title(
            'Сравнительный анализ динамики накопленного капитала Out-of-Sample\nПериод адаптивного бэктестинга: 2024–2025 гг.',
            fontsize=12, fontweight='bold')
        plt.xlabel('Временной горизонт тестирования стратегий')
        plt.ylabel('Стоимость портфеля (относительно стартового 1.0)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left')

        os.makedirs('data/charts', exist_ok=True)
        plt.savefig('data/charts/backtest_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\nГрафик бэктеста сохранен как data/charts/backtest_performance.png")
        return results_df


if __name__ == "__main__":
    backtester = MultiHorizonBacktester()
    res = backtester.run_backtest()
    print(f"\nФинальная стоимость портфелей на конец 2025 года:")
    print(f"Стратегия Марковица: {res['Markowitz'].iloc[-1]:.4f}")
    print(f"Робастная модель CVaR: {res['Robust_CVaR'].iloc[-1]:.4f}")
