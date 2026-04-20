"""
Classe Backtester.

Conforme PLANEJAMENTO.md - Seções 5.2 e 5.3:
- Simulação: prevê 1 → long, prevê 0 → cash
- Custo de transação configurável (default 0.03%)
- Sem alavancagem, capital fixo 100% quando long
- Métricas: Retorno Acumulado, CAGR, Sharpe, Sortino, Max Drawdown,
  Win Rate, Profit Factor, Calmar Ratio
- Gráficos: equity curve vs Buy&Hold, drawdown, distribuição retornos,
  heatmap mensal, sinais sobre preço
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Taxa Selic anualizada para cálculo do Sharpe (aproximação)
RISK_FREE_ANNUAL = 0.1375  # ~13.75% a.a.
TRADING_DAYS_YEAR = 252


class Backtester:
    """
    Simulador de backtest para estratégia de classificação binária.

    Rules:
    - prediction == 1 → posição comprada (long)
    - prediction == 0 → fora do mercado (cash)
    - Custo de transação aplicado a cada mudança de posição
    """

    def __init__(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        dates: List,
        actual_prices: Optional[np.ndarray] = None,
        initial_capital: float = 100_000.0,
        transaction_cost: float = 0.0003,  # 0.03% — taxa B3
        risk_free_annual: float = RISK_FREE_ANNUAL,
    ):
        """
        Parameters
        ----------
        predictions : np.ndarray
            Previsões binárias (0 ou 1).
        actual_returns : np.ndarray
            Retornos reais do ativo (log-retornos ou retornos simples).
        dates : List
            Datas correspondentes.
        actual_prices : np.ndarray, optional
            Preços de fechamento (para gráficos).
        initial_capital : float
            Capital inicial.
        transaction_cost : float
            Custo por transação (fração).
        risk_free_annual : float
            Taxa livre de risco anualizada.
        """
        self.predictions = np.array(predictions).flatten()
        self.actual_returns = np.array(actual_returns).flatten()
        self.dates = pd.DatetimeIndex(dates)
        self.actual_prices = np.array(actual_prices).flatten() if actual_prices is not None else None
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.risk_free_daily = (1 + risk_free_annual) ** (1 / TRADING_DAYS_YEAR) - 1

        self.equity_curve = None
        self.bh_equity_curve = None
        self.strategy_returns = None
        self.metrics = None
        self._has_run = False

    def run(self) -> "Backtester":
        """Executa a simulação dia a dia."""
        n = len(self.predictions)
        equity = np.zeros(n)
        equity[0] = self.initial_capital

        positions = self.predictions.copy()
        strategy_returns = np.zeros(n)

        for i in range(1, n):
            # Custo de transação se houve mudança de posição
            cost = 0.0
            if positions[i] != positions[i - 1]:
                cost = self.transaction_cost

            if positions[i] == 1:
                # Long: captura o retorno do ativo menos custo
                strategy_returns[i] = self.actual_returns[i] - cost
            else:
                # Cash: sem retorno (simplificado, não aplica CDI)
                strategy_returns[i] = -cost if cost > 0 else 0.0

            equity[i] = equity[i - 1] * (1 + strategy_returns[i])

        # Buy & Hold
        bh_equity = np.zeros(n)
        bh_equity[0] = self.initial_capital
        for i in range(1, n):
            bh_equity[i] = bh_equity[i - 1] * (1 + self.actual_returns[i])

        self.equity_curve = pd.Series(equity, index=self.dates)
        self.bh_equity_curve = pd.Series(bh_equity, index=self.dates)
        self.strategy_returns = pd.Series(strategy_returns, index=self.dates)
        self._has_run = True

        logger.info("Backtest executado com sucesso.")
        return self

    def get_equity_curve(self) -> pd.Series:
        """Retorna a série do patrimônio ao longo do tempo."""
        assert self._has_run, "Execute run() primeiro."
        return self.equity_curve

    def get_metrics(self) -> Dict[str, float]:
        """Calcula e retorna todas as métricas financeiras."""
        assert self._has_run, "Execute run() primeiro."

        ret = self.strategy_returns
        n_days = len(ret)
        n_years = n_days / TRADING_DAYS_YEAR

        # Retorno acumulado
        total_return = (self.equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        bh_return = (self.bh_equity_curve.iloc[-1] / self.initial_capital - 1) * 100

        # CAGR
        if n_years > 0 and self.equity_curve.iloc[-1] > 0:
            cagr = ((self.equity_curve.iloc[-1] / self.initial_capital) ** (1 / n_years) - 1) * 100
        else:
            cagr = 0.0

        # Sharpe Ratio (anualizado)
        excess_returns = ret - self.risk_free_daily
        if excess_returns.std() > 0:
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_YEAR)
        else:
            sharpe = 0.0

        # Sortino Ratio (penaliza só volatilidade negativa)
        downside = ret[ret < 0]
        if len(downside) > 0 and downside.std() > 0:
            sortino = (ret.mean() - self.risk_free_daily) / downside.std() * np.sqrt(TRADING_DAYS_YEAR)
        else:
            sortino = 0.0

        # Max Drawdown
        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax
        max_drawdown = drawdown.min() * 100

        # Win Rate
        trading_days = ret[self.predictions == 1]
        if len(trading_days) > 0:
            win_rate = (trading_days > 0).mean() * 100
        else:
            win_rate = 0.0

        # Profit Factor
        gains = ret[ret > 0].sum()
        losses = abs(ret[ret < 0].sum())
        profit_factor = gains / losses if losses > 0 else float("inf")

        # Calmar Ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # Número de trades (mudanças de posição)
        position_changes = np.diff(self.predictions.astype(int))
        n_trades = int(np.sum(position_changes != 0))

        self.metrics = {
            "retorno_acumulado_pct": round(total_return, 2),
            "retorno_bh_pct": round(bh_return, 2),
            "cagr_pct": round(cagr, 2),
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown_pct": round(max_drawdown, 2),
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": round(profit_factor, 4),
            "calmar_ratio": round(calmar, 4),
            "n_trades": n_trades,
            "n_days": n_days,
        }

        logger.info("Métricas financeiras:")
        for k, v in self.metrics.items():
            logger.info(f"  {k}: {v}")

        return self.metrics

    def plot_results(self, save_path: Optional[str] = None) -> plt.Figure:
        """Gráfico: equity curve da estratégia vs Buy & Hold."""
        assert self._has_run, "Execute run() primeiro."

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.equity_curve.index, self.equity_curve.values,
                label="Estratégia LSTM", linewidth=1.5, color="blue")
        ax.plot(self.bh_equity_curve.index, self.bh_equity_curve.values,
                label="Buy & Hold", linewidth=1.5, color="gray", alpha=0.7)
        ax.set_title("Equity Curve: Estratégia vs Buy & Hold", fontsize=14)
        ax.set_xlabel("Data")
        ax.set_ylabel("Patrimônio (R$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Equity curve salva: {save_path}")

        return fig

    def plot_drawdown(self, save_path: Optional[str] = None) -> plt.Figure:
        """Gráfico do drawdown ao longo do tempo."""
        assert self._has_run, "Execute run() primeiro."

        cummax = self.equity_curve.cummax()
        drawdown = (self.equity_curve - cummax) / cummax * 100

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(drawdown.index, drawdown.values, 0,
                        color="red", alpha=0.3)
        ax.plot(drawdown.index, drawdown.values, color="red", linewidth=0.8)
        ax.set_title("Drawdown", fontsize=14)
        ax.set_xlabel("Data")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Drawdown plot salvo: {save_path}")

        return fig

    def plot_returns_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Distribuição dos retornos diários da estratégia."""
        assert self._has_run, "Execute run() primeiro."

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.strategy_returns, bins=50, kde=True, ax=ax, color="steelblue")
        ax.axvline(0, color="red", linestyle="--", alpha=0.7)
        ax.set_title("Distribuição dos Retornos Diários da Estratégia", fontsize=14)
        ax.set_xlabel("Retorno Diário")
        ax.set_ylabel("Frequência")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_monthly_heatmap(self, save_path: Optional[str] = None) -> plt.Figure:
        """Heatmap de retornos mensais."""
        assert self._has_run, "Execute run() primeiro."

        monthly = self.strategy_returns.resample("ME").sum() * 100
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        })
        pivot = monthly_df.pivot_table(values="return", index="year", columns="month")
        pivot.columns = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                         "Jul", "Ago", "Set", "Out", "Nov", "Dez"][:len(pivot.columns)]

        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
                    linewidths=0.5, ax=ax)
        ax.set_title("Retornos Mensais (%)", fontsize=14)
        ax.set_ylabel("Ano")
        ax.set_xlabel("Mês")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def plot_signals_on_price(
        self, prices: np.ndarray = None, save_path: Optional[str] = None
    ) -> plt.Figure:
        """Sinais do modelo sobrepostos ao gráfico de preços."""
        if prices is None:
            prices = self.actual_prices
        assert prices is not None, "Forneça os preços para este gráfico."

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.dates, prices[:len(self.dates)], color="gray", linewidth=0.8,
                label="Preço", alpha=0.8)

        # Marcar sinais long
        long_mask = self.predictions == 1
        ax.scatter(self.dates[long_mask], prices[:len(self.dates)][long_mask],
                   color="green", alpha=0.4, s=10, label="Long (pred=1)")

        ax.set_title("Sinais do Modelo sobre o Preço", fontsize=14)
        ax.set_xlabel("Data")
        ax.set_ylabel("Preço (R$)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        return fig

    def generate_report(self, report_dir: str) -> str:
        """Gera relatório completo em Markdown e salva métricas JSON."""
        assert self._has_run, "Execute run() primeiro."
        if self.metrics is None:
            self.get_metrics()

        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        # Salvar métricas JSON
        json_path = report_path / "backtest_metrics.json"
        with open(json_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Gerar Markdown
        md = f"""# Relatório de Backtest

## Métricas Financeiras

| Métrica | Valor |
|---|---|
| Retorno Acumulado | {self.metrics['retorno_acumulado_pct']:.2f}% |
| Retorno Buy & Hold | {self.metrics['retorno_bh_pct']:.2f}% |
| CAGR | {self.metrics['cagr_pct']:.2f}% |
| Sharpe Ratio | {self.metrics['sharpe_ratio']:.4f} |
| Sortino Ratio | {self.metrics['sortino_ratio']:.4f} |
| Max Drawdown | {self.metrics['max_drawdown_pct']:.2f}% |
| Win Rate | {self.metrics['win_rate_pct']:.2f}% |
| Profit Factor | {self.metrics['profit_factor']:.4f} |
| Calmar Ratio | {self.metrics['calmar_ratio']:.4f} |
| Nº de Trades | {self.metrics['n_trades']} |
| Dias no Período | {self.metrics['n_days']} |

## Configuração
- Capital Inicial: R$ {self.initial_capital:,.2f}
- Custo por Transação: {self.transaction_cost*100:.3f}%
- Período: {self.dates[0].date()} a {self.dates[-1].date()}
"""

        md_path = report_path / "relatorio_backtest.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"Relatório salvo: {md_path}")
        return md

    def log_to_mlflow(self, report_dir: str, prefix: str = "bt") -> None:
        """Loga todas as métricas e artefatos de backtest na run ativa do MLflow."""
        if self.metrics is None:
            self.get_metrics()

        # Log métricas
        for k, v in self.metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"{prefix}_{k}", v)

        # Gerar e logar gráficos
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        fig = self.plot_results(save_path=str(report_path / "equity_curve.png"))
        mlflow.log_artifact(str(report_path / "equity_curve.png"))
        plt.close(fig)

        fig = self.plot_drawdown(save_path=str(report_path / "drawdown.png"))
        mlflow.log_artifact(str(report_path / "drawdown.png"))
        plt.close(fig)

        # Relatório
        self.generate_report(str(report_path))
        mlflow.log_artifact(str(report_path / "backtest_metrics.json"))
        mlflow.log_artifact(str(report_path / "relatorio_backtest.md"))
