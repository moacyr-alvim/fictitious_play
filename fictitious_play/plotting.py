import matplotlib.pyplot as plt
import torch

from .benchmark import (
    common_value_interior_range,
    equilibrium_bid,
    equilibrium_bid_common_value,
    equilibrium_bid_lambda,
    equilibrium_bid_risk_averse,
    theoretical_expected_payoff,
    theoretical_win_probability,
    theoretical_win_probability_lambda,
    theoretical_win_probability_risk_averse,
)
from .train import FictitiousPlayTrainer


def _mark_growth_stages(growth_rounds: list[int] | None) -> None:
    for i, r in enumerate(growth_rounds or []):
        plt.axvline(r, color="gray", linestyle=":", alpha=0.7,
                    label="Crescimento da rede" if i == 0 else None)


def save_all_plots(trainer: FictitiousPlayTrainer, history: list[dict],
                    n_agents: int, tag: str = "", critic_agent: int = 0,
                    growth_rounds: list[int] | None = None) -> None:
    rounds = [h["round"] for h in history]
    mse_per_agent = list(zip(*(h["mse"] for h in history)))

    plt.figure()
    for i, mse_series in enumerate(mse_per_agent):
        plt.plot(rounds, mse_series, label=f"Agente {i}")
    _mark_growth_stages(growth_rounds)
    plt.yscale("log")
    plt.xlabel("Rodada")
    plt.ylabel("MSE vs. equilíbrio teórico (escala log)")
    plt.legend()
    plt.title(f"Convergência ao equilíbrio de Bayes-Nash (N={n_agents})")
    plt.tight_layout()
    plt.savefig(f"convergence{tag}.png", dpi=150)
    plt.close()

    payoff_per_agent = list(zip(*(h["payoff"] for h in history)))

    plt.figure()
    for i, payoff_series in enumerate(payoff_per_agent):
        plt.plot(rounds, payoff_series, label=f"Agente {i}")
    if trainer.risk_aversion is not None:
        title = f"Payoff médio esperado (N={n_agents}, risk_aversion={trainer.risk_aversion})"
    elif trainer.common_value_noise is not None:
        # No closed-form theoretical payoff derived for this variant (would
        # need integrating the equilibrium over the signal distribution).
        title = f"Payoff médio esperado (N={n_agents}, common_value_noise={trainer.common_value_noise})"
    else:
        # With risk_aversion, bid-shading differs per agent, so the auction
        # isn't efficient and there's no single theoretical payoff shared by
        # all agents (see theoretical_expected_payoff's docstring).
        plt.axhline(theoretical_expected_payoff(n_agents), color="black", linestyle="--",
                    label="Payoff teórico")
        title = f"Payoff médio esperado vs. teórico (N={n_agents})"
    _mark_growth_stages(growth_rounds)
    plt.xlabel("Rodada")
    plt.ylabel("Payoff médio esperado")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"payoff{tag}.png", dpi=150)
    plt.close()

    payment_share = trainer.payment_share
    risk_aversion = trainer.risk_aversion
    common_value_noise = trainer.common_value_noise
    if risk_aversion is not None:
        label_suffix = f"first_price, risk_aversion={risk_aversion}"
    elif common_value_noise is not None:
        label_suffix = f"common_value, noise={common_value_noise}"
    elif payment_share is not None:
        label_suffix = f"payment_share={payment_share:.2f}"
        bid_formula = f"b*(v) = (N-1)v^N / (N*[(1-{payment_share:.2f})v^(N-1)+{payment_share:.2f}])"
    else:
        label_suffix = trainer.auction_type
        bid_formula = ("b*(v) = v*(N-1)/N" if trainer.auction_type == "first_price"
                       else "b*(v) = v^N*(N-1)/N")

    if common_value_noise is not None:
        grid_lo, grid_hi = common_value_interior_range(common_value_noise)
    else:
        grid_lo, grid_hi = 0.0, 1.0
    v_grid = torch.linspace(grid_lo, grid_hi, 200, device=trainer.device).unsqueeze(1)
    with torch.no_grad():
        v_grid_cpu = v_grid.squeeze(1).cpu()
        bids = [trainer.agents[i](v_grid).squeeze(1).cpu() for i in range(n_agents)]
        if risk_aversion is None and common_value_noise is None:
            if payment_share is not None:
                b_star = equilibrium_bid_lambda(v_grid, n_agents, payment_share).squeeze(1).cpu()
            else:
                b_star = equilibrium_bid(v_grid, n_agents, trainer.auction_type).squeeze(1).cpu()
        elif common_value_noise is not None:
            b_star = equilibrium_bid_common_value(v_grid, common_value_noise).squeeze(1).cpu()

    plt.figure()
    for i, b in enumerate(bids):
        label = f"Agente {i} (aprendido)" if common_value_noise is None else f"Agente {i} (aprendido vs. sinal)"
        plt.plot(v_grid_cpu, b, label=label)
        if risk_aversion is not None:
            b_star_i = equilibrium_bid_risk_averse(v_grid, i, tuple(risk_aversion)).squeeze(1).cpu()
            plt.plot(v_grid_cpu, b_star_i, "--", label=f"Agente {i} (teórico, rho={risk_aversion[i]})")
    if risk_aversion is None:
        label = ("Equilíbrio teórico b*(x) = x - noise" if common_value_noise is not None
                 else f"Equilíbrio teórico {bid_formula}")
        plt.plot(v_grid_cpu, b_star, "--", color="black", label=label)
    xlabel = "Sinal x" if common_value_noise is not None else "Valor v"
    plt.xlabel(xlabel)
    plt.ylabel("Lance b")
    plt.legend()
    plt.title(f"Estratégia de lance aprendida vs. equilíbrio ({label_suffix})")
    plt.tight_layout()
    plt.savefig(f"bid_function{tag}.png", dpi=150)
    plt.close()

    critic = trainer.critic_for(critic_agent)
    b_grid = torch.linspace(0, 1, 300, device=trainer.device)
    with torch.no_grad():
        learned_p_win = critic.predict_win_prob(b_grid).cpu()
        if risk_aversion is not None:
            theoretical_p_win = theoretical_win_probability_risk_averse(
                b_grid, critic_agent, tuple(risk_aversion)).cpu()
        elif payment_share is not None:
            theoretical_p_win = theoretical_win_probability_lambda(b_grid, n_agents, payment_share).cpu()
        elif common_value_noise is None:
            theoretical_p_win = theoretical_win_probability(b_grid, n_agents, trainer.auction_type).cpu()

    plt.figure()
    plt.plot(b_grid.cpu(), learned_p_win, label=f"Crítico aprendido ({trainer.critic_type})")
    if common_value_noise is None:
        plt.plot(b_grid.cpu(), theoretical_p_win, "--", color="black", label="P(vitória|b) teórica")
    else:
        # No single theoretical P(win|b) applies here: the win probability
        # genuinely depends on the agent's own signal too (via its
        # correlation with the opponent's), not just on b, so there's no
        # x-independent curve to overlay (see docs/findings_common_value.md).
        pass
    plt.xlabel("Lance b")
    plt.ylabel("P(vitória | b)")
    plt.legend()
    plt.title(f"Crítico do agente {critic_agent} (N={n_agents}, {label_suffix})")
    plt.tight_layout()
    plt.savefig(f"critic{tag}.png", dpi=150)
    plt.close()

    print(f"Figuras salvas em convergence{tag}.png, payoff{tag}.png, "
          f"bid_function{tag}.png e critic{tag}.png")
