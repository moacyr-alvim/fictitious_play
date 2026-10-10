import matplotlib.pyplot as plt
import torch

from .benchmark import (
    equilibrium_bid,
    equilibrium_bid_lambda,
    theoretical_expected_payoff,
    theoretical_win_probability,
    theoretical_win_probability_lambda,
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
    theo_payoff = theoretical_expected_payoff(n_agents)

    plt.figure()
    for i, payoff_series in enumerate(payoff_per_agent):
        plt.plot(rounds, payoff_series, label=f"Agente {i}")
    plt.axhline(theo_payoff, color="black", linestyle="--", label="Payoff teórico")
    _mark_growth_stages(growth_rounds)
    plt.xlabel("Rodada")
    plt.ylabel("Payoff médio esperado")
    plt.legend()
    plt.title(f"Payoff médio esperado vs. teórico (N={n_agents})")
    plt.tight_layout()
    plt.savefig(f"payoff{tag}.png", dpi=150)
    plt.close()

    payment_share = trainer.payment_share
    if payment_share is not None:
        label_suffix = f"payment_share={payment_share:.2f}"
        bid_formula = f"b*(v) = (N-1)v^N / (N*[(1-{payment_share:.2f})v^(N-1)+{payment_share:.2f}])"
    else:
        label_suffix = trainer.auction_type
        bid_formula = ("b*(v) = v*(N-1)/N" if trainer.auction_type == "first_price"
                       else "b*(v) = v^N*(N-1)/N")

    v_grid = torch.linspace(0, 1, 200, device=trainer.device).unsqueeze(1)
    with torch.no_grad():
        if payment_share is not None:
            b_star = equilibrium_bid_lambda(v_grid, n_agents, payment_share).squeeze(1).cpu()
        else:
            b_star = equilibrium_bid(v_grid, n_agents, trainer.auction_type).squeeze(1).cpu()
        v_grid_cpu = v_grid.squeeze(1).cpu()
        bids = [trainer.agents[i](v_grid).squeeze(1).cpu() for i in range(n_agents)]

    plt.figure()
    for i, b in enumerate(bids):
        plt.plot(v_grid_cpu, b, label=f"Agente {i} (aprendido)")
    plt.plot(v_grid_cpu, b_star, "--", color="black",
              label=f"Equilíbrio teórico {bid_formula}")
    _mark_growth_stages(growth_rounds)
    plt.xlabel("Valor v")
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
        if payment_share is not None:
            theoretical_p_win = theoretical_win_probability_lambda(b_grid, n_agents, payment_share).cpu()
        else:
            theoretical_p_win = theoretical_win_probability(b_grid, n_agents, trainer.auction_type).cpu()

    plt.figure()
    plt.plot(b_grid.cpu(), learned_p_win, label="Crítico aprendido (spline)")
    plt.plot(b_grid.cpu(), theoretical_p_win, "--", color="black", label="P(vitória|b) teórica")
    plt.xlabel("Lance b")
    plt.ylabel("P(vitória | b)")
    plt.legend()
    plt.title(f"Crítico do agente {critic_agent} (N={n_agents}, {label_suffix})")
    plt.tight_layout()
    plt.savefig(f"critic{tag}.png", dpi=150)
    plt.close()

    print(f"Figuras salvas em convergence{tag}.png, payoff{tag}.png, "
          f"bid_function{tag}.png e critic{tag}.png")
