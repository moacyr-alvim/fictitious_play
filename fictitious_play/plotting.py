import matplotlib.pyplot as plt
import torch

from .benchmark import equilibrium_bid, theoretical_expected_payoff, theoretical_win_probability
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

    v_grid = torch.linspace(0, 1, 200, device=trainer.device).unsqueeze(1)
    with torch.no_grad():
        b_star = equilibrium_bid(v_grid, n_agents).squeeze(1).cpu()
        v_grid_cpu = v_grid.squeeze(1).cpu()
        bids = [trainer.agents[i](v_grid).squeeze(1).cpu() for i in range(n_agents)]

    plt.figure()
    for i, b in enumerate(bids):
        plt.plot(v_grid_cpu, b, label=f"Agente {i} (aprendido)")
    plt.plot(v_grid_cpu, b_star, "--", color="black",
              label="Equilíbrio teórico b*(v) = v*(N-1)/N")
    plt.xlabel("Valor v")
    plt.ylabel("Lance b")
    plt.legend()
    plt.title("Estratégia de lance aprendida vs. equilíbrio")
    plt.tight_layout()
    plt.savefig(f"bid_function{tag}.png", dpi=150)
    plt.close()

    critic = trainer.critic_for(critic_agent)
    b_grid = torch.linspace(0, 1, 300, device=trainer.device)
    with torch.no_grad():
        learned_p_win = critic.predict_win_prob(b_grid).cpu()
        theoretical_p_win = theoretical_win_probability(b_grid, n_agents).cpu()

    plt.figure()
    plt.plot(b_grid.cpu(), learned_p_win, label="Crítico aprendido (spline)")
    plt.plot(b_grid.cpu(), theoretical_p_win, "--", color="black", label="P(vitória|b) teórica")
    plt.xlabel("Lance b")
    plt.ylabel("P(vitória | b)")
    plt.legend()
    plt.title(f"Crítico do agente {critic_agent} (N={n_agents})")
    plt.tight_layout()
    plt.savefig(f"critic{tag}.png", dpi=150)
    plt.close()

    print(f"Figuras salvas em convergence{tag}.png, payoff{tag}.png, "
          f"bid_function{tag}.png e critic{tag}.png")
