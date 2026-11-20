import matplotlib.pyplot as plt
import torch

from .benchmark import (
    common_value_interior_range,
    equilibrium_bid,
    equilibrium_bid_common_value,
    equilibrium_bid_lambda,
    equilibrium_bid_risk_averse,
    equilibrium_bid_wallet_game,
    theoretical_expected_payoff,
    theoretical_win_probability,
    theoretical_win_probability_lambda,
    theoretical_win_probability_risk_averse,
    theoretical_win_probability_wallet_game,
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

    # common_value_gaussian has no known theoretical bid function (see
    # common_value_posterior_mean_gaussian's docstring), so _benchmark_metrics
    # returns NaN rather than a real MSE/MaxDiff — skip this plot entirely
    # rather than render an empty/misleading one.
    if trainer.common_value_gaussian is None:
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
    elif trainer.common_value_gaussian is not None:
        title = f"Payoff médio esperado (N={n_agents}, common_value_gaussian={trainer.common_value_gaussian})"
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
    common_value_gaussian = trainer.common_value_gaussian
    common_value_wallet_game = trainer.common_value_wallet_game
    # "is_common_value" governs signal-vs-value terminology (x-axis label,
    # "aprendido" vs. "aprendido vs. sinal") for all three common-value
    # variants alike; whether each one actually *has* a known theoretical
    # bid curve to overlay is handled separately below (wallet_game does,
    # the other two don't).
    is_common_value = (common_value_noise is not None or common_value_gaussian is not None
                        or common_value_wallet_game)
    if risk_aversion is not None:
        label_suffix = f"first_price, risk_aversion={risk_aversion}"
    elif common_value_noise is not None:
        label_suffix = f"common_value, noise={common_value_noise}"
    elif common_value_gaussian is not None:
        label_suffix = f"common_value_gaussian, params={common_value_gaussian}"
    elif common_value_wallet_game:
        label_suffix = f"common_value_wallet_game, N={n_agents}"
    elif payment_share is not None:
        label_suffix = f"payment_share={payment_share:.2f}"
        bid_formula = f"b*(v) = (N-1)v^N / (N*[(1-{payment_share:.2f})v^(N-1)+{payment_share:.2f}])"
    else:
        label_suffix = trainer.auction_type
        bid_formula = ("b*(v) = v*(N-1)/N" if trainer.auction_type == "first_price"
                       else "b*(v) = v^N*(N-1)/N")

    # Whether each variant actually has a known theoretical bid curve
    # (wallet_game does; common_value_noise/gaussian don't — see their
    # benchmark.py docstrings).
    no_theoretical_bid = common_value_gaussian is not None
    if common_value_noise is not None:
        grid_lo, grid_hi = common_value_interior_range(common_value_noise)
    elif common_value_gaussian is not None:
        # No boundary to respect (unbounded support — see
        # common_value_posterior_mean_gaussian's docstring), just a wide
        # practical plotting range around the signal distribution.
        prior_mean, prior_std, noise_std = common_value_gaussian
        signal_std = (prior_std ** 2 + noise_std ** 2) ** 0.5
        grid_lo, grid_hi = prior_mean - 4 * signal_std, prior_mean + 4 * signal_std
    else:
        grid_lo, grid_hi = 0.0, 1.0
    v_grid = torch.linspace(grid_lo, grid_hi, 200, device=trainer.device).unsqueeze(1)
    with torch.no_grad():
        v_grid_cpu = v_grid.squeeze(1).cpu()
        bids = [trainer.agents[i](v_grid).squeeze(1).cpu() for i in range(n_agents)]
        if risk_aversion is None and not is_common_value:
            if payment_share is not None:
                b_star = equilibrium_bid_lambda(v_grid, n_agents, payment_share).squeeze(1).cpu()
            else:
                b_star = equilibrium_bid(v_grid, n_agents, trainer.auction_type).squeeze(1).cpu()
        elif common_value_noise is not None:
            b_star = equilibrium_bid_common_value(v_grid, common_value_noise).squeeze(1).cpu()
        elif common_value_wallet_game:
            b_star = equilibrium_bid_wallet_game(v_grid, n_agents).squeeze(1).cpu()
        # common_value_gaussian: no b_star, no known equilibrium to plot.

    plt.figure()
    for i, b in enumerate(bids):
        label = f"Agente {i} (aprendido)" if not is_common_value else f"Agente {i} (aprendido vs. sinal)"
        plt.plot(v_grid_cpu, b, label=label)
        if risk_aversion is not None:
            b_star_i = equilibrium_bid_risk_averse(v_grid, i, tuple(risk_aversion)).squeeze(1).cpu()
            plt.plot(v_grid_cpu, b_star_i, "--", label=f"Agente {i} (teórico, rho={risk_aversion[i]})")
    if risk_aversion is None and not no_theoretical_bid:
        if common_value_noise is not None:
            label = "Equilíbrio teórico b*(x) = x - noise"
        elif common_value_wallet_game:
            label = "Equilíbrio teórico b*(x) = (N-1)(N+2)/(2N) * x"
        else:
            label = f"Equilíbrio teórico {bid_formula}"
        plt.plot(v_grid_cpu, b_star, "--", color="black", label=label)
    xlabel = "Sinal x" if is_common_value else "Valor v"
    plt.xlabel(xlabel)
    plt.ylabel("Lance b")
    plt.legend()
    plt.title(f"Estratégia de lance aprendida vs. equilíbrio ({label_suffix})")
    plt.tight_layout()
    plt.savefig(f"bid_function{tag}.png", dpi=150)
    plt.close()

    critic = trainer.critic_for(critic_agent)
    if common_value_gaussian is not None:
        prior_mean, prior_std, noise_std = common_value_gaussian
        signal_std = (prior_std ** 2 + noise_std ** 2) ** 0.5
        b_grid = torch.linspace(0, prior_mean + 4 * signal_std, 300, device=trainer.device)
    else:
        b_grid = torch.linspace(0, 1, 300, device=trainer.device)
    with torch.no_grad():
        learned_p_win = critic.predict_win_prob(b_grid).cpu()
        if risk_aversion is not None:
            theoretical_p_win = theoretical_win_probability_risk_averse(
                b_grid, critic_agent, tuple(risk_aversion)).cpu()
        elif payment_share is not None:
            theoretical_p_win = theoretical_win_probability_lambda(b_grid, n_agents, payment_share).cpu()
        elif common_value_wallet_game:
            theoretical_p_win = theoretical_win_probability_wallet_game(b_grid, n_agents).cpu()
        elif not is_common_value:
            theoretical_p_win = theoretical_win_probability(b_grid, n_agents, trainer.auction_type).cpu()

    plt.figure()
    plt.plot(b_grid.cpu(), learned_p_win, label=f"Crítico aprendido ({trainer.critic_type})")
    if not no_theoretical_bid and (not is_common_value or common_value_wallet_game):
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

    convergence_part = "" if trainer.common_value_gaussian is not None else f"convergence{tag}.png, "
    print(f"Figuras salvas em {convergence_part}payoff{tag}.png, "
          f"bid_function{tag}.png e critic{tag}.png")
