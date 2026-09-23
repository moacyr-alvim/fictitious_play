import argparse

import matplotlib.pyplot as plt
import torch

from fictitious_play.benchmark import equilibrium_bid, theoretical_win_probability
from fictitious_play.train import FictitiousPlayTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--n-rounds", type=int, default=200)
    parser.add_argument("--sample-size", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critic-agent", type=int, default=0,
                         help="Which agent's critic to plot.")
    parser.add_argument("--tag", type=str, default="",
                         help="Suffix appended to output figure filenames.")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = FictitiousPlayTrainer(
        n_agents=args.n_agents, n_rounds=args.n_rounds,
        sample_size=args.sample_size, seed=args.seed,
    )
    print(f"Device: {trainer.device}")
    history = trainer.run()

    rounds = [h["round"] for h in history]
    mse_per_agent = list(zip(*(h["mse"] for h in history)))

    plt.figure()
    for i, mse_series in enumerate(mse_per_agent):
        plt.plot(rounds, mse_series, label=f"Agente {i}")
    plt.yscale("log")
    plt.xlabel("Rodada")
    plt.ylabel("MSE vs. equilíbrio teórico (escala log)")
    plt.legend()
    plt.title(f"Convergência ao equilíbrio de Bayes-Nash (N={args.n_agents})")
    plt.tight_layout()
    plt.savefig(f"convergence{args.tag}.png", dpi=150)

    v_grid = torch.linspace(0, 1, 200, device=trainer.device).unsqueeze(1)
    with torch.no_grad():
        b_star = equilibrium_bid(v_grid, args.n_agents).squeeze(1).cpu()
        v_grid_cpu = v_grid.squeeze(1).cpu()
        bids = [trainer.agents[i](v_grid).squeeze(1).cpu() for i in range(args.n_agents)]

    plt.figure()
    for i, b in enumerate(bids):
        plt.plot(v_grid_cpu, b, label=f"Agente {i} (aprendido)")
    plt.plot(v_grid_cpu, b_star, "--", color="black",
              label=f"Equilíbrio teórico b*(v) = v*(N-1)/N")
    plt.xlabel("Valor v")
    plt.ylabel("Lance b")
    plt.legend()
    plt.title("Estratégia de lance aprendida vs. equilíbrio")
    plt.tight_layout()
    plt.savefig(f"bid_function{args.tag}.png", dpi=150)

    critic = trainer.critic_for(args.critic_agent)
    b_grid = torch.linspace(0, 1, 300, device=trainer.device)
    with torch.no_grad():
        learned_p_win = critic.predict_win_prob(b_grid).cpu()
        theoretical_p_win = theoretical_win_probability(b_grid, args.n_agents).cpu()

    plt.figure()
    plt.plot(b_grid.cpu(), learned_p_win, label="Crítico aprendido (KDE)")
    plt.plot(b_grid.cpu(), theoretical_p_win, "--", color="black", label="P(vitória|b) teórica")
    plt.xlabel("Lance b")
    plt.ylabel("P(vitória | b)")
    plt.legend()
    plt.title(f"Crítico do agente {args.critic_agent} (N={args.n_agents})")
    plt.tight_layout()
    plt.savefig(f"critic{args.tag}.png", dpi=150)

    print(f"Figuras salvas em convergence{args.tag}.png, bid_function{args.tag}.png e critic{args.tag}.png")


if __name__ == "__main__":
    main()
