import matplotlib.pyplot as plt
import torch

from fictitious_play.benchmark import equilibrium_bid
from fictitious_play.train import FictitiousPlayTrainer


def main():
    trainer = FictitiousPlayTrainer(n_rounds=200, sample_size=300_000, seed=0)
    history = trainer.run()

    rounds = [h["round"] for h in history]
    mse0 = [h["mse_agent_0"] for h in history]
    mse1 = [h["mse_agent_1"] for h in history]

    plt.figure()
    plt.plot(rounds, mse0, label="Agente 0")
    plt.plot(rounds, mse1, label="Agente 1")
    plt.yscale("log")
    plt.xlabel("Rodada")
    plt.ylabel("MSE vs. equilíbrio teórico (escala log)")
    plt.legend()
    plt.title("Convergência ao equilíbrio de Bayes-Nash (b*(v) = v/2)")
    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)

    v_grid = torch.linspace(0, 1, 200).unsqueeze(1)
    with torch.no_grad():
        b0 = trainer.agents[0](v_grid).squeeze(1)
        b1 = trainer.agents[1](v_grid).squeeze(1)
        b_star = equilibrium_bid(v_grid).squeeze(1)

    plt.figure()
    plt.plot(v_grid.squeeze(1), b0, label="Agente 0 (aprendido)")
    plt.plot(v_grid.squeeze(1), b1, label="Agente 1 (aprendido)")
    plt.plot(v_grid.squeeze(1), b_star, "--", label="Equilíbrio teórico b*(v) = v/2")
    plt.xlabel("Valor v")
    plt.ylabel("Lance b")
    plt.legend()
    plt.title("Estratégia de lance aprendida vs. equilíbrio")
    plt.tight_layout()
    plt.savefig("bid_function.png", dpi=150)

    print("Figuras salvas em convergence.png e bid_function.png")


if __name__ == "__main__":
    main()
