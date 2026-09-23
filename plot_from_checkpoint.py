import argparse

from fictitious_play.plotting import save_all_plots
from fictitious_play.train import FictitiousPlayTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n-agents", type=int, required=True)
    parser.add_argument("--critic-agent", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = FictitiousPlayTrainer(n_agents=args.n_agents, n_rounds=0)
    round_idx = trainer.load_checkpoint(args.checkpoint)
    print(f"Checkpoint carregado (rodada {round_idx}, {len(trainer.history)} rodadas no histórico)")

    save_all_plots(trainer, trainer.history, args.n_agents, tag=args.tag, critic_agent=args.critic_agent)


if __name__ == "__main__":
    main()
