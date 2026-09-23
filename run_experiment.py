import argparse

from fictitious_play.plotting import save_all_plots
from fictitious_play.train import FictitiousPlayTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--n-rounds", type=int, default=200)
    parser.add_argument("--sample-size", type=int, default=300_000)
    parser.add_argument("--belief-decay", type=float, default=0.9,
                         help="Fraction of each agent's belief buffer kept per update "
                              "(0 = no memory / latest snapshot only).")
    parser.add_argument("--max-diff-threshold", type=float, default=0.02,
                         help="Stop early once every agent's maximum absolute deviation from "
                              "the equilibrium bid function (over the eval grid) drops to or "
                              "below this. Use a negative value to disable.")
    parser.add_argument("--payoff-rel-tol", type=float, default=0.15,
                         help="Stop early once every agent's expected payoff changes by less "
                              "than this fraction between consecutive rounds, sustained for "
                              "--payoff-window rounds. Use a negative value to disable.")
    parser.add_argument("--payoff-window", type=int, default=10)
    parser.add_argument("--critic-n-quantiles", type=int, default=400,
                         help="Number of quantile points the spline critic is fit on.")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                         help="Save a checkpoint every this many rounds (0 disables periodic "
                              "checkpointing; a final checkpoint is always saved).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critic-agent", type=int, default=0,
                         help="Which agent's critic to plot.")
    parser.add_argument("--tag", type=str, default="",
                         help="Suffix appended to output figure/checkpoint filenames.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = f"checkpoints/checkpoint{args.tag}.pt"
    trainer = FictitiousPlayTrainer(
        n_agents=args.n_agents, n_rounds=args.n_rounds,
        sample_size=args.sample_size, belief_decay=args.belief_decay,
        critic_n_quantiles=args.critic_n_quantiles,
        max_diff_threshold=args.max_diff_threshold if args.max_diff_threshold >= 0 else None,
        payoff_rel_tol=args.payoff_rel_tol if args.payoff_rel_tol >= 0 else None,
        payoff_stability_window=args.payoff_window,
        checkpoint_path=checkpoint_path, checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    print(f"Device: {trainer.device}")
    history = trainer.run()

    save_all_plots(trainer, history, args.n_agents, tag=args.tag, critic_agent=args.critic_agent)


if __name__ == "__main__":
    main()
