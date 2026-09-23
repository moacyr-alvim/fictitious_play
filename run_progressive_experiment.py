import argparse

from fictitious_play.plotting import save_all_plots
from fictitious_play.train import FictitiousPlayTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--stages", type=int, nargs="+", default=[0, 1, 4, 16, 32],
                         help="Hidden-layer sizes to grow the actor through, in order. "
                              "0 = no hidden layer (a constant shading factor).")
    parser.add_argument("--rounds-per-stage", type=int, default=300,
                         help="Safety cap on rounds per stage; each stage still stops early "
                              "once the usual max-diff/payoff criteria are met.")
    parser.add_argument("--sample-size", type=int, default=300_000)
    parser.add_argument("--belief-decay", type=float, default=0.8)
    parser.add_argument("--max-diff-threshold", type=float, default=0.02)
    parser.add_argument("--payoff-rel-tol", type=float, default=0.08)
    parser.add_argument("--payoff-window", type=int, default=10)
    parser.add_argument("--critic-n-quantiles", type=int, default=1000)
    parser.add_argument("--freeze-mode", type=str, default="none",
                         choices=["none", "permanent", "temporary"],
                         help="What happens to previously-trained parameters when the actor "
                              "grows: keep training them jointly (none), lock them forever "
                              "(permanent), or lock them for --unfreeze-after-rounds rounds "
                              "then resume joint training (temporary).")
    parser.add_argument("--unfreeze-after-rounds", type=int, default=20,
                         help="Only used with --freeze-mode temporary.")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                         help="Save a checkpoint every this many rounds (0 disables periodic "
                              "checkpointing; a final checkpoint is always saved).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critic-agent", type=int, default=0,
                         help="Which agent's critic to plot.")
    parser.add_argument("--tag", type=str, default="_progressive",
                         help="Suffix appended to output figure/checkpoint filenames.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = f"checkpoints/checkpoint{args.tag}.pt"
    trainer = FictitiousPlayTrainer(
        n_agents=args.n_agents, n_rounds=args.rounds_per_stage,
        sample_size=args.sample_size, belief_decay=args.belief_decay,
        critic_n_quantiles=args.critic_n_quantiles, actor_hidden_size=args.stages[0],
        freeze_mode=args.freeze_mode, unfreeze_after_rounds=args.unfreeze_after_rounds,
        max_diff_threshold=args.max_diff_threshold if args.max_diff_threshold >= 0 else None,
        payoff_rel_tol=args.payoff_rel_tol if args.payoff_rel_tol >= 0 else None,
        payoff_stability_window=args.payoff_window,
        checkpoint_path=checkpoint_path, checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    print(f"Device: {trainer.device}")

    for stage_idx, hidden_size in enumerate(args.stages):
        if stage_idx > 0:
            print(f"--- Crescendo para hidden_size={hidden_size} (rodada {len(trainer.history)}) ---")
            trainer.grow_agents(hidden_size)
        print(f"Estagio {stage_idx}: hidden_size={hidden_size}")
        trainer.run()

    save_all_plots(trainer, trainer.history, args.n_agents, tag=args.tag,
                    critic_agent=args.critic_agent, growth_rounds=trainer.growth_rounds)


if __name__ == "__main__":
    main()
