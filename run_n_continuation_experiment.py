import argparse
import os

from fictitious_play.plotting import save_all_plots
from fictitious_play.train import FictitiousPlayTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents-schedule", type=int, nargs="+", default=[2, 3, 4, 5],
                         help="Agent counts to step through, in order. The first value is "
                              "where lambda is ramped from 0 to 1 (see --lambdas); every "
                              "later value just adds that many agents (warm-started as a "
                              "copy of an existing one, see FictitiousPlayTrainer.add_agents) "
                              "and keeps training at the final lambda, no lambda reset.")
    parser.add_argument("--lambdas", type=float, nargs="+",
                         default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         help="payment_share schedule run once, at the first (smallest) "
                              "agent count in --n-agents-schedule.")
    parser.add_argument("--rounds-per-lambda-stage", type=int, default=150)
    parser.add_argument("--rounds-per-n-stage", type=int, default=150,
                         help="Safety cap on rounds run after growing to a new agent count "
                              "(before growing again); each stage still stops early once the "
                              "usual max-diff/payoff criteria are met.")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=5_000)
    parser.add_argument("--belief-decay", type=float, default=0.98)
    parser.add_argument("--max-diff-threshold", type=float, default=0.02)
    parser.add_argument("--payoff-rel-tol", type=float, default=0.08)
    parser.add_argument("--payoff-window", type=int, default=10)
    parser.add_argument("--critic-n-quantiles", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critic-agent", type=int, default=0)
    parser.add_argument("--tag", type=str, default="_n_continuation")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = f"checkpoints/checkpoint{args.tag}.pt"
    start_n = args.n_agents_schedule[0]
    trainer = FictitiousPlayTrainer(
        n_agents=start_n, n_rounds=args.rounds_per_lambda_stage,
        sample_size=args.sample_size, patience=args.patience, max_steps=args.max_steps,
        belief_decay=args.belief_decay, critic_n_quantiles=args.critic_n_quantiles,
        payment_share=args.lambdas[0],
        # Pure critic + belief buffer ("continuação analítica") — no PINN terms.
        payoff_weight=1.0, pinn_weight=0.0, pinn_opponent_weight=0.0,
        max_diff_threshold=args.max_diff_threshold if args.max_diff_threshold >= 0 else None,
        payoff_rel_tol=args.payoff_rel_tol if args.payoff_rel_tol >= 0 else None,
        payoff_stability_window=args.payoff_window,
        checkpoint_path=checkpoint_path, checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    print(f"Device: {trainer.device}")

    lambda_start_idx = 0
    n_start_idx = 0
    if args.resume and os.path.exists(checkpoint_path):
        trainer.load_checkpoint()
        print(f"Retomado de {checkpoint_path}: rodada {len(trainer.history)}, "
              f"n_agents={trainer.n_agents}, payment_share={trainer.payment_share:.2f}")
        if trainer.n_agents == start_n and trainer.payment_share in args.lambdas:
            lambda_start_idx = args.lambdas.index(trainer.payment_share) + 1
            trainer.run()
        else:
            lambda_start_idx = len(args.lambdas)  # lambda ramp already finished
            n_start_idx = args.n_agents_schedule.index(trainer.n_agents) + 1
            trainer.run()

    trainer.n_rounds = args.rounds_per_lambda_stage
    for stage_idx in range(lambda_start_idx, len(args.lambdas)):
        lam = args.lambdas[stage_idx]
        if stage_idx > 0:
            print(f"--- lambda={lam:.2f} (rodada {len(trainer.history)}, n_agents={trainer.n_agents}) ---")
            trainer.set_payment_share(lam)
        print(f"Estagio lambda {stage_idx}: payment_share={lam:.2f}")
        trainer.run()

    trainer.n_rounds = args.rounds_per_n_stage
    for n_idx in range(max(n_start_idx, 1), len(args.n_agents_schedule)):
        n = args.n_agents_schedule[n_idx]
        additional = n - trainer.n_agents
        print(f"--- Crescendo para n_agents={n} (rodada {len(trainer.history)}) ---")
        trainer.add_agents(additional)
        print(f"Estagio n_agents {n_idx}: n_agents={n}")
        trainer.run()

    save_all_plots(trainer, trainer.history, trainer.n_agents, tag=args.tag,
                    critic_agent=args.critic_agent, growth_rounds=trainer.growth_rounds)


if __name__ == "__main__":
    main()
