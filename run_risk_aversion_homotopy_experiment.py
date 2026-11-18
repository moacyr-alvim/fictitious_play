import argparse
import os

from fictitious_play.plotting import save_all_plots
from fictitious_play.train import FictitiousPlayTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho0-schedule", type=float, nargs="+",
                         default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
                         help="agent 0's CRRA risk-aversion coefficient at each stage, in "
                              "order. Paired elementwise with --rho1-schedule. Stage 0 "
                              "(rho=1.0, risk-neutral) is where the symmetric closed-form "
                              "first-price equilibrium applies, so training starts from the "
                              "easiest point on the continuum and is warm-started onward from "
                              "there, the same way run_homotopy_experiment.py ramps "
                              "payment_share from 0.")
    parser.add_argument("--rho1-schedule", type=float, nargs="+",
                         default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         help="agent 1's CRRA risk-aversion coefficient at each stage, "
                              "paired elementwise with --rho0-schedule. Defaults to staying "
                              "risk-neutral throughout (only agent 0 is ramped).")
    parser.add_argument("--rounds-per-stage", type=int, default=150,
                         help="Safety cap on rounds per stage; each stage still stops "
                              "early once the usual max-diff/payoff criteria are met.")
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=8_000)
    parser.add_argument("--belief-decay", type=float, default=0.95,
                         help="Fraction of each agent's belief buffer kept per update. Higher "
                              "than the plain run_experiment.py default (0.9) — longer memory "
                              "damps the best-response cycling a persistently warm-started "
                              "critic (critic-type nn) is prone to (see "
                              "docs/findings_mixture_beta_critic.md).")
    parser.add_argument("--max-diff-threshold", type=float, default=0.02)
    parser.add_argument("--payoff-rel-tol", type=float, default=-1.0,
                         help="No single theoretical payoff applies under risk_aversion (see "
                              "FictitiousPlayTrainer.run's theo_str branch), so this defaults "
                              "to disabled (-1) rather than the plain-auction default.")
    parser.add_argument("--payoff-window", type=int, default=10)
    parser.add_argument("--critic-n-quantiles", type=int, default=1000)
    parser.add_argument("--critic-type", type=str, default="nn",
                         choices=["spline", "beta", "nn", "beta_mixture"])
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--critic-agent", type=int, default=0)
    parser.add_argument("--tag", type=str, default="_risk_aversion_homotopy")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.rho0_schedule) != len(args.rho1_schedule):
        raise ValueError("--rho0-schedule and --rho1-schedule must have the same length")
    schedule = list(zip(args.rho0_schedule, args.rho1_schedule))

    checkpoint_path = f"checkpoints/checkpoint{args.tag}.pt"
    trainer = FictitiousPlayTrainer(
        n_agents=2, n_rounds=args.rounds_per_stage,
        sample_size=args.sample_size, patience=args.patience, max_steps=args.max_steps,
        belief_decay=args.belief_decay, critic_n_quantiles=args.critic_n_quantiles,
        critic_type=args.critic_type, auction_type="first_price",
        risk_aversion=list(schedule[0]),
        max_diff_threshold=args.max_diff_threshold if args.max_diff_threshold >= 0 else None,
        payoff_rel_tol=args.payoff_rel_tol if args.payoff_rel_tol >= 0 else None,
        payoff_stability_window=args.payoff_window,
        checkpoint_path=checkpoint_path, checkpoint_every=args.checkpoint_every,
        seed=args.seed,
    )
    print(f"Device: {trainer.device}")

    start_stage_idx = 0
    if args.resume and os.path.exists(checkpoint_path):
        trainer.load_checkpoint()
        restored_rho = tuple(trainer.risk_aversion)
        matches = [i for i, rho in enumerate(schedule) if tuple(rho) == restored_rho]
        if not matches:
            raise ValueError(f"Checkpoint's risk_aversion={restored_rho} isn't in "
                              f"--rho0-schedule/--rho1-schedule — can't tell which stage to "
                              f"resume from. Pass the same schedule used to produce it.")
        start_stage_idx = matches[0]
        stage_start = trainer.growth_rounds[-1] if trainer.growth_rounds else 0
        elapsed_in_stage = len(trainer.history) - stage_start
        remaining = args.rounds_per_stage - elapsed_in_stage
        print(f"Retomado de {checkpoint_path}: rodada {len(trainer.history)}, "
              f"estagio {start_stage_idx} (risk_aversion={restored_rho}), "
              f"{elapsed_in_stage} rodadas ja neste estagio (orcamento {args.rounds_per_stage}), "
              f"faltam {max(remaining, 0)}")
        if remaining > 0:
            trainer.n_rounds = remaining
            trainer.run()
        trainer.n_rounds = args.rounds_per_stage
        start_stage_idx += 1

    for stage_idx in range(start_stage_idx, len(schedule)):
        rho = list(schedule[stage_idx])
        if stage_idx > 0:
            print(f"--- Avancando para risk_aversion={rho} (rodada {len(trainer.history)}) ---")
            trainer.set_risk_aversion(rho)
        print(f"Estagio {stage_idx}: risk_aversion={rho}")
        trainer.run()

    save_all_plots(trainer, trainer.history, trainer.n_agents, tag=args.tag,
                    critic_agent=args.critic_agent, growth_rounds=trainer.growth_rounds)


if __name__ == "__main__":
    main()
