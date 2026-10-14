import argparse
import os

from fictitious_play.plotting import save_all_plots
from fictitious_play.train import FictitiousPlayTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--lambdas", type=float, nargs="+",
                         default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         help="payment_share values to step through, in order. 0.0 = "
                              "first-price, 1.0 = all-pay. Each stage warm-starts from "
                              "the previous one's converged weights (no network surgery "
                              "needed, unlike growing hidden_size: payment_share only "
                              "changes the loss/payoff formulas).")
    parser.add_argument("--rounds-per-stage", type=int, default=300,
                         help="Safety cap on rounds per stage; each stage still stops "
                              "early once the usual max-diff/payoff criteria are met.")
    parser.add_argument("--sample-size", type=int, default=300_000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=20_000)
    parser.add_argument("--belief-decay", type=float, default=0.9)
    parser.add_argument("--max-diff-threshold", type=float, default=0.02)
    parser.add_argument("--payoff-rel-tol", type=float, default=0.08)
    parser.add_argument("--payoff-window", type=int, default=10)
    parser.add_argument("--critic-n-quantiles", type=int, default=1000)
    parser.add_argument("--payoff-weight", type=float, default=1.0,
                         help="Weight of the usual critic-based expected-payoff loss. Set to 0 "
                              "(together with --pinn-weight 0) for a pure PINN run driven only "
                              "by --pinn-opponent-weight.")
    parser.add_argument("--pinn-weight", type=float, default=0.0,
                         help="Weight of the critic-derivative-based PINN regularizer (see "
                              "fictitious_play.train.foc_residual). 0 disables it.")
    parser.add_argument("--pinn-opponent-weight", type=float, default=0.0,
                         help="Weight of the opponent-actor-based PINN residual (exact, no "
                              "critic — see fictitious_play.train.foc_residual_opponent). "
                              "Only n_agents=2.")
    parser.add_argument("--opponent-ema-decay", type=float, default=0.9,
                         help="Decay of the EMA shadow actor that foc_residual_opponent "
                              "inverts against instead of the opponent's literal live actor — "
                              "gives the opponent-based PINN term the same kind of memory "
                              "belief_decay gives the critic. Ignored if "
                              "--opponent-ema-decay-start/--opponent-ema-decay-end are given. "
                              "Only matters if --pinn-opponent-weight > 0.")
    parser.add_argument("--opponent-ema-decay-start", type=float, default=None,
                         help="If set together with --opponent-ema-decay-end, the EMA decay "
                              "follows a linear schedule in the current payment_share instead "
                              "of a fixed value: this is the decay at payment_share=0.")
    parser.add_argument("--opponent-ema-decay-end", type=float, default=None,
                         help="Decay at payment_share=1, paired with --opponent-ema-decay-start.")
    parser.add_argument("--checkpoint-every", type=int, default=20,
                         help="Save a checkpoint every this many rounds (0 disables "
                              "periodic checkpointing; a final checkpoint is always saved).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-newton-predictor", action="store_true",
                         help="Before each payment_share change, take a damped "
                              "block-diagonal Newton predictor step (see "
                              "FictitiousPlayTrainer.newton_predict_step) to warm-start "
                              "closer to the new stage's equilibrium.")
    parser.add_argument("--newton-damping", type=float, default=1e-2)
    parser.add_argument("--critic-agent", type=int, default=0,
                         help="Which agent's critic to plot.")
    parser.add_argument("--tag", type=str, default="_homotopy",
                         help="Suffix appended to output figure/checkpoint filenames.")
    parser.add_argument("--resume", action="store_true",
                         help="If a checkpoint already exists at the target path, load it "
                              "and continue from its stage/round instead of starting over.")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = f"checkpoints/checkpoint{args.tag}.pt"
    opponent_ema_decay_schedule = None
    if args.opponent_ema_decay_start is not None and args.opponent_ema_decay_end is not None:
        opponent_ema_decay_schedule = (args.opponent_ema_decay_start, args.opponent_ema_decay_end)
    trainer = FictitiousPlayTrainer(
        n_agents=args.n_agents, n_rounds=args.rounds_per_stage,
        sample_size=args.sample_size, patience=args.patience, max_steps=args.max_steps,
        belief_decay=args.belief_decay,
        critic_n_quantiles=args.critic_n_quantiles, payment_share=args.lambdas[0],
        payoff_weight=args.payoff_weight, pinn_weight=args.pinn_weight,
        pinn_opponent_weight=args.pinn_opponent_weight,
        opponent_ema_decay=args.opponent_ema_decay,
        opponent_ema_decay_schedule=opponent_ema_decay_schedule,
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
        start_stage_idx = args.lambdas.index(trainer.payment_share)
        print(f"Retomado de {checkpoint_path}: rodada {len(trainer.history)}, "
              f"estagio {start_stage_idx} (payment_share={trainer.payment_share:.2f})")
        trainer.run()
        start_stage_idx += 1

    for stage_idx in range(start_stage_idx, len(args.lambdas)):
        lam = args.lambdas[stage_idx]
        if stage_idx > 0:
            print(f"--- Avancando para payment_share={lam:.2f} (rodada {len(trainer.history)}) ---")
            if args.use_newton_predictor:
                trainer.newton_predict_step(lam, damping=args.newton_damping)
            trainer.set_payment_share(lam)
        print(f"Estagio {stage_idx}: payment_share={lam:.2f}")
        trainer.run()

    save_all_plots(trainer, trainer.history, args.n_agents, tag=args.tag,
                    critic_agent=args.critic_agent, growth_rounds=trainer.growth_rounds)


if __name__ == "__main__":
    main()
