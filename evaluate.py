"""Run N pick-place episodes through the full LLM harness and report the objective success rate.

Each trial uses a fresh randomly-laid-out scene (build_scene(seed=...)) with the same
instruction ("把红方块放到蓝色区域"). Success is judged independently via
agent.tools.check_task_success on the final scene state, not the LLM's own self-report —
an LLM can say "done" while being wrong, so run_episode's `ground_truth` field is authoritative.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from agent.harness import run_episode

INSTRUCTION = "把红方块放到蓝色区域"
BLOCK_COLOR = "red"
TARGET_COLOR = "blue"

LOGS_DIR = Path(__file__).resolve().parent / "logs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pick-place success rate over N random scenes")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()

    trials = []
    for i in range(args.n_trials):
        seed = args.seed_start + i
        print(f"\n=== trial {i + 1}/{args.n_trials} (seed={seed}) ===")
        result = run_episode(
            INSTRUCTION,
            max_steps=args.max_steps,
            seed=seed,
            block_color=BLOCK_COLOR,
            target_color=TARGET_COLOR,
        )
        ground_truth = result.get("ground_truth") or {}
        success = bool(ground_truth.get("task_success"))
        distance_mm = ground_truth.get("xy_distance_m", float("nan")) * 1000

        print(f"llm_success={result['success']} ground_truth_success={success} "
              f"xy_distance={distance_mm:.1f}mm steps_used={result['steps_used']}")
        print(f"trace: {result['trace_path']}")

        trials.append(
            {
                "seed": seed,
                "llm_success": result["success"],
                "ground_truth_success": success,
                "xy_distance_mm": distance_mm,
                "steps_used": result["steps_used"],
                "trace_path": result["trace_path"],
            }
        )

    n_success = sum(t["ground_truth_success"] for t in trials)
    success_rate = n_success / len(trials)

    print("\n=== summary ===")
    for t in trials:
        mark = "✓" if t["ground_truth_success"] else "✗"
        print(f"  seed={t['seed']:>3} {mark} dist={t['xy_distance_mm']:.1f}mm steps={t['steps_used']}")
    print(f"\nsuccess rate: {n_success}/{len(trials)} = {success_rate:.0%}")

    summary_path = LOGS_DIR / f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"instruction": INSTRUCTION, "success_rate": success_rate, "trials": trials}, ensure_ascii=False, indent=2)
    )
    print(f"summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
