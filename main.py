"""Entry point: python main.py --instruction "把红方块放到蓝色区域"."""

import argparse

from agent.harness import run_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Genesis Arm Agent - Franka pick-place LLM agent")
    parser.add_argument("--instruction", required=True, help="中文自然语言指令")
    parser.add_argument("--max-steps", type=int, default=20, help="LLM tool-calling 循环最大步数")
    parser.add_argument("--seed", type=int, default=None, help="场景随机布局种子")
    parser.add_argument("--show-viewer", action="store_true", help="是否弹出 Genesis viewer 窗口")
    args = parser.parse_args()

    result = run_episode(
        args.instruction,
        max_steps=args.max_steps,
        seed=args.seed,
        show_viewer=args.show_viewer,
    )

    status = "成功" if result["success"] else "未完成"
    print(f"\n[{status}] {result['message']}")
    print(f"steps_used: {result['steps_used']}")
    print(f"trace: {result['trace_path']}")


if __name__ == "__main__":
    main()
