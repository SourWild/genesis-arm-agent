# Genesis Arm Agent

用中文自然语言指挥一台仿真 Franka Panda 机械臂做桌面 pick-place（比如"把红方块放到蓝色区域"）。LLM（Kimi K2）通过 tool calling 驱动 [Genesis](https://genesis-embodied-ai.github.io/) 物理仿真里的机械臂，全程语音播报关键节点，执行过程记录成 trace 日志。

一个周末搭出来的原型项目，不追求生产级健壮性。

## 硬件与环境要求

- macOS，Apple Silicon（M 系列芯片，用 Metal 后端跑仿真，Mac 上的 `say` 命令做语音播报）
- Python 3.11（[uv](https://docs.astral.sh/uv/) 管理虚拟环境和依赖）
- Moonshot（Kimi）API key
- 无需 GPU/CUDA — Genesis 用 `gs.metal` 后端（rigid body only），PyTorch 走 MPS

## 装机步骤

```bash
git clone <this-repo>
cd genesis-arm-agent

# 装依赖（会自动创建 .venv，装 genesis-world / torch / openai 等）
uv sync

# 配置 API key
cp .env.example .env
# 编辑 .env，填入真实的 MOONSHOT_API_KEY
```

`.env` 需要三个变量（`.env.example` 里有模板）：

```
MOONSHOT_API_KEY=sk-xxxxxxxx
MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
MOONSHOT_MODEL=kimi-k2.6
```

`MOONSHOT_MODEL` 要填你账号里实际开通的模型名 —— Moonshot 的模型名变动比较快，登录 [platform.moonshot.cn](https://platform.moonshot.cn) 控制台确认一下再填，避免遇到 404。

## Demo 命令

```bash
# Phase 1 冒烟测试：Franka 移动 + 开合 gripper，会弹出 Genesis viewer 窗口
uv run python sanity_check.py

# 单次指令，跑完整 LLM pick-place 流程
uv run python main.py --instruction "把红方块放到蓝色区域"

# 可选参数
uv run python main.py --instruction "把绿方块放到红色区域" --seed 3 --show-viewer

# 跑 N 次随机场景，统计客观成功率（不信任 LLM 自己说的"完成"，用 check_task_success 复核）
uv run python evaluate.py --n-trials 10

# 录一段 30 秒的屏幕+系统音频 demo（需要先装 BlackHole，见脚本内注释）
./record_demo.sh
```

每次 `main.py` / `evaluate.py` 运行都会在 `logs/` 下生成一份 `{timestamp}.jsonl` trace，记录每一步的 LLM 请求/响应、tool 调用、tool 返回结果，方便事后复盘。

## 架构（文字版）

```
用户中文指令 ("把红方块放到蓝色区域")
        │
        ▼
┌────────────────────┐        agent/prompts.py
│   LLM Planner       │◀────── system prompt：坐标系、5 个工具的语义、
│   (Kimi K2)          │        标准抓取/放置流程、失败重试指引
└─────────┬───────────┘
          │ tool call (JSON)
          ▼
┌────────────────────────────┐
│   Harness Loop               │  agent/harness.py
│   run_episode()，最多 20 步   │  - ToolExecutor 把 scene 绑定进 tool 调用
│                              │  - 每步写入 logs/*.jsonl trace
│                              │  - 语音播报关键节点 (utils/voice.py)
│                              │  - API/格式错误重试 3 次
└─────────┬────────────────────┘
          │ dispatch(tool_name, args)
          ▼
┌──────────────────────────────────────────────────┐
│   5 个 Tool                agent/tools.py          │
│   get_scene_state / get_camera_image / move_ee /   │
│   control_gripper / check_task_success             │
│   统一返回 {success, message, data}                 │
└─────────┬──────────────────────────────────────────┘
          │ 调用
          ▼
┌────────────────────┐
│   Controller         │  sim/controller.py
│   IK + PD 位置控制    │  move_ee_smooth（5cm/waypoint 插值）
│                      │  pick() / place()（固化的抓取/放置序列）
└─────────┬────────────┘
          │
          ▼
┌────────────────────┐
│   Genesis 仿真        │  sim/scene.py
│   Franka Panda +     │  gs.metal 后端，rigid body only，float32
│   3 彩色方块 +        │  build_scene() / reset_scene()
│   3 彩色目标区域 +     │
│   top/side 相机       │
└──────────────────────┘
```

## 开发笔记：踩过的坑（面试可以聊的技术细节）

这个项目里最有技术含量的部分不是搭 LLM tool-calling 循环，而是让机械臂物理上真的能抓稳东西。踩过两个非平凡的坑，都在 `sim/controller.py` / `sim/scene.py` 里：

1. **姿态漂移导致撞飞方块**：一开始 `move_ee` 的 IK 调用只给了目标位置(`pos`)，没约束姿态(`quat`)。夹爪下降时朝向随意漂，直接把方块撞飞，甚至把机械臂送到工作区外的诡异姿态（末端误差一度到 1.6 米）。
   修法：Franka 的 home pose 下夹爪本来就是垂直朝下的（用 `gs.quat_to_R` 查旋转矩阵第三列，发现局部 Z 轴正好对齐世界 -Z），把这个姿态存成 `scene.grasp_quat`，全程 IK 都锁住它。

2. **抓取点坐标系搞错，这个才是真正的根因**：`move_ee` 一直把 `hand` link（手腕法兰盘）的坐标当成"末端位置"。但手指从手腕往下还有约 11cm（第一次估算错成只有 5.8cm，因为查的是手指 link 的 pivot 原点，不是碰撞几何体的最低点）。结果只要一下降到方块高度，指尖其实早就怼进桌面了，物理引擎顶着不让继续下降，也顶得夹爪合不拢。
   排查方法：打印实际接触点（`entity.get_contacts()`），发现指尖和桌面在 z≈0 处有穿透力；再用 `geom.get_AABB()` 量出手指碰撞几何体的真实最低点，才拿到准确的偏移量。修完之后 `get_ee_pos()`/`move_ee` 操作的坐标就是**指尖抓取点**而不是手腕，`sim/scene.py` 的 `grasp_point_offset` 就是这个测出来的偏移。

一个附带的教训：`control_gripper` 一开始默认只给 15 步收敛，而手指全程 4cm 行程在力矩受限的情况下根本走不完——调试时误以为是"抓空"，其实是给的时间不够。

## 已知限制

- **单环境、纯仿真**：`n_envs=1`，没有并行化；没有接真实机械臂，只在 Genesis 仿真里闭环。
- **固定俯视抓取姿态**：夹爪全程保持垂直向下（`scene.grasp_quat`），不支持斜向/侧向抓取，无法处理需要重新定向才能抓取的物体。
- **抓取判定比较粗糙**：靠手指间距阈值判断"抓空"，没有真正的力/触觉反馈，边缘情况可能误判。
- **`move_ee` 的失败阈值是经验值**：IK 不可达（1cm）、位置偏差过大（2cm）、抓空（1cm）这几个数字是手工调出来的，没有做大规模统计校准。
- **只有 3 种固定颜色**（红/绿/蓝）的方块和目标区域，不支持任意数量、形状、颜色的物体。
- **一次只处理单个 pick-place 目标**：没有专门测试过"把红方块放蓝色区域，再把绿方块放红色区域"这种一句话里的复合指令（LLM 理论上可以多轮工具调用做到，但没验证过）。
- **成功判据是"宽松"的**：只看方块中心到目标区域中心的水平距离（<4cm），不检查方块朝向、是否被碰倒、是否叠在别的方块上。
- **语音播报只在 macOS 有效**：`utils/voice.py` 调用系统 `say` 命令，非 macOS 环境会静默跳过（不报错，但也不出声）。
- **评测样本量小**：`evaluate.py` 默认只跑 10 次，成功率数字统计意义有限，仅作为回归检查，不是严谨的基准测试。
- **Genesis 单进程限制**：`gs.init()` 每个进程只能调一次，`evaluate.py` 连续跑多个 episode 时用 `gs.destroy()` 重置绕过这个限制，没有做过长时间连续运行的压力测试，不排除内存/状态累积的问题。
- **API 调用是同步阻塞的**：没有用 streaming，LLM 响应慢的时候整个循环会卡住，用户看不到中间过程。
