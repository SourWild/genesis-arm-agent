# Genesis Arm Agent - Project Context

## 硬件与环境
- MacBook Air M4, 24GB RAM, macOS
- Python 3.11 通过 uv 管理
- 无 CUDA，只有 Apple Silicon MPS

## 技术栈约束（重要，不要偏离）
- Genesis: 只用 rigid body solver，禁用 MPM / soft body / fluid / differentiable
- Genesis backend: gs.metal 或 gs.cpu，不要写 gs.cuda
- Tensor dtype: 全程 float32，禁止 float64
- 单环境仿真（n_envs=1），不做并行
- 场景：Franka 单臂 + 桌面 pick-place，不要 humanoid、不要复杂场景

## 项目结构

```
genesis-arm-agent/
├── CLAUDE.md          # 本文件
├── README.md          # 用户文档
├── pyproject.toml     # uv 管理
├── .env                # API keys（gitignore）
├── main.py            # 入口
├── sim/
│   ├── scene.py        # Genesis 场景搭建
│   └── controller.py   # Franka 控制封装
├── agent/
│   ├── tools.py         # 4 个 tool 定义
│   ├── harness.py       # LLM loop
│   └── prompts.py       # system prompt
├── utils/
│   ├── voice.py         # macOS say 播报
│   └── logging.py       # trace 存 jsonl
└── logs/                # 运行日志
```

## LLM 配置
- 首选：Kimi K2 (moonshot-v1-8k 或最新版本)，通过 OpenAI 兼容接口调用
- 备选：Claude API (claude-sonnet-4-6)
- API key 从 .env 读取
- 每次调用有重试（3 次，指数退避）

## Tool 接口规范（4 个 tool 都遵守）
所有 tool 返回统一格式：
```python
{
    "success": bool,
    "message": str,        # 人类可读的执行结果
    "data": dict | None    # 结构化数据
}
```

## 坐标系
- 世界坐标系：z 向上，桌面 z=0
- 单位：米
- Franka base 在原点，工作区在 x ∈ [0.3, 0.7], y ∈ [-0.3, 0.3]

## 编码风格
- 类型标注必须写
- 关键函数写 docstring（参数含义、单位）
- 不要写单元测试（这是原型项目）
- 每个 Phase 结束跑一次 smoke test 而已

## 分阶段规则
- 每完成一个 Phase 停下来跑通再继续
- 每个 Phase 结束 git commit，message 用 "Phase N: xxx"
- 遇到 Mac 特定的报错（dtype、metal backend），优先查 Genesis GitHub issues

## 禁止事项
- 不要装 CUDA 相关的包
- 不要用 float64 或 double precision
- 不要引入训练代码 / RL / policy learning
- 不要用 ROS
- 不要为了完整性写测试框架
