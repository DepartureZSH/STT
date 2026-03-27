# STT (Schedule/Timetable) Project

## 项目概述

本项目是一个**课程时间表调度（Student/Subject Timetabling）优化系统**，基于 ITC 标准测试数据集，采用"图分割 + MIP 求解 + 多策略优化"的混合流水线架构，解决大规模大学排课问题。

---

## 整体流水线

```
原始 XML 实例
     │
     ▼
[DataLoader] 加载实例 → 判断规模
     │
     ├─ 小规模 ──────────────────────────────────────┐
     │                                              │
     └─ 大规模 → [GraphMapping] → [Segmentation]     │
                    (图构建)        (社区切割)        │
                        └─────── 若干小子问题 ────────┘
                                       │
                                       ▼
                            [GraphMapping] 对每个子问题建图
                                       │
                                       ▼
                                  [MIP Solver]
                                  (Gurobi 求解)
                                       │
                                       ▼
                              [valid / invalid 解列表]
                                       │
                                  [Optimizer]
                          ┌────────────┼────────────┐
                       [SOTA]       [MARL]       [Graph]
                          └────────────┴────────────┘
                                       │
                                       ▼
                               最终优化解 (XML)
```

---

## 目录结构与模块说明

### `data/`

| 路径 | 说明 |
|------|------|
| `data/source/early/` | ITC 早期标准测试实例（小/中规模） |
| `data/source/middle/` | 中等难度实例 |
| `data/source/late/` | 高难度/大规模实例 |
| `data/source/test/` | 评测专用实例，不用于训练 |
| `data/solutions/` | 每个实例对应的 98 条候选解（solution1~98），供优化器训练/测试 |

所有实例均为 ITC-2019 XML 格式，包含课程、学生、教室、时段、约束等完整信息。

---

### `src/data/` — 数据处理层

#### `DataReader.py`
- **职责**：解析 XML 实例文件，将课程、教室、时段、学生、约束等信息读入结构化的 Python 对象（或字典/DataFrame）。
- **输入**：`data/source/**/*.xml`
- **输出**：统一的实例数据对象（Instance）
- **关键功能**：
  - 解析 `<Courses>`、`<Rooms>`、`<Times>`、`<Students>`、`<Constraints>` 等 XML 节点
  - 构建课程-学生、课程-教室的关联关系
  - 区分硬约束（Hard）与软约束（Soft），并记录权重

#### `GraphMapping.py`
- **职责**：将调度实例映射为图结构，为社区检测和图优化器提供统一的图表示。
- **输入**：Instance 数据对象
- **输出**：图对象（节点=课程/班级，边=冲突/关联关系）
- **关键功能**：
  - 构建**冲突图**：若两门课程存在共同学生或教室资源冲突，则添加边
  - 节点特征编码：课程规模、约束类型、优先级等
  - 边权重：冲突强度、学生重叠数
  - 供 `Segmentation.py`（社区切割）和 `src/optimazier/GRAPH.py`（图优化）使用

#### `Segmentation.py`
- **职责**：对大规模实例的冲突图进行社区检测，将大问题分割为若干可被 MIP 独立求解的小子问题。
- **输入**：GraphMapping 输出的图对象
- **输出**：`[小课程组1, 小课程组2, ...]`，每组为一个独立的子问题
- **关键功能**：
  - 使用社区检测算法（如 Louvain / Leiden / Spectral Clustering）划分图社区
  - 控制每个子问题规模，使 Gurobi 在时间限制内可解
  - 处理跨社区约束（inter-community constraints），避免分割引入硬约束违反
  - 输出每个子图对应的子实例，格式与原始 Instance 兼容

---

### `src/optimazier/` — 优化器层

> 注意：目录名拼写为 `optimazier`（项目原始拼写），勿改。

#### `SOTA.py`
- **职责**：实现最先进（State-of-the-Art）的启发式/元启发式优化方法，作为基线。
- **输入**：MIP 初始解列表
- **输出**：优化后的解
- **关键功能**：
  - 实现模拟退火（SA）、禁忌搜索（Tabu Search）或大邻域搜索（LNS）等经典方法
  - 针对软约束违反进行局部改进（邻域移动：课程重分配、时段互换、教室互换）
  - 提供标准化的 `optimize(solutions) -> solution` 接口
  - 作为 MARL 和 Graph 方法的性能对比基线

#### `MARL.py`
- **职责**：实现多智能体强化学习（Multi-Agent Reinforcement Learning）优化器。
- **输入**：MIP 初始解列表（作为环境初始状态）
- **输出**：经 RL 策略改进的解
- **关键功能**：
  - 将每门课程（或每个时段分配）建模为一个智能体
  - 状态空间：当前排课方案、约束满足情况、邻居智能体状态
  - 动作空间：移动到新时段/教室
  - 奖励函数：软约束违反减少量（负惩罚）
  - 多智能体协调策略（集中训练、分布执行 CTDE）
  - 使用 `data/solutions/` 中的 98 条解进行训练/微调
  - 提供 `train(instances, solutions)` 和 `optimize(solution)` 接口

#### `GRAPH.py`
- **职责**：实现基于图神经网络（GNN）的图优化器。
- **输入**：GraphMapping 输出的图结构 + MIP 初始解
- **输出**：图模型预测的改进解
- **关键功能**：
  - 基于 GNN（如 GAT / GCN）对冲突图进行节点嵌入
  - 将排课优化建模为图上的节点分类或边预测任务
  - 利用图结构捕捉全局依赖，指导局部搜索
  - 与 `src/data/GraphMapping.py` 共享图表示，保持一致性
  - 提供 `train()` 和 `optimize()` 接口

---

### `src/solver/` — 求解器层

#### `gurobi.py`
- **职责**：将每个小课程组（子实例）编码为混合整数规划（MIP）模型，调用 Gurobi 求解。
- **输入**：小课程组（子实例数据）
- **输出**：可行解或不可行标记
- **关键功能**：
  - 定义决策变量：`x[c, r, t] = 1` 表示课程 `c` 在时段 `t` 分配到教室 `r`
  - 编码硬约束（必须满足）：无冲突、教室容量、教师可用性等
  - 编码软约束（目标函数）：最小化违反加权惩罚
  - 设置求解时间限制（time limit），保证流水线吞吐量
  - 批量并行处理多个子问题
  - 返回解的有效性标记（valid/invalid）和目标值

#### `train.py`
- **职责**：驱动 MARL / Graph 优化器的训练流程。
- **输入**：`data/source/` 实例 + `data/solutions/` 候选解
- **输出**：训练好的模型权重
- **关键功能**：
  - 数据集划分（early/middle 用于训练，late 用于验证，test 用于评测）
  - 训练循环：加载实例 → DataLoader → GraphMapping → 构造训练样本
  - 调用 MARL/GRAPH 的 `train()` 方法
  - 记录训练指标（loss、软约束违反数、解质量）
  - 模型保存与加载（checkpointing）

---

### `src/utils/` — 工具层

#### `dataReader.py`
- **职责**：底层 XML 解析工具函数（被 `src/data/DataReader.py` 调用）。
- 提供通用的 XML 节点读取、属性提取辅助函数。

#### `solutionReader.py`
- **职责**：读取 `data/solutions/` 下的 XML 格式候选解，解析为 Python 字典。
- 输出格式：`{course_id: (room_id, time_id), ...}`

#### `solutionWriter.py`
- **职责**：将 Python 解对象序列化回 ITC 标准 XML 格式，用于提交和评测。

#### `constraints.py`
- **职责**：集中定义并实现所有约束的检查逻辑。
- **关键功能**：
  - `check_hard(solution, instance) -> List[Violation]`
  - `check_soft(solution, instance) -> (penalty: float, details: List)`
  - 供 MIP 建模、RL 奖励函数、验证器复用

#### `validator.py`
- **职责**：对最终解进行完整性验证，输出违约报告。
- **关键功能**：
  - 调用 `constraints.py` 检查所有硬/软约束
  - 输出标准化评分（penalty）和违约详情
  - 对标 ITC-2019 官方 validator 行为

---

## 数据流总结

```
XML 实例 → DataReader → Instance 对象
                            │
              ┌─────────────┤
              │  小规模      │  大规模
              │             ▼
              │    GraphMapping → 冲突图
              │             │
              │         Segmentation → [子实例1, 子实例2, ...]
              │             │
              └─────────────┤
                            ▼
                  [对每个子实例] GraphMapping
                            │
                         Gurobi MIP
                            │
                    [解1(valid), 解2(invalid), ...]
                            │
              ┌─────────────┼─────────────┐
           SOTA           MARL          GRAPH
              └─────────────┴─────────────┘
                            │
                    solutionWriter → 最终 XML 解
                            │
                        validator → 评分报告
```

---

## 关键设计约定

1. **Instance 对象格式**：由 `DataReader` 定义，作为全系统的数据契约，所有模块均以此为输入。
2. **解的表示**：统一使用 `Dict[course_id, Tuple[room_id, time_id]]`，由 `solutionReader/Writer` 负责 IO 转换。
3. **图表示**：`GraphMapping` 输出 `networkx.Graph` 或 `torch_geometric.Data`，供 `Segmentation` 和 `GRAPH` 共用。
4. **规模判定阈值**：小实例（课程数 < N）直接进入 MIP；大实例先分割。阈值 N 需根据 Gurobi 实际求解时间实验确定。

---

## 依赖

```
gurobipy       # MIP 求解
networkx       # 图构建与社区检测
python-louvain # 社区检测（或 leidenalg）
torch          # MARL / GNN 模型
torch_geometric # 图神经网络
xmltodict      # XML 解析
numpy / pandas # 数据处理
```

---

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整流水线（参考 main.ipynb / main.py）
python main.py
```