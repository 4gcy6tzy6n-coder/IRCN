# v0.6 代码完成前审计记录

| 审计项 | 结果 | 证据 |
|---|---|---|
| 预实现合同复核 | ACCEPT | `PRE_IMPLEMENTATION_REVIEW.md` 与独立复核记录 |
| 标签含完整预决策快照哈希 | PASS | `snapshot_sha256 = canonical_array_hash(states[tick])`；完整性测试校验每行字段 |
| 唯一 run 目录与失败保留 | PASS | 重复 ID 禁止覆盖；失败记录独占创建；异常逐 epoch 日志 |
| 训练确定性 | PASS | 两次独立运行的参数、loss history、best epoch、标准化及预测数据哈希一致 |
| Golden 碰撞标签 | PASS | 实际运行线性教师反事实递推，核验 0.0832 与 0.1025 |
| 数据划分与标准化 | PASS | scaler 只接收训练数组；训练/验证/测试按冻结种子隔离 |
| 统计单位 | PASS | 先轨迹内计算，再种子内汇总；bootstrap 以种子为单位 |
| gzip 输出和工件完整性 | PASS | 无损确定性 gzip；manifest 哈希与逐行数量测试通过 |
| 定向 v0.6 测试 | PASS | 13 passed |
| 仓库回归 | PASS | 47 passed |
| 依赖 / 编译 | PASS | `uv pip check`、`py_compile` 通过 |
| 独立代码复核 | ACCEPT | `/root/next_stage_critic` 完成的最终复核 |

## 修复过的问题

1. 每条标签此前缺少反事实前全状态快照 hash。
2. 重复运行曾共用固定输出目录，可能覆盖证据。
3. epoch 日志此前仅在训练成功结束后保留，异常时不完整。
4. 固定种子单次前向测试不足以证明训练可复现。
5. 碰撞 golden 只检查配置文字，未执行真实反事实。
6. 早停、训练标准化、种子聚合、失败记录和输出清单缺少行为测试。
7. 原始 CSV 单文件超过 GitHub 普通 Git 文件限制；改用确定性无损 gzip，并保存迁移校验 hash。

以上问题均在不更改冻结数据、模型、种子和判定门槛的前提下修复。独立复核未发现剩余重大问题。科学解释边界见 `E1_report.md` 与 `decision_v1.md`。
