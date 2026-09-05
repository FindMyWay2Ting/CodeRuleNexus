# Code Wiki 多仓库评测

本目录将扫描器输出与人工 Ground Truth 分开保存，避免使用系统自己的结果证明系统正确。

## 目录约定

- `corpus.yaml`：仓库、固定 Commit、数据集角色和评测状态。
- `runs/`：扫描器在固定版本上的原始观测摘要；写入后不得回填为 Ground Truth。
- `ground_truth/`：根据官方架构文档、协议文件、部署配置和人工源码核验建立的标准答案。
- `reports/`：Precision、Recall、F1、证据准确率、断链和误报分类报告。

## 防止过拟合

1. 开发集可以用于定位错误和调整通用规则。
2. Holdout 仓库只运行并记录结果，不依据其错误调整扫描规则。
3. 如果依据 Holdout 结果修改了实现，该仓库必须降级为开发集，并选择新的 Holdout。
4. 项目名、业务目录名和业务资源名不得成为生产扫描规则。
5. 框架适配器必须基于公开协议、依赖、语法或配置证据，并在多个独立仓库验证。
6. Agent 推断写为 `inferred`；扫描器事实写为 `verified`；证据不足写为 `unknown`。

## 当前 OpenTelemetry Demo 基线

首次基线固定在 Commit `d6fd782ee9ceedaec4a1c1f81017e8617d063240`。本次只记录系统观测值，Ground Truth 尚未完成，因此不得计算或宣称 Precision、Recall。

Ground Truth 应从同一 Commit 的 `compose*.yaml`、`.env`、`pb/demo.proto`、官方架构文档和人工源码核验建立，至少覆盖服务、语言、RPC、Kafka、数据库、下游服务和关键调用边。
