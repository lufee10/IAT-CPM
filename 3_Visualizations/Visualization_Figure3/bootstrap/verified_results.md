# 双向跨范围泛化损失核验

Cross-range generalization losses persisted in both directions (lower-to-higher: Δr = 0.251, 95% percentile bootstrap CI [0.120, 0.382]; higher-to-lower: Δr = 0.264, 95% percentile bootstrap CI [0.153, 0.367]).

所有 r、Δr 和 5,000 次配对 bootstrap 由本 notebook 从预测重算。共同评价目标为 pooled 折特异残差，不是原稿 source-referenced 定义；不重新训练模型，不展示分组模型减 pooled 模型的差异。包内无历史参照表，未做与历史数值的逐位核对。