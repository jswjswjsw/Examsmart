## 目标
- 提升检索命中率与答案清晰度：先召回问答型片段保障“直答”，再补全背景型片段以增强上下文。
- 保持性能与稳定性：在现有优化（快/慢模式、BM25兜底、评分融合）基础上做最小改动。

## 动作 / 目的 / 效果
- 动作：第一次查询 where={"type":"qa_pair"}（对每个 query_text）
  - 目的：保障最清晰的答案优先进入候选池
  - 效果：所有与问题相关的 QA 对，哪怕相似度略低，也被优先检出并打上标签 source_type=qa
- 动作：第二次查询 where={"type":{"$ne":"qa_pair"}}（对每个 query_text）
  - 目的：保障背景知识全面性，补全条文、提醒与流程说明
  - 效果：召回非 QA 的普通文本，标签 source_type=bg，供生成阶段补充上下文
- 动作：候选合并与去重（按 id）
  - 目的：避免重复与顺序混乱
  - 效果：得到统一候选池 initial，保留 embed 相似度与来源标签
- 动作：评分融合调整（final_score 重构）
  - 目的：强化 QA 直答，兼顾嵌入/重排/词面匹配
  - 效果：final_score = α·rerank_prob + β·embed_norm + γ·tfidf_norm + δ·kw_hits + ε·qa_boost（qa_boost=1 对 QA 增权）
- 动作：上下文选择 Top-5，并至少包含 1 条 QA 片段
  - 目的：生成阶段既能直答又有背景
  - 效果：避免“正确答案被截断”，提升回答质量与依据完整性

## 技术实现
- 新增：_retrieve_by_meta(where: dict, query_texts: List[str], top_k: int) → 使用 Chroma `query(..., where=where)` 进行过滤检索，返回带 source_type 标记的条目
- 修改：query_rag 流程
  1. 生成 query_texts（快/慢模式）
  2. qa_initial = _retrieve_by_meta({"type":"qa_pair"}, query_texts, qa_top_k)
  3. bg_initial = _retrieve_by_meta({"type":{"$ne":"qa_pair"}}, query_texts, bg_top_k)
  4. initial = 合并去重并按 embed 相似度预排序，截断到 pool_k
  5. 触发兜底：关键词与 BM25（保持现有逻辑）
  6. 重排：硅基流动 BAAI/bge-reranker-v2-m3
  7. 评分融合：加入 qa_boost 权重
  8. 选择上下文 Top-5，至少 1 条 QA
  9. 生成答案（遵循已集成的 RAG 策略提示）
- 参数化：
  - 新增命令行/交互参数：`--qa_top_k`、`--bg_top_k`、`:set qa_top_k`、`:set bg_top_k`
  - 默认：qa_top_k=40、bg_top_k=30（可依据耗时调整）
- 调试：`--debug` 打印两次查询的命中条数与ids、评分分解（各权重项），便于定位问题

## 验收标准
- QA问题场景：Top-1 上下文至少包含一个匹配的 QA 片段；答案引用明确依据与页码
- 综合问题场景：最终上下文含 QA 与背景混合；不再出现“上下文非空却拒答”的情况
- 性能：在 fast_mode 下总耗时与当前持平或略增但可控（< 10–12s，视模型与网络）

## 风险与兼容
- Chroma `where` 依赖写入时的 metadata 类型字段存在并一致；若无该字段，回退到现有流程
- 大量非 QA 背景文本时需控制 bg_top_k 与 pool_k，防止重排负载过大
- 权重需可调：提供默认值并允许通过命令行与交互动态调整