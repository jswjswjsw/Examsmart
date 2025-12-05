## 输入与遍历
- 读取 `data/test.xlsx` 第一张工作表，从第 2 行开始逐行（序号表示一条测试）。
- 仅取表头为 `问题`（或 `问句/query/问题描述/标题`）的列作为查询文本。

## 执行与提取
- 每行调用 `query_rag`（`src/3_query.py:213`），参数 `top_k=50`、`rerank_k=10`、`fast_mode=True`。
- 提取字段：
  - `answer`：写入 `大模型回答` 列。
  - `contexts`：转成可读文本并写入 `参考问答对` 列，格式 `[page:X] 文本`，取前 3 条（`src/excel_eval.py:44` 的思路）。
  - `taotalscore`：以检索重排后的第 1 条 `context` 的 `final_score` 作为该行的总分（`src/3_query.py:245-250`），若 `< 0.3` 写入 `相似度是否低于0.3（若是的话填是，不是的话可以不填）` 为 `是`，否则留空。

## 列处理
- 若 `大模型回答`、`参考问答对`、`相似度是否低于0.3…` 不存在，自动在首行追加列并写入（沿用 `src/excel_eval.py:66-78` 的列创建方式）。

## 落盘
- 执行完毕后保存到：
  - 覆盖原文件：`python src/excel_eval.py --in data/test.xlsx`
  - 输出 新文件：`python src/excel_eval.py --in data/test.xlsx --out data/test_out.xlsx`

## 验收
- 每条测试均在对应行写入：`大模型回答` 与 `参考问答对`。
- 基于 `taotalscore` 判定的相似度阈值结果正确填入/留空。

## 备注
- `taotalscore` 的取值定义为 Top1 `context.final_score`，与当前实现一致（`src/3_query.py:245-250`）。如需改为平均分或 TopK 加权，可在实现阶段调整。