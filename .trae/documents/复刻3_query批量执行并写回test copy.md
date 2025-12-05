## 绝对路径文件
- 输入固定读取：`d:\RagProject\data\test.xlsx`
- 输出固定写入：`d:\RagProject\data\test copy.xlsx`
- 使用 openpyxl 以绝对路径加载与保存，避免相对路径问题；处理完每批次统一保存到 `test copy.xlsx`。

## 批处理脚本
- 新增 `src/3_query_batch.py`，复用 `query_rag`（`src/3_query.py:238`）。
- 表头识别“问题”列（`问题/问句/query/问题描述/标题`）；若缺少输出列则自动创建：`大模型回答`、`参考问答对`、`相似度是否低于0.3（若是的话填是，不是的话可以不填）`。

## 写回与阈值
- `answer` → 写入 `大模型回答`；
- `contexts` Top3 → `[page:X] 文本` 以 `---` 分隔写入 `参考问答对`；
- `taotalscore` → 取 Top1 `context.final_score`，若 `< 0.3` 写入阈值列为 `是`，否则留空。

## 轮询等待与重试
- 每条问题调用 `query_rag`，若失败或结果不完整，指数退避重试（默认 3 次，起始 1000ms，倍增）。成功后立即写回该行；每次成功后可短暂停顿（默认 300ms）避免限流。

## 批次控制
- 每次脚本运行仅处理 10 条，从第 2 行开始，连续读取非空问题的 10 行。支持参数 `--start` 与 `--count`（默认 `--count 10`）。
- 完成批次后保存到 `d:\RagProject\data\test copy.xlsx`。

## 运行命令
- `python src/3_query_batch.py --start 2 --count 10`（脚本内部固定使用上述绝对路径读写）。

## 验收
- 指定两个绝对路径文件成功读写；批次 10 条均在 `test copy.xlsx` 写入回答与参考问答对；阈值列按 Top1 `final_score` 正确标注；失败行经重试后不中断整体处理。