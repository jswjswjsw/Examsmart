# RAG智能问答系统 - 考试咨询助手

基于 LangChain + FAISS + 通义千问 构建的智能问答系统

## 功能特性

- ✅ **文档加载与分割**: 自动加载CSV文件并智能分割
- ✅ **向量化与索引**: 使用FAISS构建高效向量索引
- ✅ **语义检索**: 基于向量相似度的精确检索
- ✅ **语义重排序**: 使用CrossEncoder提升检索精度
- ✅ **智能问答**: 调用通义千问大模型生成准确回答
- ✅ **交互式对话**: 支持连续对话模式

## 项目结构

```
RagProject/
├── data/                           # 数据文件夹
│   ├── 考试院faq_triplets 0328 - 高考学考.csv
│   ├── 考试院faq_triplets 0328 - 研考成考.csv
│   ├── 考试院faq_triplets 0328 - 证书考试.csv
│   ├── 考试院faq_triplets 0328 - 中考中招.csv
│   └── 考试院faq_triplets 0328 - 自学考试.csv
├── vector_store/                   # 向量存储（自动生成）
│   └── exam_qa_faiss/
├── src/                            # 源代码
│   ├── DataTreating.py            # 文档处理模块
│   ├── BuildIndex.py              # 索引构建模块
│   ├── User.py                    # 用户交互模块
│   ├── Response.py                # 回答生成模块
│   ├── main.py                    # 主程序
│   └── config.py                  # 配置文件
├── run.bat                        # Windows启动脚本
├── run.sh                         # Linux/Mac启动脚本
└── README.md                      # 说明文档
```

### 创建并激活Python虚拟环境

为了隔离项目依赖，建议您使用Python虚拟环境。

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

###  可以一键安装依赖

本项目使用 `requirements.txt` 文件来管理所有依赖。请运行以下命令进行安装：
```bash
pip install -r requirements.txt
```

##  或者分步使用pip安装依赖

```bash
pip install langchain langchain-community
pip install faiss-cpu
pip install sentence-transformers
pip install dashscope
pip install pandas
pip install beautifulsoup4
```

## 配置API密钥

### Windows:
```cmd
set DASHSCOPE_API_KEY=your-api-key-here
```

### Linux/Mac:
```bash
export DASHSCOPE_API_KEY=your-api-key-here
```

## 使用方法

### 方法1: 使用启动脚本（推荐）

**Windows:**
```cmd
run.bat
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

### 方法2: 直接运行

```bash
cd src
python main.py
```

## 使用流程

1. **首次运行**: 系统会自动构建向量索引（约1-2分钟）
2. **测试模式**: 可选择运行预设问题测试
3. **交互模式**: 进入对话模式，实时问答
4. **后续运行**: 直接加载已有索引，快速启动

## 系统架构

```
用户输入
   ↓
文档加载与分割 (DataTreating)
   ↓
向量化与索引构建 (BuildIndex)
   ↓
向量检索 (User)
   ↓
语义重排序 (User)
   ↓
大模型生成回答 (Response)
   ↓
返回结果
```

## 参数配置

在 `config.py` 中可以调整以下参数：

- `CHUNK_SIZE`: 文档分块大小（默认500）
- `CHUNK_OVERLAP`: 分块重叠大小（默认50）
- `RETRIEVE_TOP_K`: 初步检索数量（默认10）
- `FINAL_TOP_K`: 最终返回数量（默认3）
- `MODEL_NAME`: 大模型名称（默认qwen-turbo）

## 常见问题

### Q: 如何获取通义千问API密钥？如何获取openai密钥？
A: 访问 https://dashscope.aliyun.com/ 注册并获取
A: 访问 https://www.closeai-asia.com/
### Q: 重排序模型加载失败？
A: 系统会自动降级使用向量检索排序，不影响使用

### Q: 如何更新数据？
A: 将新的CSV文件放入 `data/` 目录，删除 `vector_store/` 文件夹，重新运行程序

## 技术栈

- **LangChain**: RAG框架
- **FAISS**: 向量检索引擎
- **Sentence-Transformers**: 文本嵌入模型
- **通义千问**: 大语言模型
- **Pandas**: 数据处理

## 许可证

MIT License
