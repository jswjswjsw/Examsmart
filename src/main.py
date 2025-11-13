from DataTreating import DataTreating
from BuildIndex import BuildIndex
from User import User
from Response import Response
import os
from dotenv import load_dotenv
load_dotenv()


def build_index_pipeline():
    """构建索引的完整流程"""
    print("\n" + "=" * 70)
    print("📚 步骤1: 构建FAISS向量索引")
    print("=" * 70)
    
    current_dir = os.path.dirname(__file__)
    data_dir = os.path.join(current_dir, "../data")
    vector_store_path = os.path.join(current_dir, "../vector_store")
    
    try:
        # 1. 加载和分割文档
        print("\n[1.1] 文档加载与分割")
        print("-" * 70)
        processor = DataTreating(data_dir)
        documents = processor.process(chunk_size=500, chunk_overlap=50)
        
        if not documents:
            print("❌ 错误：没有加载到文档，请检查 data 目录")
            return False
        
        # 2. 构建向量索引
        print("\n[1.2] 构建向量索引并保存")
        print("-" * 70)
        indexer = BuildIndex(vector_store_path=vector_store_path)
        indexer.process(documents, index_name="exam_qa_faiss")
        
        print("\n✅ 索引构建完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 构建索引时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def initialize_system(use_rerank: bool = True):
    """初始化系统的所有模块"""
    print("\n" + "=" * 70)
    print("🔧 步骤2: 初始化系统模块")
    print("=" * 70)
    
    current_dir = os.path.dirname(__file__)
    vector_store_path = os.path.join(current_dir, "../vector_store")
    
    try:
        # 1. 初始化用户查询模块
        print("\n[2.1] 加载向量索引")
        print("-" * 70)
        user_module = User(
            vector_store_path=vector_store_path,
            index_name="exam_qa_faiss"
        )
        user_module.load_index()
        
        # 2. 初始化重排序模型（可选）
        if use_rerank:
            print("\n[2.2] 初始化语义重排序模型")
            print("-" * 70)
            try:
                user_module.initialize_reranker(model_name="BAAI/bge-reranker-base")
            except Exception as e:
                print(f"⚠️ 重排序模型加载失败: {str(e)}")
                print("将使用向量检索的原始排序")
                use_rerank = False
        
        # 3. 初始化回答生成模块
        # 将 OPENAI_API_KEY1 同步到标准环境变量（与 HelloWorld.ipynb 一致）
        if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY1"):
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY1")
        if os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")
        print("\n[2.3] 初始化大语言模型")
        print("-" * 70)
        
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY1")
        if not api_key:
            print("⚠️ 未设置 OPENAI_API_KEY/OPENAI_API_KEY1 环境变量（或 .env 未加载）")
            print("\n将以检索模式运行（不生成回答）")
            return user_module, None, use_rerank
        
        response_module = Response(
            api_key=api_key,
            model_name="gpt-4o-mini"
        )
        response_module.initialize_chain()
        
        print("\n✅ 系统初始化完成！")
        return user_module, response_module, use_rerank
        
    except FileNotFoundError:
        print("\n❌ 未找到向量索引，请先运行构建索引流程")
        return None, None, False
    except Exception as e:
        print(f"\n❌ 初始化系统时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, False


def test_qa_system(user_module: User, response_module: Response, use_rerank: bool):
    """测试问答系统"""
    print("\n" + "=" * 70)
    print("🧪 步骤3: 测试问答功能")
    print("=" * 70)
    
    test_questions = [
        "如何报名高考？",
        "自学考试的报名流程是什么？",
        "研究生考试什么时候开始报名？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(test_questions)}: {question}")
        print(f"{'='*70}")
        
        try:
            # 检索相关文档
            retrieved_docs = user_module.query(
                question,
                retrieve_k=10,
                final_k=3,
                use_rerank=use_rerank
            )
            
            if not retrieved_docs:
                print("❌ 未找到相关文档")
                continue
            
            # 如果有response_module，生成回答
            if response_module:
                result = response_module.generate_answer_with_sources(
                    question,
                    retrieved_docs,
                    show_sources=True
                )
                response_module.display_answer(result)
            else:
                # 只显示检索结果
                print("\n📚 检索结果 (未生成回答):")
                user_module.display_results(retrieved_docs, show_metadata=False)
            
        except Exception as e:
            print(f"❌ 处理问题时出错: {str(e)}")
        
        print("\n" + "─"*70)


def interactive_qa_mode(user_module: User, response_module: Response, use_rerank: bool):
    """交互式问答模式"""
    print("\n" + "=" * 70)
    print("🤖 步骤4: 交互式问答模式")
    print("=" * 70)
    print("输入问题开始对话，输入 'exit'、'quit' 或 '退出' 结束")
    print("=" * 70 + "\n")
    
    conversation_count = 0
    
    while True:
        try:
            question = input("\n👤 请输入您的问题: ").strip()
            
            if question.lower() in ['exit', 'quit', '退出']:
                print(f"\n👋 再见！本次对话共进行了 {conversation_count} 轮")
                break
            
            if not question:
                print("⚠️ 问题不能为空，请重新输入")
                continue
            
            conversation_count += 1
            
            # 检索相关文档
            print(f"\n{'─'*70}")
            retrieved_docs = user_module.query(
                question,
                retrieve_k=10,
                final_k=3,
                use_rerank=use_rerank
            )
            
            if not retrieved_docs:
                print("\n❌ 未找到相关信息，请换个问题试试")
                conversation_count -= 1
                continue
            
            # 生成回答
            if response_module:
                result = response_module.generate_answer_with_sources(
                    question,
                    retrieved_docs,
                    show_sources=True
                )
                response_module.display_answer(result)
            else:
                # 只显示检索结果
                print("\n📚 检索结果 (未生成回答):")
                user_module.display_results(retrieved_docs, show_metadata=False)
            
        except KeyboardInterrupt:
            print(f"\n\n👋 再见！本次对话共进行了 {conversation_count} 轮")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")


def main():
    """主函数：完整的RAG问答系统"""
    print("\n" + "=" * 70)
    print("🚀 RAG智能问答系统 - 考试咨询助手")
    print("=" * 70)
    print("基于 LangChain + FAISS + Openai")
    print("=" * 70)
    
    current_dir = os.path.dirname(__file__)
    vector_store_path = os.path.join(current_dir, "../vector_store")
    index_path = os.path.join(vector_store_path, "exam_qa_faiss")
    
    # ========== 步骤1: 检查并构建索引 ==========
    if not os.path.exists(index_path):
        print("\n📝 未检测到索引文件，开始构建...")
        success = build_index_pipeline()
        if not success:
            print("\n❌ 索引构建失败，程序退出")
            return
    else:
        print("\n✅ 检测到已有索引文件")
    
    # ========== 步骤2: 初始化系统 ==========
    user_module, response_module, use_rerank = initialize_system(use_rerank=True)
    
    if user_module is None:
        print("\n❌ 系统初始化失败，程序退出")
        return
    
    # ========== 步骤3: 测试系统 ==========
    print("\n是否运行测试？")
    run_test = input("输入 'y' 运行测试，直接回车跳过: ").strip().lower()
    
    if run_test == 'y':
        test_qa_system(user_module, response_module, use_rerank)
    
    # ========== 步骤4: 交互式问答 ==========
    print("\n是否进入交互式问答模式？")
    start_interactive = input("输入 'y' 开始，'n' 退出 (默认y): ").strip().lower()
    
    if start_interactive != 'n':
        interactive_qa_mode(user_module, response_module, use_rerank)
    
    print("\n" + "=" * 70)
    print("✅ 程序结束，感谢使用！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()