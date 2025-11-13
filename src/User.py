from typing import List, Tuple
from langchain_classic.schema import Document
from BuildIndex import BuildIndex
import os


class User:
    """用户交互类，负责处理用户查询、向量检索和语义重排序"""
    
    def __init__(
        self,
        vector_store_path: str = "../vector_store",
        index_name: str = "exam_qa_faiss"
    ):
        """
        初始化用户交互模块
        
        Args:
            vector_store_path: 向量存储路径
            index_name: 索引名称
        """
        self.vector_store_path = vector_store_path
        self.index_name = index_name
        self.indexer = None
        self.reranker = None
        
    def load_index(self):
        """加载向量索引"""
        print("正在加载向量索引...")
        self.indexer = BuildIndex(vector_store_path=self.vector_store_path)
        self.indexer.load_vector_store(index_name=self.index_name)
        print("✅ 向量索引加载成功！")
    
    def initialize_reranker(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序模型
        
        Args:
            model_name: 重排序模型名称
        """
        try:
            from sentence_transformers import CrossEncoder
            
            print(f"正在加载重排序模型: {model_name}")
            self.reranker = CrossEncoder(model_name)
            print("✅ 重排序模型加载成功！")
            
        except Exception as e:
            print(f"⚠️ 加载重排序模型失败: {str(e)}")
            print("将使用向量检索的原始排序")
            self.reranker = None
    
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 10
    ) -> List[Tuple[Document, float]]:
        """
        从向量库中检索相关文档
        
        Args:
            query: 用户查询
            top_k: 初步检索的文档数量
            
        Returns:
            (Document, score)元组列表
        """
        if self.indexer is None:
            raise ValueError("向量索引未加载，请先调用 load_index()")
        
        print(f"\n🔍 检索中...")
        results = self.indexer.search_with_score(query, k=top_k)
        print(f"✅ 初步检索到 {len(results)} 个相关文档")
        
        return results
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Tuple[Document, float]],
        top_k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        使用重排序模型对检索结果进行重新排序
        
        Args:
            query: 用户查询
            documents: 初步检索的文档列表
            top_k: 最终返回的文档数量
            
        Returns:
            重排序后的(Document, score)元组列表
        """
        if self.reranker is None:
            print("⚠️ 未加载重排序模型，返回原始检索结果")
            return documents[:top_k]
        
        print(f"\n🔄 语义重排序中...")
        
        # 准备文档对
        pairs = [[query, doc.page_content] for doc, _ in documents]
        
        # 计算重排序分数
        rerank_scores = self.reranker.predict(pairs)
        
        # 将文档与新分数配对
        reranked_results = [
            (documents[i][0], float(rerank_scores[i])) 
            for i in range(len(documents))
        ]
        
        # 按分数降序排序（分数越高越相关）
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"✅ 重排序完成，返回前 {top_k} 个最相关文档")
        
        return reranked_results[:top_k]
    
    def query(
        self,
        user_input: str,
        retrieve_k: int = 10,
        final_k: int = 3,
        use_rerank: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        处理用户查询的完整流程
        
        Args:
            user_input: 用户输入
            retrieve_k: 初步检索的文档数量
            final_k: 最终返回的文档数量
            use_rerank: 是否使用重排序
            
        Returns:
            最终的(Document, score)元组列表
        """
        if not user_input.strip():
            print("⚠️ 查询内容为空")
            return []
        
        print(f"\n{'='*60}")
        print(f"用户查询: {user_input}")
        print(f"{'='*60}")
        
        # 1. 向量检索
        retrieved_docs = self.retrieve_documents(user_input, top_k=retrieve_k)
        
        if not retrieved_docs:
            print("❌ 未找到相关文档")
            return []
        
        # 2. 语义重排序（可选）
        if use_rerank and self.reranker is not None:
            final_results = self.rerank_documents(
                user_input, 
                retrieved_docs, 
                top_k=final_k
            )
        else:
            final_results = retrieved_docs[:final_k]
        
        return final_results
    
    def display_results(
        self,
        results: List[Tuple[Document, float]],
        show_metadata: bool = True
    ):
        """
        显示查询结果
        
        Args:
            results: 查询结果列表
            show_metadata: 是否显示元数据
        """
        print(f"\n{'='*60}")
        print(f"检索结果 (共 {len(results)} 条)")
        print(f"{'='*60}")
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n📄 结果 {i} (相关度: {score:.4f})")
            print(f"{'─'*60}")
            print(f"内容:\n{doc.page_content}")
            
            if show_metadata:
                print(f"\n元数据:")
                for key, value in doc.metadata.items():
                    print(f"  - {key}: {value}")
            
            print(f"{'─'*60}")
    
    def interactive_query(self):
        """交互式查询模式"""
        print("\n" + "="*60)
        print("🤖 智能问答系统 - 交互模式")
        print("="*60)
        print("输入 'exit' 或 'quit' 退出")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("👤 请输入您的问题: ").strip()
                
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print("\n👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                # 执行查询
                results = self.query(
                    user_input,
                    retrieve_k=10,
                    final_k=3,
                    use_rerank=True
                )
                
                # 显示结果
                if results:
                    self.display_results(results, show_metadata=True)
                else:
                    print("\n❌ 未找到相关信息，请换个问题试试")
                
                print("\n" + "-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")


# 测试代码
if __name__ == "__main__":
    # 创建用户交互实例
    current_dir = os.path.dirname(__file__)
    vector_store_path = os.path.join(current_dir, "../vector_store")
    
    user_module = User(
        vector_store_path=vector_store_path,
        index_name="exam_qa_faiss"
    )
    
    # 加载索引
    user_module.load_index()
    
    # 初始化重排序模型（可选）
    user_module.initialize_reranker(model_name="BAAI/bge-reranker-base")
    
    # 测试单次查询
    query = "如何报名高考"
    results = user_module.query(query, retrieve_k=10, final_k=3)
    user_module.display_results(results)
    
    # 交互式查询
    # user_module.interactive_query()