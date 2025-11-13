from typing import List, Tuple
from langchain_classic.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


class BuildIndex:
    """构建向量索引类，负责文档向量化和存储（使用FAISS）"""
    
    def __init__(
        self, 
        embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        vector_store_path: str = "../vector_store"
    ):
        """
        初始化索引构建器
        
        Args:
            embedding_model_name: 嵌入模型名称
            vector_store_path: 向量存储路径
        """
        self.vector_store_path = vector_store_path
        self.embedding_model_name = embedding_model_name
        self.embeddings = None
        self.vector_store = None
        
        # 确保向量存储目录存在
        os.makedirs(vector_store_path, exist_ok=True)
    
    def initialize_embeddings(self):
        """初始化嵌入模型"""
        print(f"正在加载嵌入模型: {self.embedding_model_name}")
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={'device': 'cpu'},  # 使用CPU，如有GPU可改为'cuda'
                encode_kwargs={'normalize_embeddings': True}  # 归一化向量
            )
            print("嵌入模型加载成功！")
            return self.embeddings
            
        except Exception as e:
            print(f"加载嵌入模型时出错: {str(e)}")
            raise
    
    def build_vector_store(self, documents: List[Document]) -> FAISS:
        """
        构建向量存储（向量化 + 构建索引 + 存储）
        
        Args:
            documents: 文档列表
            
        Returns:
            FAISS向量存储对象
        """
        if not documents:
            raise ValueError("文档列表为空，无法构建向量存储")
        
        if self.embeddings is None:
            self.initialize_embeddings()
        
        print(f"开始构建FAISS向量索引，共 {len(documents)} 个文档片段...")
        
        try:
            # 使用FAISS构建向量存储（这一步完成了向量化和索引构建）
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            print("✅ FAISS向量索引构建成功！")
            return self.vector_store
            
        except Exception as e:
            print(f"构建向量索引时出错: {str(e)}")
            raise
    
    def save_vector_store(self, index_name: str = "faiss_index"):
        """
        保存向量存储到本地
        
        Args:
            index_name: 索引名称
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化，请先构建向量索引")
        
        save_path = os.path.join(self.vector_store_path, index_name)
        
        try:
            self.vector_store.save_local(save_path)
            print(f"✅ 向量索引已保存到: {save_path}")
            
        except Exception as e:
            print(f"保存向量索引时出错: {str(e)}")
            raise
    
    def load_vector_store(self, index_name: str = "faiss_index") -> FAISS:
        """
        从本地加载向量存储
        
        Args:
            index_name: 索引名称
            
        Returns:
            FAISS向量存储对象
        """
        if self.embeddings is None:
            self.initialize_embeddings()
        
        load_path = os.path.join(self.vector_store_path, index_name)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"向量索引不存在: {load_path}")
        
        try:
            self.vector_store = FAISS.load_local(
                load_path, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ 向量索引已加载: {load_path}")
            return self.vector_store
            
        except Exception as e:
            print(f"加载向量索引时出错: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 3) -> List[Document]:
        """
        在向量存储中搜索相似文档
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            相似文档列表
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
            
        except Exception as e:
            print(f"搜索时出错: {str(e)}")
            raise
    
    def search_with_score(self, query: str, k: int = 3):
        """
        在向量存储中搜索相似文档并返回分数

        Returns:
            List[Tuple[Document, float]]
        """
        if self.vector_store is None:
            raise ValueError("向量存储未初始化")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            print(f"搜索时出错: {str(e)}")
            raise
    
    def process(
        self, 
        documents: List[Document], 
        index_name: str = "faiss_index"
    ) -> FAISS:
        """
        完整的索引构建流程：初始化 -> 构建 -> 保存
        
        Args:
            documents: 文档列表
            index_name: 索引名称
            
        Returns:
            FAISS向量存储对象
        """
        # 1. 初始化嵌入模型
        self.initialize_embeddings()
        
        # 2. 构建向量存储（向量化 + 构建索引）
        self.build_vector_store(documents)
        
        # 3. 保存向量存储到磁盘
        self.save_vector_store(index_name)
        
        return self.vector_store


# 测试代码
if __name__ == "__main__":
    from DataTreating import DataTreating
    
    # 处理文档
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    processor = DataTreating(data_dir)
    documents = processor.process(chunk_size=500, chunk_overlap=50)
    
    # 构建索引
    vector_store_path = os.path.join(os.path.dirname(__file__), "../vector_store")
    indexer = BuildIndex(vector_store_path=vector_store_path)
    indexer.process(documents, index_name="exam_qa_faiss")
    
    # 测试搜索
    print("\n" + "=" * 60)
    print("测试搜索功能")
    print("=" * 60)
    query = "如何报名考试"
    
    # 测试 search_with_score
    results = indexer.search_with_score(query, k=3)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n结果 {i} (距离: {score:.4f}):")
        print(f"内容: {doc.page_content[:100]}...")
        print(f"来源: {doc.metadata.get('source', 'unknown')}")