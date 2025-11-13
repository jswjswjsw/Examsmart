import pandas as pd
from typing import List, Dict
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document
import os


class DataTreating:
    """文档处理类，负责加载和分割文档"""
    
    def __init__(self, data_dir: str = "../data"):
        """
        初始化文档处理器
        
        Args:
            data_dir: 数据文件夹路径
        """
        self.data_dir = data_dir
        self.documents: List[Document] = []
    
    def load_csv_files(self) -> List[Document]:
        """
        加载所有CSV文件并转换为Document对象
        
        Returns:
            Document对象列表
        """
        documents = []
        
        # 获取data目录下所有CSV文件
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            try:
                # 读取CSV文件
                df = pd.read_csv(file_path, encoding='utf-8')
                
                # 假设CSV包含问题和答案列
                # 根据实际列名调整
                for idx, row in df.iterrows():
                    # 将每行数据转换为文本
                    content = self._row_to_text(row, csv_file)
                    
                    # 创建Document对象
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": csv_file,
                            "row_index": idx
                        }
                    )
                    documents.append(doc)
                    
                print(f"成功加载 {csv_file}，共 {len(df)} 条数据")
                
            except Exception as e:
                print(f"加载 {csv_file} 时出错: {str(e)}")
        
        self.documents = documents
        return documents
    
    def _row_to_text(self, row: pd.Series, source_file: str) -> str:
        """
        将DataFrame的一行转换为文本
        
        Args:
            row: DataFrame的一行
            source_file: 源文件名
            
        Returns:
            格式化的文本内容
        """
        # 根据实际CSV结构调整
        text_parts = []
        
        for col_name, value in row.items():
            if pd.notna(value):  # 忽略空值
                text_parts.append(f"{col_name}: {value}")
        
        return "\n".join(text_parts)
    
    def split_documents(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        分割文档
        
        Args:
            chunk_size: 每个块的字符数
            chunk_overlap: 块之间的重叠字符数
            
        Returns:
            分割后的Document列表
        """
        if not self.documents:
            print("警告: 没有文档需要分割，请先加载文档")
            return []
        
        # 创建文本分割器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "?", "；", ".", "!", "?", ";", " ", ""]
        )
        
        # 分割文档
        split_docs = text_splitter.split_documents(self.documents)
        
        print(f"文档分割完成: 原始文档 {len(self.documents)} 个，分割后 {len(split_docs)} 个片段")
        
        return split_docs
    
    def process(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50
    ) -> List[Document]:
        """
        完整的文档处理流程：加载 -> 分割
        
        Args:
            chunk_size: 每个块的字符数
            chunk_overlap: 块之间的重叠字符数
            
        Returns:
            处理后的Document列表
        """
        # 1. 加载文档
        self.load_csv_files()
        
        # 2. 分割文档
        split_docs = self.split_documents(chunk_size, chunk_overlap)
        
        return split_docs


# 测试代码
if __name__ == "__main__":
    # 创建数据处理器
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    processor = DataTreating(data_dir)
    
    # 处理文档
    documents = processor.process(chunk_size=500, chunk_overlap=50)
    
    # 打印示例
    if documents:
        print(f"\n示例文档片段:")
        print(f"内容: {documents[0].page_content[:200]}...")
        print(f"元数据: {documents[0].metadata}")