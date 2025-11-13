from typing import List, Tuple
from langchain_classic.schema import Document
from langchain_classic.prompts import PromptTemplate
from langchain_community.llms import Tongyi
from langchain_classic.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI


class Response:
    """回答生成类，负责调用大模型根据检索结果生成回答"""
    
    def __init__(
        self,
        api_key: str = None,
        model_name: str = "gpt-4o-mini"
    ):
        """
        初始化回答生成器
        
        Args:
            api_key: OpenAI API密钥
            model_name: 模型名称
        """
        # 优先使用 OPENAI_API_KEY 或 OPENAI_API_KEY1（与 HelloWorld.ipynb 一致）
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY1")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        self.model_name = model_name
        self.llm = None
        self.chain = None
        
        if not self.api_key:
            raise ValueError("请在 .env 中设置 OPENAI_API_KEY 或 OPENAI_API_KEY1")
    
    def initialize_llm(self):
        """初始化大语言模型（OpenAI via langchain_openai.ChatOpenAI）"""
        print(f"正在初始化 OpenAI 模型: {self.model_name}")
        
        try:
            # 直接使用 ChatOpenAI，支持通过 .env 注入的 api_key/base_url
            self.llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=0.7,
                max_tokens=2000
            )
            print("✅ OpenAI 客户端初始化成功！")
        except Exception as e:
            print(f"❌ 初始化 OpenAI 时出错: {str(e)}")
            raise
    
    def initialize_chain(self):
        """兼容主流程的初始化入口（不再使用 LLMChain）"""
        if self.llm is None:
            self.initialize_llm()
        print("✅ LLM 初始化完成（使用 OpenAI 客户端）")
        
        prompt_template = self.create_prompt_template()
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            verbose=False
        )
        
        print("✅ LLM链初始化成功！")
    
    def create_prompt_template(self) -> PromptTemplate:
        """
        创建提示词模板
        
        Returns:
            PromptTemplate对象
        """
        template = """你是一个专业的考试咨询助手，负责回答用户关于各类考试的问题。

请根据以下检索到的相关信息，准确、详细地回答用户的问题。

检索到的相关信息：
{context}

用户问题：{question}

回答要求：
1. 如果检索到的信息能够回答问题，请基于这些信息给出准确、详细的回答
2. 如果检索到的信息不足以回答问题，请诚实地告知用户，并给出可能的建议
3. 回答要结构清晰，条理分明
4. 使用友好、专业的语气
5. 如果涉及具体的时间、流程、要求等，请特别注明

回答："""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def initialize_chain(self):
        """兼容主流程的初始化入口（不再使用LLMChain）"""
        if self.llm is None:
            self.initialize_llm()
        print("✅ LLM 初始化完成（使用 OpenAI 客户端）")
        
        prompt_template = self.create_prompt_template()
        
        self.chain = LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            verbose=False
        )
        
        print("✅ LLM链初始化成功！")
    
    def format_context(
        self, 
        retrieved_docs: List[Tuple[Document, float]]
    ) -> str:
        """
        格式化检索到的文档为上下文
        
        Args:
            retrieved_docs: 检索到的文档列表
            
        Returns:
            格式化的上下文字符串
        """
        if not retrieved_docs:
            return "未找到相关信息"
        
        context_parts = []
        
        for i, (doc, score) in enumerate(retrieved_docs, 1):
            context_parts.append(f"[文档{i}] (相关度: {score:.4f})")
            context_parts.append(doc.page_content)
            context_parts.append("")  # 空行分隔
        
        return "\n".join(context_parts)
    
    def generate_answer(
        self,
        question: str,
        retrieved_docs: List[Tuple[Document, float]]
    ) -> str:
        """
        生成回答
        """
        if self.llm is None:
            self.initialize_llm()
        
        # 格式化上下文
        context = self.format_context(retrieved_docs)
        
        print(f"\n🤖 正在生成回答...")
        
        try:
            # 使用提示模板生成用户消息（与 notebook 的用法对齐）
            prompt_template = self.create_prompt_template()
            user_prompt = prompt_template.format(context=context, question=question)
            
            # 使用 ChatOpenAI 直接调用
            response = self.llm.invoke(user_prompt)
            
            print("✅ 回答生成完成！")
            return response.content.strip()
            
        except Exception as e:
            error_msg = f"生成回答时出错: {str(e)}"
            print(f"❌ {error_msg}")
            return f"抱歉，{error_msg}"
    
    def generate_answer_with_sources(
        self,
        question: str,
        retrieved_docs: List[Tuple[Document, float]],
        show_sources: bool = True
    ) -> dict:
        """
        生成回答并附带来源信息
        
        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档列表
            show_sources: 是否显示来源
            
        Returns:
            包含回答和来源的字典
        """
        # 生成回答
        answer = self.generate_answer(question, retrieved_docs)
        
        # 提取来源信息
        sources = []
        if show_sources:
            for i, (doc, score) in enumerate(retrieved_docs, 1):
                source_info = {
                    "index": i,
                    "score": score,
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                sources.append(source_info)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }
    
    def display_answer(self, result: dict):
        """
        显示回答结果
        
        Args:
            result: generate_answer_with_sources返回的结果
        """
        print("\n" + "="*60)
        print("💬 问答结果")
        print("="*60)
        
        print(f"\n❓ 问题: {result['question']}")
        print(f"\n✅ 回答:\n{result['answer']}")
        
        if result.get('sources'):
            print(f"\n📚 参考来源:")
            print("─"*60)
            for source in result['sources']:
                print(f"\n[来源{source['index']}] (相关度: {source['score']:.4f})")
                print(f"内容: {source['content']}")
                if source.get('metadata'):
                    print(f"元数据: {source['metadata']}")
        
        print("\n" + "="*60)


# 测试代码
if __name__ == "__main__":
    from User import User
    
    # 设置API密钥（请替换为你的实际API密钥）
    os.environ["DASHSCOPE_API_KEY"] = "your-api-key-here"
    
    # 1. 创建用户查询模块
    current_dir = os.path.dirname(__file__)
    vector_store_path = os.path.join(current_dir, "../vector_store")
    
    user_module = User(
        vector_store_path=vector_store_path,
        index_name="exam_qa_faiss"
    )
    
    # 2. 加载索引
    user_module.load_index()
    
    # 3. 初始化重排序（可选）
    # user_module.initialize_reranker()
    
    # 4. 创建回答生成模块
    response_module = Response(model_name="qwen-turbo")
    response_module.initialize_chain()
    
    # 5. 测试完整流程
    test_question = "如何报名高考？"
    
    # 检索相关文档
    retrieved_docs = user_module.query(
        test_question,
        retrieve_k=10,
        final_k=3,
        use_rerank=False
    )
    
    # 生成回答
    result = response_module.generate_answer_with_sources(
        test_question,
        retrieved_docs,
        show_sources=True
    )
    
    # 显示结果
    response_module.display_answer(result)