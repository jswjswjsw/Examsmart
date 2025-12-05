import os
import json
import shutil
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb


def load_chunks(filepath: str) -> List[Dict[str, Any]]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_env_for_siliconflow():
    load_dotenv()
    silicon_key = os.getenv('SILICON_API_KEY')
    silicon_base = os.getenv('SILICON_BASE_URL')
    if not silicon_key or not silicon_base:
        raise RuntimeError('缺少硅基流动 API 配置，请在 .env 中设置 SILICON_API_KEY 和 SILICON_BASE_URL')

    os.environ['OPENAI_API_KEY'] = silicon_key
    os.environ['OPENAI_API_BASE'] = silicon_base
    os.environ['OPENAI_BASE_URL'] = silicon_base


def build_embeddings_model():
    return OpenAIEmbeddings(model='BAAI/bge-m3')


def build_persist_dir() -> str:
    base = os.path.join(os.path.dirname(__file__), '..', 'target')
    persist_dir = os.path.join(base, 'chroma')
    os.makedirs(persist_dir, exist_ok=True)
    return persist_dir


def reset_persist_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def to_texts_and_metadata(chunks: List[Dict[str, Any]]):
    def sanitize(meta: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in (meta or {}).items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = v
            elif isinstance(v, (list, tuple)):
                try:
                    out[k] = ' | '.join(str(x) for x in v)
                except Exception:
                    out[k] = str(v)
            elif isinstance(v, dict):
                out[k] = json.dumps(v, ensure_ascii=False)
            else:
                out[k] = str(v)
        return out
    texts = [c['content'] for c in chunks]
    metadatas = [sanitize(c.get('metadata', {})) for c in chunks]
    ids = [f"chunk_{c.get('chunk_id', i)}" for i, c in enumerate(chunks)]
    return texts, metadatas, ids


def main():
    chunks_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json')
    print(f"加载分块数据: {chunks_path}")
    if not os.path.exists(chunks_path):
        print('未找到 chunks.json，请先运行 1_dataTreating.py 生成分块数据')
        return

    chunks = load_chunks(chunks_path)
    print(f"分块数量: {len(chunks)}")

    ensure_env_for_siliconflow()
    embeddings = build_embeddings_model()

    persist_dir = build_persist_dir()
    collection_name = 'rag_material'

    # Skip if already processed
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        existing = None
        for c in client.list_collections():
            if getattr(c, 'name', None) == collection_name:
                existing = c
                break
        if existing:
            print(f"检测到集合 {collection_name} 已存在，当前条数 {existing.count()}，将执行 upsert 以更新内容。")
    except Exception:
        existing = None

    texts, metadatas, ids = to_texts_and_metadata(chunks)
    print('开始计算嵌入并写入 ChromaDB（可能需数分钟）...')
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name)

    def batches(seq, size):
        for i in range(0, len(seq), size):
            yield i, seq[i:i + size]

    batch_size = 64
    for idx, batch_texts in batches(texts, batch_size):
        batch_metas = metadatas[idx: idx + batch_size]
        batch_ids = ids[idx: idx + batch_size]
        batch_vectors = embeddings.embed_documents(batch_texts)
        try:
            # Prefer upsert if available
            if hasattr(collection, 'upsert'):
                collection.upsert(documents=batch_texts, metadatas=batch_metas, ids=batch_ids, embeddings=batch_vectors)
            else:
                collection.add(documents=batch_texts, metadatas=batch_metas, ids=batch_ids, embeddings=batch_vectors)
        except Exception:
            # Fallback to add
            collection.add(documents=batch_texts, metadatas=batch_metas, ids=batch_ids, embeddings=batch_vectors)

    print(f"向量库已持久化到: {persist_dir}")

    summary = {
        'collection': collection_name,
        'count': collection.count(),
        'persist_directory': os.path.abspath(persist_dir),
        'embedding_model': 'BAAI/bge-m3',
        'provider': 'SiliconFlow (OpenAI兼容API)'
    }

    summary_path = os.path.join(os.path.dirname(__file__), '..', 'target', 'embedding_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"摘要信息已保存到: {summary_path}")


if __name__ == '__main__':
    main()
