import os
import json
import base64
import argparse
from typing import List, Dict, Any, Optional
import requests
from dotenv import load_dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import time
import shlex
import math

RAG_POLICY_PROMPT = (
    "# 身份与目标 (Identity and Goal)\n"
    "你是一位考试信息和政策咨询助手。你的核心使命是准确、权威、严谨地解读考试信息和政策，帮助考生理解复杂的规定。"
    "你的回答必须 100% 来源于提供的知识库内容。\n"
    "---\n"
    "# 约束规则 (Constraints and Rules)\n"
    "1. 信息来源唯一性：所有回复内容必须严格基于知识库中检索到的信息。\n"
    "2. 避免猜测与幻想：禁止进行任何推测、编造或提供知识库中不存在的个人意见、学习技巧或职业建议。\n"
    "   如果知识库中没有相关信息，请礼貌地说明：\"很抱歉，我的知识库中没有找到关于您这个问题的明确官方记录。\"\n"
    "3. 语气与风格：保持专业、客观、严谨的语气。用清晰的白话解读政策，但不能过于随意。\n"
    "4. 实时性强调：对于涉及时间、日期、地点和费用的回答，务必提供最新可核实的年份和信息，并在回答结尾提醒：\"请始终以官方最新发布的通知为准。\"\n"
    "---\n"
    "# 知识库利用与输出格式 (RAG & Output Format)\n"
    "当用户提问时，必须按以下步骤执行和格式化输出：\n"
    "1. 检索优先：将用户问题转化为检索关键词，在知识库中找到最相关、最具权威性的 1 到 3 个文本块。\n"
    "2. 整合与提炼：将检索到的信息整合提炼，形成一个连贯、完整且直达核心的答案。\n"
    "3. 关键信息突出：在最终回复中，使用**粗体**突出显示所有日期、数字、官方链接、核心要求等关键信息。\n"
    "4. 提供依据（重要）：在回答的最后，用小括号或引用形式，标明回答的原始政策文件/条款依据（给出来源与页码）。\n"
    "5. 当上下文非空时，必须基于上下文给出结论或条件性说明，不得整体拒答或输出\"未找到\"。\n"
)


_ENV_READY = False
_COLLECTION = None
_EMBEDDER = None
_CHAT_SESSION = None
_RERANK_SESSION = None
_IMAGE_SUMMARY_CACHE = {}
_QUERY_EXPAND_CACHE = {}


def _ensure_env():
    global _ENV_READY
    if _ENV_READY:
        return
    load_dotenv()
    silicon_key = os.getenv('SILICON_API_KEY')
    silicon_base = os.getenv('SILICON_BASE_URL')
    openai_key = os.getenv('OPENAI_API_KEY')
    openai_base = os.getenv('OPENAI_BASE_URL')
    if not silicon_key or not silicon_base:
        raise RuntimeError('缺少硅基流动配置')
    if not openai_key or not openai_base:
        raise RuntimeError('缺少OpenAI配置')
    _ENV_READY = True


def _get_collection():
    global _COLLECTION
    if _COLLECTION is not None:
        return _COLLECTION
    base = os.path.join(os.path.dirname(__file__), '..', 'target')
    persist_dir = os.path.join(base, 'chroma')
    client = chromadb.PersistentClient(path=persist_dir)
    _COLLECTION = client.get_or_create_collection(name='rag_material')
    return _COLLECTION


def _get_embeddings():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER
    _EMBEDDER = OpenAIEmbeddings(model='BAAI/bge-m3', api_key=os.getenv('SILICON_API_KEY'), base_url=os.getenv('SILICON_BASE_URL'))
    return _EMBEDDER


def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s


def _get_chat_session():
    global _CHAT_SESSION
    if _CHAT_SESSION is None:
        _CHAT_SESSION = _build_session()
    return _CHAT_SESSION


def _get_rerank_session():
    global _RERANK_SESSION
    if _RERANK_SESSION is None:
        _RERANK_SESSION = _build_session()
    return _RERANK_SESSION


def _read_image_as_data_url(path: str) -> str:
    if path.startswith('http://') or path.startswith('https://'):
        r = requests.get(path, timeout=30)
        r.raise_for_status()
        b64 = base64.b64encode(r.content).decode('utf-8')
        ctype = r.headers.get('Content-Type', '')
        mime = ctype if ctype.startswith('image/') else 'image/jpeg'
        return f'data:{mime};base64,{b64}'
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    ext = os.path.splitext(path)[1].lower().strip('.') or 'png'
    mime = 'image/png' if ext in ['png'] else 'image/jpeg'
    return f'data:{mime};base64,{b64}'


def _openai_chat(messages: List[Dict[str, Any]], model: str) -> str:
    key = os.getenv('OPENAI_API_KEY')
    base = os.getenv('OPENAI_BASE_URL')
    url = base.rstrip('/') + '/chat/completions'
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    payload = {'model': model, 'messages': messages}
    resp = _get_chat_session().post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data['choices'][0]['message']['content']


def _describe_images(paths: List[str]) -> str:
    content = []
    for p in paths:
        try:
            content.append({'type': 'image_url', 'image_url': {'url': _read_image_as_data_url(p)}})
        except Exception:
            continue
    content.insert(0, {'type': 'text', 'text': '请提取图片中的关键信息（实体、文字、数值、表格）用于检索，输出简洁中文要点。'})
    messages = [{'role': 'user', 'content': content}]
    return _openai_chat(messages, model='gpt-4o-mini')


def _condense_query(user_text: str, image_desc: str = '') -> str:
    prompt = {
        'role': 'user',
        'content': f"将问题改写为适合检索的简短表达，保留核心名词与动词，输出不超过30字的中文：\n问题：{user_text}\n图片要点：{image_desc}"
    }
    return _openai_chat([prompt], model='gpt-4o-mini')


def _expand_queries(user_text: str, image_desc: str = '') -> List[str]:
    base = user_text.strip()
    condensed = _condense_query(user_text, image_desc).strip()
    prompt = {
        'role': 'user',
        'content': f"请给出该问题的3-5个中文同义检索式，每个一行，仅输出短语。\n问题：{user_text}\n图片要点：{image_desc}"
    }
    syn = _openai_chat([prompt], model='gpt-4o-mini')
    syn_list = [s.strip() for s in syn.split('\n') if s.strip()]
    uniq = []
    for q in [base, condensed] + syn_list:
        if q and q not in uniq:
            uniq.append(q)
    return uniq[:6]


def _initial_retrieve(col, embedder, query_texts: List[str], top_k: int):
    pool: Dict[str, Dict[str, Any]] = {}
    qvs = _get_embeddings().embed_documents(query_texts)
    for idx, qv in enumerate(qvs):
        r = col.query(query_embeddings=[qv], n_results=top_k, include=['documents', 'metadatas', 'distances'])
        docs = r.get('documents', [[]])[0]
        metas = r.get('metadatas', [[]])[0]
        ids = r.get('ids', [[]])[0]
        dists = r.get('distances', [[]])[0]
        for i in range(len(docs)):
            sim = 1.0 - float(dists[i]) if dists and i < len(dists) else 0.0
            prev = pool.get(ids[i])
            if prev is None or sim > prev.get('embed_similarity', -1.0):
                pool[ids[i]] = {'id': ids[i], 'text': docs[i], 'metadata': metas[i], 'embed_similarity': sim}
    return list(pool.values())


def _retrieve_by_meta(col, query_texts: List[str], top_k: int, where: Dict[str, Any], source_type: str) -> List[Dict[str, Any]]:
    pool: Dict[str, Dict[str, Any]] = {}
    qvs = _get_embeddings().embed_documents(query_texts)
    for qv in qvs:
        r = col.query(query_embeddings=[qv], n_results=top_k, include=['documents', 'metadatas', 'distances'], where=where)
        docs = r.get('documents', [[]])[0]
        metas = r.get('metadatas', [[]])[0]
        ids = r.get('ids', [[]])[0]
        dists = r.get('distances', [[]])[0]
        for i in range(len(docs)):
            sim = 1.0 - float(dists[i]) if dists and i < len(dists) else 0.0
            prev = pool.get(ids[i])
            item = {'id': ids[i], 'text': docs[i], 'metadata': metas[i], 'embed_similarity': sim, 'source_type': source_type}
            if prev is None or sim > prev.get('embed_similarity', -1.0):
                pool[ids[i]] = item
    return list(pool.values())


def _rerank_with_siliconflow(query: str, docs: List[Dict[str, Any]]):
    key = os.getenv('SILICON_API_KEY')
    base = os.getenv('SILICON_BASE_URL')
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    documents = [d['text'] for d in docs]
    payload = {'model': 'BAAI/bge-reranker-v2-m3', 'query': query, 'documents': documents}
    for endpoint in ['/v1/rerank', '/rerank']:
        try:
            url = base.rstrip('/') + endpoint
            resp = _get_rerank_session().post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get('results') or data.get('data') or []
            if isinstance(results, list) and results:
                def sigmoid(x: float) -> float:
                    try:
                        return 1.0 / (1.0 + math.exp(-x))
                    except OverflowError:
                        return 0.0 if x < 0 else 1.0
                scored = []
                for i, item in enumerate(results):
                    if isinstance(item, dict):
                        score = item.get('score') or item.get('relevance', item.get('relevance_score'))
                        idx = item.get('index', i)
                    else:
                        score = None
                        idx = i
                    s = float(score) if score is not None else 0.0
                    scored.append((idx, s, sigmoid(s)))
                scored.sort(key=lambda x: x[1], reverse=True)
                enriched = []
                for idx, s, p in scored:
                    d = docs[idx].copy()
                    d['rerank_score'] = s
                    d['rerank_prob'] = p
                    enriched.append(d)
                return enriched
        except Exception:
            continue
    return docs

def _extract_keywords(text: str) -> List[str]:
    tokens = []
    buf = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff' or ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append(''.join(buf))
                buf = []
    if buf:
        tokens.append(''.join(buf))
    return [t for t in tokens if len(t) >= 2]


def _load_chunks(filepath: str) -> List[Dict[str, Any]]:
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def _keyword_fallback(user_text: str, limit: int = 30) -> List[Dict[str, Any]]:
    kws = _extract_keywords(user_text)
    if not kws:
        return []
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json')
    chunks = _load_chunks(path)
    results = []
    for c in chunks:
        content = c.get('content', '')
        hits = sum(1 for k in kws if k in content)
        if hits > 0:
            results.append({
                'id': f"chunk_{c.get('chunk_id')}",
                'text': content,
                'metadata': c.get('metadata', {}),
                'embed_similarity': 0.0
            })
            if len(results) >= limit:
                break
    return results


def _select_contexts(query: str, reranked: List[Dict[str, Any]], k: int = 5) -> List[Dict[str, Any]]:
    kws = _extract_keywords(query)
    by_score = sorted(reranked, key=lambda d: d.get('final_score', 0.0), reverse=True)
    chosen = by_score[:k]
    def has_kw(d):
        t = d.get('text', '')
        return any(k in t for k in kws)
    if kws:
        kw_docs = [d for d in by_score if has_kw(d)]
        if kw_docs and not any(has_kw(d) for d in chosen):
            chosen = chosen[:-1] + [kw_docs[0]]
    # 保证至少一条QA进入上下文
    def is_qa(d):
        return (d.get('metadata', {}).get('type') == 'qa_pair') or (d.get('source_type') == 'qa')
    def is_section(d):
        return (d.get('metadata', {}).get('type') == 'section_chunk') or (d.get('source_type') == 'section')
    if not any(is_qa(d) for d in chosen):
        for d in by_score:
            if is_qa(d):
                chosen = chosen[:-1] + [d]
                break
    if not any(is_section(d) for d in chosen):
        for d in by_score:
            if is_section(d):
                chosen = chosen[:-1] + [d]
                break
    return chosen


def _bm25_fallback(user_text: str, limit: int = 50) -> List[Dict[str, Any]]:
    def tok(s: str) -> List[str]:
        out = []
        buf = []
        for ch in s:
            if '\u4e00' <= ch <= '\u9fff' or ch.isalnum():
                buf.append(ch)
            else:
                if buf:
                    out.append(''.join(buf))
                    buf = []
        if buf:
            out.append(''.join(buf))
        return out
    q_tokens = [t for t in tok(user_text) if len(t) >= 2]
    if not q_tokens:
        return []
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json')
    docs = _load_chunks(path)
    if not docs:
        return []
    corpus = []
    for d in docs:
        t = tok(d.get('content', ''))
        corpus.append(t)
    N = len(corpus)
    avgdl = sum(len(dt) for dt in corpus) / max(1, N)
    df = {}
    for k in q_tokens:
        c = 0
        for dt in corpus:
            if k in dt:
                c += 1
        df[k] = c
    import math as _m
    idf = {k: _m.log((N - df[k] + 0.5) / (df[k] + 0.5) + 1.0) for k in q_tokens}
    k1 = 1.5
    b = 0.75
    scores = []
    for i, dt in enumerate(corpus):
        s = 0.0
        dl = len(dt)
        for k in q_tokens:
            tf = dt.count(k)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * (dl / max(1.0, avgdl)))
            s += idf[k] * ((tf * (k1 + 1)) / denom)
        if s > 0:
            scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    out = []
    for i, s in scores[:limit]:
        c = docs[i]
        out.append({
            'id': f"chunk_{c.get('chunk_id')}",
            'text': c.get('content', ''),
            'metadata': c.get('metadata', {}),
            'embed_similarity': 0.0
        })
    return out


def _build_answer(query: str, reranked: List[Dict[str, Any]], model: str = 'gpt-4o-mini') -> str:
    if not reranked:
        return "很抱歉，我的知识库中没有找到关于您这个问题的明确官方记录。请始终以官方最新发布的通知为准。"
    top_ctx = _select_contexts(query, reranked, k=5)
    context = []
    for i, d in enumerate(top_ctx, 1):
        page = d.get('metadata', {}).get('page', '?')
        context.append(f"[{i}] 来源: material.json 第{page}页\n{d['text']}")
    ctx_text = '\n\n'.join(context)
    system = {'role': 'system', 'content': RAG_POLICY_PROMPT}
    user = {'role': 'user', 'content': f"用户问题:\n{query}\n\n知识库片段(最多5条):\n{ctx_text}\n\n仅基于以上片段作答；若缺少关键字段，请指出缺少项，但不得整体拒答；在结尾附上依据（来源与页码）以及提示：请始终以官方最新发布的通知为准。"}
    return _openai_chat([system, user], model=model)


def query_rag(user_text: str, image_paths: Optional[List[str]] = None, top_k: int = 50, rerank_k: int = 10, alpha: float = 0.7, pool_k: int = 80, fast_mode: bool = True, qa_top_k: int = 40, bg_top_k: int = 30) -> Dict[str, Any]:
    _ensure_env()
    col = _get_collection()
    embedder = _get_embeddings()
    image_desc = ''
    if image_paths:
        key = tuple(image_paths)
        if key in _IMAGE_SUMMARY_CACHE:
            image_desc = _IMAGE_SUMMARY_CACHE[key]
        elif not fast_mode:
            image_desc = _describe_images(image_paths)
            _IMAGE_SUMMARY_CACHE[key] = image_desc
    else:
        image_desc = ''
    qkey = (user_text.strip(), image_desc.strip(), fast_mode)
    if qkey in _QUERY_EXPAND_CACHE:
        queries = _QUERY_EXPAND_CACHE[qkey]
    else:
        if fast_mode:
            condensed = _condense_query(user_text, image_desc).strip()
            queries = [user_text.strip()] + ([condensed] if condensed and condensed != user_text.strip() else [])
        else:
            queries = _expand_queries(user_text, image_desc)
        _QUERY_EXPAND_CACHE[qkey] = queries
    final_query = queries[0]
    t0 = time.time()
    qa_initial = _retrieve_by_meta(col, queries, qa_top_k, where={'type': 'qa_pair'}, source_type='qa')
    section_initial = _retrieve_by_meta(col, queries, max(10, bg_top_k // 2), where={'type': 'section_chunk'}, source_type='section')
    bg_initial = _retrieve_by_meta(col, queries, max(5, bg_top_k // 2), where={'type': {'$ne': 'qa_pair'}}, source_type='bg')
    # 合并去重
    seen = {}
    for d in qa_initial + section_initial + bg_initial:
        if d['id'] not in seen or d['embed_similarity'] > seen[d['id']]['embed_similarity']:
            seen[d['id']] = d
    initial = list(seen.values())
    if len(initial) > pool_k:
        initial.sort(key=lambda x: x.get('embed_similarity', 0.0), reverse=True)
        initial = initial[:pool_k]
    t1 = time.time()
    # 关键词兜底：若召回数量偏少或不含关键词，进行词面匹配补充
    need_fallback = len(initial) < max(10, rerank_k)
    if need_fallback:
        fb = _keyword_fallback(user_text)
        if fb:
            # 合并去重
            seen = {d['id'] for d in initial}
            initial.extend([d for d in fb if d['id'] not in seen])
        bm = _bm25_fallback(user_text)
        if bm:
            seen = {d['id'] for d in initial}
            initial.extend([d for d in bm if d['id'] not in seen])
    reranked = _rerank_with_siliconflow(final_query, initial)
    kws = _extract_keywords(user_text)
    kw_total = max(1, len(kws))
    # 词面评分（基于初检候选）
    # 计算基于候选集合的IDF
    df = {k: 0 for k in kws}
    for d in initial:
        text = d.get('text', '')
        for k in kws:
            if k in text:
                df[k] += 1
    N = max(1, len(initial))
    idf = {k: math.log((N + 1) / (df[k] + 1)) for k in kws}
    # 计算每条的lex分
    max_lex = 0.0
    lex_scores = {}
    for d in initial:
        text = d.get('text', '')
        s = 0.0
        for k in kws:
            tf = text.count(k)
            if tf > 0:
                s += tf * idf[k]
        lex_scores[d['id']] = s
        if s > max_lex:
            max_lex = s
    # 评分融合
    for d in reranked:
        e = float(d.get('embed_similarity', 0.0))
        r = float(d.get('rerank_prob', d.get('rerank_score', 0.0)))
        l = lex_scores.get(d['id'], 0.0)
        l_norm = (l / max_lex) if max_lex > 0 else 0.0
        e_norm = ((e + 1.0) / 2.0)
        kw_hits = sum(1 for k in kws if k in d.get('text', '')) / kw_total
        meta_type = d.get('metadata', {}).get('type')
        src_type = d.get('source_type')
        qa_boost = 1.0 if (meta_type == 'qa_pair' or src_type == 'qa') else 0.0
        section_boost = 0.5 if (meta_type == 'section_chunk' or src_type == 'section') else 0.0
        beta = 0.2
        gamma = 0.2
        delta_kw = 0.1
        epsilon_qa = 0.1
        zeta_section = 0.05
        d['final_score'] = alpha * r + beta * e_norm + gamma * l_norm + delta_kw * kw_hits + epsilon_qa * qa_boost + zeta_section * section_boost
    reranked.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
    reranked = reranked[:rerank_k]
    t2 = time.time()
    answer = _build_answer(final_query, reranked)
    t3 = time.time()
    return {'query': final_query, 'answer': answer, 'contexts': reranked, 'timing': {'retrieve_ms': int((t1 - t0) * 1000), 'rerank_ms': int((t2 - t1) * 1000), 'generate_ms': int((t3 - t2) * 1000), 'total_ms': int((t3 - t0) * 1000)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str)
    parser.add_argument('--images', nargs='*')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--rerank_k', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--pool_k', type=int, default=80)
    parser.add_argument('--qa_top_k', type=int, default=40)
    parser.add_argument('--bg_top_k', type=int, default=30)
    parser.add_argument('--fast_mode', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    args = parser.parse_args()
    q = args.query
    if q:
        result = query_rag(q, args.images, args.top_k, args.rerank_k, args.alpha, args.pool_k, bool(args.fast_mode), args.qa_top_k, args.bg_top_k)
        out = {'query': result['query'], 'answer': result['answer'], 'contexts': result['contexts'], 'timing': result.get('timing')}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    print('进入交互模式，输入问题后回车；输入 exit/quit/q 退出；输入 :images 路径… 设置会话图片；:add_images 路径… 追加；:clear_images 清空；:list_images 查看；:fast_on/:fast_off；:set <key> <value>')
    session_images = list(args.images or [])
    fast_mode = bool(args.fast_mode)
    top_k = args.top_k
    rerank_k = args.rerank_k
    alpha = args.alpha
    pool_k = args.pool_k
    qa_top_k = args.qa_top_k
    bg_top_k = args.bg_top_k
    while True:
        print('> ', end='')
        try:
            line = input()
        except EOFError:
            break
        if not line:
            continue
        t = line.strip()
        if t.lower() in ('exit', 'quit', 'q'):
            break
        if t.startswith(':'):
            parts = shlex.split(t)
            cmd = parts[0].lower()
            if cmd == ':images':
                session_images = [p for p in parts[1:] if p.startswith('http') or os.path.exists(p)]
                print(f'已设置图片 {len(session_images)} 个')
                continue
            if cmd == ':add_images':
                add = [p for p in parts[1:] if p.startswith('http') or os.path.exists(p)]
                session_images.extend(add)
                print(f'已追加图片，当前 {len(session_images)} 个')
                continue
            if cmd == ':clear_images':
                session_images = []
                print('已清空图片')
                continue
            if cmd == ':list_images':
                print(json.dumps(session_images, ensure_ascii=False))
                continue
            if cmd == ':fast_on':
                fast_mode = True
                print('快速模式已开启')
                continue
            if cmd == ':fast_off':
                fast_mode = False
                print('快速模式已关闭')
                continue
            if cmd == ':set' and len(parts) >= 3:
                key = parts[1]
                val = parts[2]
                if key == 'top_k':
                    top_k = int(val)
                elif key == 'rerank_k':
                    rerank_k = int(val)
                elif key == 'alpha':
                    alpha = float(val)
                elif key == 'pool_k':
                    pool_k = int(val)
                elif key == 'qa_top_k':
                    qa_top_k = int(val)
                elif key == 'bg_top_k':
                    bg_top_k = int(val)
                print(f'已设置 {key}={val}')
                continue
        try:
            result = query_rag(t, session_images, top_k, rerank_k, alpha, pool_k, fast_mode, qa_top_k, bg_top_k)
            out = {'query': result['query'], 'answer': result['answer'], 'contexts': result['contexts'], 'timing': result.get('timing')}
            if args.debug:
                out['debug'] = {
                    'queries': _QUERY_EXPAND_CACHE.get((t.strip(), '', fast_mode), []),
                    'selected_ids': [c.get('id') for c in result['contexts']]
                }
            print(json.dumps(out, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f'错误: {e}')


if __name__ == '__main__':
    main()
