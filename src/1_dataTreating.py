import json
import re
import os
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def simple_html_table_to_markdown(html: str) -> str:
    """
    Converts a simple HTML table to a Markdown-like text representation.
    """
    # Remove newlines in HTML to simplify regex
    html = html.replace('\n', '')
    
    # Extract rows
    rows = re.findall(r'<tr>(.*?)</tr>', html)
    
    markdown_rows = []
    for row in rows:
        # Extract cells
        cells = re.findall(r'<td>(.*?)</td>', row)
        # Clean tags from cells if any
        cleaned_cells = [re.sub(r'<.*?>', '', cell).strip() for cell in cells]
        # Join with pipe
        markdown_row = "| " + " | ".join(cleaned_cells) + " |"
        markdown_rows.append(markdown_row)
        
    return "\n".join(markdown_rows)

def extract_content_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten the JSON structure into a linear list of content items.
    Each item: {'type': 'text'|'table', 'content': str, 'page': int}
    """
    items = []
    if 'pdf_info' not in data:
        return items

    for page_idx, page in enumerate(data['pdf_info']):
        if 'para_blocks' in page:
            for block in page['para_blocks']:
                block_type = block.get('type')
                
                if block_type == 'text':
                    text_parts = []
                    if 'lines' in block:
                        for line in block['lines']:
                            if 'spans' in line:
                                for span in line['spans']:
                                    if 'content' in span:
                                        text_parts.append(span['content'])
                    if text_parts:
                        items.append({
                            'type': 'text',
                            'content': "".join(text_parts),
                            'page': page_idx + 1
                        })
                        
                elif block_type == 'table':
                    if 'blocks' in block:
                        for sub_block in block['blocks']:
                            if sub_block.get('type') == 'table_caption':
                                caption_parts = []
                                if 'lines' in sub_block:
                                    for line in sub_block['lines']:
                                        if 'spans' in line:
                                            for span in line['spans']:
                                                if 'content' in span:
                                                    caption_parts.append(span['content'])
                                if caption_parts:
                                    items.append({
                                        'type': 'text',
                                        'content': "Table Caption: " + "".join(caption_parts),
                                        'page': page_idx + 1
                                    })
                                    
                            elif sub_block.get('type') == 'table_body':
                                if 'lines' in sub_block:
                                    for line in sub_block['lines']:
                                        if 'spans' in line:
                                            for span in line['spans']:
                                                if 'html' in span:
                                                    md_table = simple_html_table_to_markdown(span['html'])
                                                    items.append({
                                                        'type': 'table',
                                                        'content': md_table,
                                                        'page': page_idx + 1
                                                    })
    return items

def detect_sections(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Detect major/minor section headers and annotate items with section_path
    major_pat = re.compile(r'^[一二三四五六七八九十百]+、\s*.*')
    minor_pat = re.compile(r'^[（\(][一二三四五六七八九十]+[）\)]\s*.*')
    # Some keyword-based headings
    kw_heads = [
        '报名篇', '录取篇', '志愿', '材料', '档案', '补贴', '春考', '秋考', '艺考', '校考', '资格', '政策', '办理', '确认'
    ]
    current_major = ''
    current_minor = ''
    annotated: List[Dict[str, Any]] = []
    for it in items:
        text = it.get('content', '').strip()
        is_major = bool(major_pat.match(text))
        is_minor = bool(minor_pat.match(text))
        if is_major:
            current_major = text
            current_minor = ''
        elif is_minor:
            current_minor = text
        else:
            # Heuristic: extremely short lines containing known headings
            if any(k in text for k in kw_heads) and len(text) <= 20:
                # Prefer set as minor under current major
                current_minor = text
        path: List[str] = []
        if current_major:
            path.append(current_major)
        if current_minor:
            path.append(current_minor)
        new_it = dict(it)
        new_it['section_path'] = path
        annotated.append(new_it)
    return annotated

def segment_qa_pairs(items: List[Dict[str, Any]]) -> List[Document]:
    """
    Segment content into QA pairs.
    Strategies:
    1. Identify 'Question' start by regex `^(\d+\.\s*)?问[：:]`
    2. Identify 'Answer' start by regex `^答[：:]`
    3. Group content between Q and A as 'Question' part.
    4. Group content after A until next Q as 'Answer' part.
    5. Non-QA content is treated as standalone chunks.
    """
    
    # Regex patterns
    q_pattern = re.compile(r'^(\d+\.\s*)?问[：:]')
    a_pattern = re.compile(r'^\s*答[：:]')
    
    documents = []
    
    # State variables
    current_qa = None # {'question': [], 'answer': [], 'page': int, 'section_path': List[str], 'qa_id': int}
    non_qa_buffer = [] # For content that is not part of any QA
    
    state = 'IDLE' # IDLE, IN_QUESTION, IN_ANSWER

    qa_counter = 0

    def flush_qa(qa_obj):
        if qa_obj and (qa_obj['question'] or qa_obj['answer']):
            # Join parts
            q_text = "\n".join(qa_obj['question'])
            a_text = "\n".join(qa_obj['answer'])
            
            # Construct full text
            # If Q and A exist, format nicely
            if q_text and a_text:
                full_text = f"{q_text}\n{a_text}"
            elif q_text:
                 full_text = q_text
            else:
                 full_text = a_text
                 
            doc = Document(
                page_content=full_text,
                metadata={"page": qa_obj['page'], "source": "material.json", "type": "qa_pair", "qa_id": qa_obj.get('qa_id'), "section_path": qa_obj.get('section_path', [])}
            )
            documents.append(doc)

    def flush_buffer(buffer):
        if buffer:
            text = "\n".join([item['content'] for item in buffer])
            if text.strip():
                # For non-QA content, we might want to split it further if it's too long,
                # but here we just create a base document to be split later if needed.
                # Or we can rely on the standard splitter for these.
                doc = Document(
                    page_content=text,
                    metadata={"page": buffer[0]['page'], "source": "material.json", "type": "general_text"}
                )
                documents.append(doc)
    
    current_section: List[str] = []
    for item in items:
        content = item['content']
        page = item['page']
        current_section = item.get('section_path', current_section)
        
        is_q_start = q_pattern.match(content)
        is_a_start = a_pattern.match(content)
        
        if is_q_start:
            # New Question found
            if state == 'IN_QUESTION' or state == 'IN_ANSWER':
                # Close previous QA
                flush_qa(current_qa)
            elif state == 'IDLE':
                # Flush any non-QA buffer
                flush_buffer(non_qa_buffer)
                non_qa_buffer = []
            
            # Start new QA
            qa_counter += 1
            current_qa = {'question': [content], 'answer': [], 'page': page, 'section_path': current_section, 'qa_id': qa_counter}
            state = 'IN_QUESTION'
            
        elif is_a_start:
            # Answer start found
            if state == 'IN_QUESTION':
                # Perfect, switch to answer
                current_qa['answer'].append(content)
                state = 'IN_ANSWER'
            elif state == 'IN_ANSWER':
                # Another answer block? Append to current answer
                current_qa['answer'].append(content)
            elif state == 'IDLE':
                # Answer without Question? Treat as general text
                non_qa_buffer.append(item)
                
        else:
            # Normal content (text or table)
            if state == 'IN_QUESTION':
                # Continuation of Question
                current_qa['question'].append(content)
            elif state == 'IN_ANSWER':
                # Continuation of Answer
                current_qa['answer'].append(content)
            else: # IDLE
                non_qa_buffer.append(item)
                
    # Flush remaining
    if state == 'IN_QUESTION' or state == 'IN_ANSWER':
        flush_qa(current_qa)
    elif non_qa_buffer:
        flush_buffer(non_qa_buffer)
        
    return documents

def aggregate_by_section(qa_docs: List[Document], max_len: int = 1800) -> List[Document]:
    grouped: Dict[Tuple[str, ...], List[Document]] = {}
    for d in qa_docs:
        path = tuple(d.metadata.get('section_path', []) or [])
        grouped.setdefault(path, []).append(d)
    out: List[Document] = []
    for path, docs in grouped.items():
        buf = []
        ids = []
        pages = []
        def flush_buf():
            if not buf:
                return
            text = "\n\n".join(buf)
            meta = {
                "source": "material.json",
                "type": "section_chunk",
                "section_path": list(path),
                "qa_ids": ids.copy(),
            }
            if pages:
                meta["page_range"] = [min(pages), max(pages)]
            out.append(Document(page_content=text, metadata=meta))
        for d in docs:
            t = d.page_content
            qid = d.metadata.get('qa_id')
            pg = d.metadata.get('page')
            next_len = (len("\n\n".join(buf)) if buf else 0) + len(t)
            if buf and next_len > max_len:
                flush_buf()
                buf = []
                ids = []
                pages = []
            buf.append(t)
            if qid is not None:
                ids.append(qid)
            if pg is not None:
                pages.append(pg)
        flush_buf()
    return out

def chunk_documents(documents: List[Document]) -> List[Document]:
    qa_docs = [doc for doc in documents if doc.metadata.get('type') == 'qa_pair']
    general_docs = [doc for doc in documents if doc.metadata.get('type') == 'general_text']
    final_chunks: List[Document] = []
    # 1. Aggregated section chunks
    section_chunks = aggregate_by_section(qa_docs, max_len=1800)
    final_chunks.extend(section_chunks)
    # 2. General text splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    if general_docs:
        final_chunks.extend(text_splitter.split_documents(general_docs))
    # 3. QA docs kept as atomic (split if extremely long)
    qa_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    for doc in qa_docs:
        if len(doc.page_content) > 1000:
            final_chunks.extend(qa_splitter.split_documents([doc]))
        else:
            final_chunks.append(doc)
    return final_chunks

def main():
    input_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'material.json')
    output_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json')
    
    print(f"Loading data from {input_file}...")
    if not os.path.exists(input_file):
        print("File not found!")
        return

    data = load_json(input_file)
    
    print("Extracting content items...")
    items = extract_content_items(data)
    print(f"Extracted {len(items)} raw items.")
    print("Detecting sections and annotating...")
    items = detect_sections(items)
    print("Segmenting QA pairs...")
    documents = segment_qa_pairs(items)
    print(f"Identified {len(documents)} logical documents (QA pairs + general text).")
    
    print("Final Chunking...")
    chunks = chunk_documents(documents)
    print(f"Generated {len(chunks)} chunks.")
    
    # Save chunks
    output_data = []
    for i, chunk in enumerate(chunks):
        output_data.append({
            "chunk_id": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
        
    print(f"Chunks saved to {output_file}")
    
    # Print sample QA
    qa_samples = [c for c in chunks if c.metadata.get('type') == 'qa_pair']
    if qa_samples:
        print("\n--- Sample QA Chunk ---")
        print(qa_samples[0].page_content)
        print("-----------------------")

if __name__ == "__main__":
    main()
