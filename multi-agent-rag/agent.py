"""
基於Agent架構的溯源多模態RAG系統
使用SequentialAgent組織RAG流程：資料加載 → 向量化 → 檢索 → 回答生成
"""

import os
import json
from typing import Dict, Any
import numpy as np
import faiss
import mimetypes
from google.adk.agents import Agent, SequentialAgent, ParallelAgent
from google.adk.models.lite_llm import LiteLlm

from dotenv import load_dotenv
load_dotenv(override=True)
QWEN_MODEL = "openai/qwen3-vl:235b-cloud"
QWEN_BASE_URL='http://localhost:11434/v1'
QWEN_API_KEY='your_api_key_here'

# 全域FAISS儲存
_FAISS_STORAGE = {
    "text_index": None,
    "insight_index": None,
    "text_chunks": None,
    "insights": None,
    "rag_data": None
}

# ================ 工具函式定義 ================

def load_rag_data_tool(json_data: str) -> Dict[str, Any]:
    """RAG資料加載工具 - 處理JSON字串"""
    print(f"[工具] 載入RAG資料，資料長度: {len(json_data)} 字元")
    
    try:
        # 解析JSON字串
        data = json.loads(json_data)
        
        # 統計資料量
        text_count = len(data.get("text_chunks_with_tracing", []))
        table_count = len(data.get("table_data", []))
        chart_count = len(data.get("chart_data", []))
        insights_count = len(data.get("insights_with_source", []))
        
        load_result = {
            "load_status": "success",
            "data_summary": {
                "text_chunks_count": text_count,
                "table_count": table_count,
                "chart_count": chart_count,
                "insights_count": insights_count
            },
            "document_info": {
                "document_type": data.get("document_classification", {}).get("document_name", "未知"),
                "summary": data.get("document_summary", "無概要"),
                "confidence": data.get("document_classification", {}).get("confidence", 0.0)
            },
            "rag_data": data,  # 完整資料儲存
            "ready_for_vectorization": True,
            "message": f"資料載入成功: {text_count}個文本塊, {table_count}個表格, {insights_count}個洞察"
        }
        
        print(f"[工具] 資料載入成功: {text_count}個文本塊")
        return load_result
        
    except Exception as e:
        print(f"[工具] 資料載入失敗: {str(e)}")
        return {
            "load_status": "failed",
            "error": str(e),
            "ready_for_vectorization": False
        }

def vectorize_content_tool(rag_data: Dict[str, Any]) -> Dict[str, Any]:
    """內容向量化工具"""
    global _FAISS_STORAGE
    print(f"[工具] 開始FAISS向量化處理...")
    
    try:
        vectorization_result = {
            "vectorization_status": "success",
            "vectorized_content": {},
            "indexing_complete": True,
            "ready_for_search": True
        }
        
        # 儲存RAG資料到全域變數
        _FAISS_STORAGE["rag_data"] = rag_data
        
        # 1. 向量化text_chunks_with_tracing並建立FAISS索引
        if "text_chunks_with_tracing" in rag_data:
            chunks = rag_data["text_chunks_with_tracing"]
            if chunks:
                print(f"[FAISS] 處理 {len(chunks)} 個文本塊...")
                text_contents = [chunk["content"] for chunk in chunks]
                text_embeddings = get_bailian_embedding(text_contents)
                
                # 確保資料型態為float32（FAISS要求）
                text_embeddings = text_embeddings.astype(np.float32)
                
                # 驗證向量資料
                if np.any(np.isnan(text_embeddings)) or np.any(np.isinf(text_embeddings)):
                    raise ValueError("向量包含NaN或無窮大值")
                
                # 建立FAISS索引 (使用內積計算，適合歸一化向量)
                dimension = text_embeddings.shape[1]
                text_index = faiss.IndexFlatIP(dimension)
                
                # 向量歸一化（用於餘弦相似度）
                faiss.normalize_L2(text_embeddings)
                
                # 加入向量到索引
                text_index.add(text_embeddings)
                
                # 儲存到全域儲存（避免序列化問題）
                _FAISS_STORAGE["text_index"] = text_index
                _FAISS_STORAGE["text_chunks"] = chunks
                vectorization_result["vectorized_content"]["text_chunks_vectorized"] = len(chunks)
                
                print(f"[FAISS] 文本索引建立完成: {len(chunks)}個向量, 維度: {dimension}")
        
        # 2. 向量化insights_with_source並建立FAISS索引
        if "insights_with_source" in rag_data:
            insights = rag_data["insights_with_source"]
            if insights:
                print(f"[FAISS] 處理 {len(insights)} 個洞察...")
                insight_texts = [insight["insight"] for insight in insights]
                insight_embeddings = get_bailian_embedding(insight_texts)
                
                # 確保資料型態為float32（FAISS要求）
                insight_embeddings = insight_embeddings.astype(np.float32)
                
                # 驗證向量資料
                if np.any(np.isnan(insight_embeddings)) or np.any(np.isinf(insight_embeddings)):
                    raise ValueError("洞察向量包含NaN或無窮大值")
                
                # 建立FAISS索引
                dimension = insight_embeddings.shape[1]
                insight_index = faiss.IndexFlatIP(dimension)
                
                # 向量歸一化
                faiss.normalize_L2(insight_embeddings)
                
                # 加入向量到索引
                insight_index.add(insight_embeddings)
                
                # 儲存到全域儲存（避免序列化問題）
                _FAISS_STORAGE["insight_index"] = insight_index
                _FAISS_STORAGE["insights"] = insights
                vectorization_result["vectorized_content"]["insights_vectorized"] = len(insights)
                
                print(f"[FAISS] 洞察索引建立完成: {len(insights)}個向量, 維度: {dimension}")
        
        vectorization_result["vectorized_content"]["embedding_model"] = "bge-m3"
        vectorization_result["vectorized_content"]["vector_dimension"] = dimension if 'dimension' in locals() else 1536
        vectorization_result["vectorized_content"]["index_type"] = "FAISS_IndexFlatIP"
        vectorization_result["vectorized_content"]["storage_method"] = "global_storage"
        vectorization_result["message"] = "FAISS向量化完成，索引已儲存到全域空間"
        
        indices_count = sum(1 for k in ["text_index", "insight_index"] if _FAISS_STORAGE[k] is not None)
        print(f"[FAISS] 向量化完成，建立了 {indices_count} 個索引")
        return vectorization_result
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"[工具] FAISS向量化失敗: {str(e)}")
        print(f"[除錯] 詳細錯誤資訊:\n{error_details}")
        
        return {
            "vectorization_status": "failed",
            "error": str(e),
            "error_details": error_details,
            "ready_for_search": False
        }

def search_with_tracing_tool(query: str) -> Dict[str, Any]:
    """帶溯源的FAISS檢索工具（使用全域儲存）"""
    global _FAISS_STORAGE
    print(f"🔍 [FAISS] 執行溯源檢索: {query}")
    print(f"擷取到的使用者問題：{query}")
    try:
        results = {
            "text_matches": [],
            "table_matches": [], 
            "insight_matches": [],
            "query_analysis": {
                "user_intent": f"使用者查詢關於: {query}",
                "search_strategy": "FAISS向量檢索 + 關鍵詞比對",
                "confidence": 0.9
            },
            "search_engine": "FAISS_GLOBAL"
        }
        
        # 1. FAISS文本塊檢索
        if (_FAISS_STORAGE["text_index"] is not None and 
            _FAISS_STORAGE["text_chunks"] is not None):
            
            text_index = _FAISS_STORAGE["text_index"]
            chunks = _FAISS_STORAGE["text_chunks"]
            
            print(f"🔤 [FAISS] 檢索文本塊，索引大小: {text_index.ntotal}")
            
            # 查詢向量化並歸一化
            query_embedding = get_bailian_embedding([query])
            query_embedding = query_embedding.astype(np.float32)  # FAISS要求float32
            faiss.normalize_L2(query_embedding)
            
            # FAISS檢索
            top_k = min(5, text_index.ntotal)  # 檢索top-k個結果
            similarities, indices = text_index.search(query_embedding, top_k)
            
            # 處理檢索結果
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1 and similarity > 0.3:  # FAISS返回的是餘弦相似度
                    chunk = chunks[idx]
                    results["text_matches"].append({
                        "chunk_id": chunk["chunk_id"],
                        "content": chunk["content"],
                        "similarity": float(similarity),
                        "source_type": chunk["source_type"],
                        "position": chunk["position"],
                        "confidence": chunk["confidence"],
                        "faiss_rank": i + 1
                    })
            
            print(f"✅ [FAISS] 文本檢索完成: {len(results['text_matches'])}個匹配")
        
        # 2. 傳統關鍵詞檢索表格資料
        if (_FAISS_STORAGE["rag_data"] is not None and 
            "table_data" in _FAISS_STORAGE["rag_data"]):
            
            rag_data = _FAISS_STORAGE["rag_data"]
            for table in rag_data["table_data"]:
                if any(keyword in table["description"].lower() or keyword in table["extracted_content"].lower() 
                       for keyword in query.lower().split()):
                    results["table_matches"].append({
                        "table_id": table["table_id"],
                        "description": table["description"],
                        "relevance": 0.8,
                        "match_type": "keyword"
                    })
            
            print(f"📊 [關鍵詞] 表格檢索完成: {len(results['table_matches'])}個匹配")
        
        # 3. FAISS洞察檢索
        if (_FAISS_STORAGE["insight_index"] is not None and 
            _FAISS_STORAGE["insights"] is not None):
            
            insight_index = _FAISS_STORAGE["insight_index"]
            insights = _FAISS_STORAGE["insights"]
            
            print(f"💡 [FAISS] 檢索洞察，索引大小: {insight_index.ntotal}")
            
            # 查詢向量化並歸一化
            query_embedding = get_bailian_embedding([query])
            query_embedding = query_embedding.astype(np.float32)  # FAISS要求float32
            faiss.normalize_L2(query_embedding)
            
            # FAISS檢索
            top_k = min(3, insight_index.ntotal)
            similarities, indices = insight_index.search(query_embedding, top_k)
            
            # 處理檢索結果
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx != -1 and similarity > 0.3:
                    insight = insights[idx]
                    results["insight_matches"].append({
                        "insight": insight["insight"],
                        "source_evidence": insight["source_evidence"],
                        "confidence": insight["confidence"],
                        "similarity": float(similarity),
                        "insight_type": insight.get("insight_type", "general"),
                        "faiss_rank": i + 1
                    })
            
            print(f"✅ [FAISS] 洞察檢索完成: {len(results['insight_matches'])}個匹配")
        
        # 統計結果
        results["total_matches"] = len(results["text_matches"]) + len(results["table_matches"]) + len(results["insight_matches"])
        results["search_quality"] = "high" if results["total_matches"] >= 3 else "medium" if results["total_matches"] >= 1 else "low"
        results["ready_for_answer"] = results["total_matches"] > 0
        
        print(f"🎯 [FAISS] 檢索完成: {results['total_matches']}個比對結果")
        print(f"    📝 文本: {len(results['text_matches'])} | 📊 表格: {len(results['table_matches'])} | 💡 洞察: {len(results['insight_matches'])}")
        
        return results
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ [FAISS] 檢索失敗: {str(e)}")
        print(f"🔍 [除錯] 詳細錯誤資訊:\n{error_details}")
        
        return {
            "error": str(e),
            "error_details": error_details,
            "total_matches": 0,
            "ready_for_answer": False,
            "search_engine": "FAISS_ERROR"
        }

# ================ RAG智能體定義 ================

# 1. 資料加載專家
data_loader_agent = Agent(
    name="data_loader_agent",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY
    ),
    description="RAG資料加載與預處理專家",
    instruction="""
            你是RAG資料加載專家。從使用者訊息中擷取JSON資料字串，使用load_rag_data_tool工具解析資料。

            任務步驟：
            1. 從使用者訊息中識別並擷取JSON資料字串
            2. 呼叫load_rag_data_tool(json_data_string)解析JSON資料
            3. 總結加載結果

            輸出加載狀態報告：
            - 資料加載狀態（成功/失敗）
            - 文件類型與概要  
            - 各類資料統計（文本塊、表格、洞察數量）
            - 是否準備好進行向量化處理

            注意：傳入工具的必須是完整的JSON字串，不是檔案路徑。
            資料加載完成後將儲存在session.state中供後續使用。
            """,
    tools=[load_rag_data_tool],
    output_key="data_loading_result"
)

# 2. 向量化專家
vectorization_agent = Agent(
    name="vectorization_agent", 
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY
    ),
    description="FAISS向量化專家",
    instruction="""
        你是FAISS向量化專家。從session.state['data_loading_result']取得已加載的RAG資料，使用vectorize_content_tool工具建立FAISS索引。

        任務步驟：
        1. 從session.state中取得data_loading_result
        2. 擷取其中的rag_data
        3. 呼叫vectorize_content_tool(rag_data)進行FAISS向量化
        4. 總結FAISS索引建立結果

        FAISS向量化特點：
        - 使用Ollama Embedding API生成向量（支援分批處理，每批最多25個）
        - 建立IndexFlatIP索引（適合餘弦相似度）
        - 向量L2歸一化處理
        - **重要**: 索引儲存在全域空間，避免ADK序列化問題

        呼叫工具後，輸出向量化狀態報告：
        - FAISS索引建立狀態
        - 處理的內容統計（文本塊數量、洞察數量）
        - 向量維度和索引類型資訊
        - 全域儲存狀態
        - 是否準備好進行FAISS檢索

        FAISS索引已安全儲存到全域空間，供後續檢索使用。
""",
    tools=[vectorize_content_tool],
    output_key="vectorization_result"
)

# 3. 檢索專家
retrieval_agent = Agent(
    name="retrieval_agent",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY
    ),
    description="FAISS多模態檢索專家 - 支援精確溯源",
    instruction="""
你是FAISS多模態檢索專家，使用search_with_tracing_tool執行高效的向量檢索。

**重要**: FAISS索引已儲存在全域空間，工具函式會自動訪問。

任務步驟：
1. 從使用者訊息中擷取查詢問題
2. 呼叫search_with_tracing_tool(query)執行FAISS向量檢索
3. 分析檢索結果品質

FAISS檢索特點：
- 文本塊：使用FAISS IndexFlatIP進行餘弦相似度檢索
- 表格資料：使用傳統關鍵詞比對
- 洞察資訊：使用FAISS IndexFlatIP進行語義檢索
- 所有結果都包含faiss_rank排序資訊
- 索引儲存在全域空間，避免序列化問題

呼叫工具後，輸出檢索狀態報告：
- 查詢意圖分析
- FAISS檢索結果統計（文本/表格/洞察匹配數量）
- 檢索品質評估和排序資訊
- 是否準備好生成回答

確保檢索結果完整並包含溯源資訊和FAISS排序。
""",
    tools=[search_with_tracing_tool],
    output_key="retrieval_result"
)

# 4. 回答生成專家
answer_generation_agent = Agent(
    name="answer_generation_agent",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY
    ),
    description="FAISS溯源回答生成專家",
    instruction="""
你是FAISS溯源回答生成專家，基於session.state中的FAISS檢索結果生成帶有精確來源資訊的回答。

從session.state取得：
- 'data_loading_result': 文檔資訊
- 'vectorization_result': FAISS向量化狀態（全域儲存）
- 'retrieval_result': FAISS檢索結果（包含text_matches, table_matches, insight_matches）

FAISS檢索結果特點：
- text_matches: 包含faiss_rank排序，similarity為餘弦相似度
- table_matches: 關鍵詞匹配，包含match_type
- insight_matches: 包含faiss_rank排序和insight_type
- search_engine: "FAISS_GLOBAL" 表示使用全域儲存

任務：
1. 分析FAISS檢索到的各類匹配結果
2. 基於匹配內容和FAISS排序生成準確回答
3. 提供完整的溯源資訊（chunk_id、位置、FAISS排序、相似度等）
4. 評估回答的可信度

輸出格式：
**答案概要**: [一句話回答使用者問題]

**詳細解答**: 
[基於FAISS檢索到的信息進行詳細回答，引用具體內容和排序]

**資訊來源**:
- 來源1: [chunk_id] - [source_type] - [position] - [內容概要] (FAISS相似度: 0.95, 排序: #1)
- 來源2: [table_id] - [表格描述] (關鍵詞匹配)
- 來源3: [洞察內容] (FAISS相似度: 0.85, 排序: #2, 類型: trend)

**相關資料**:
[如果有結構化資料，列出關鍵資訊]

**補充建議**:
[基於內容提供實用建議]

**回答可信度**: [high/medium/low] (基於FAISS排序、相似度和來源數量綜合評估)

注意：確保每個資訊都能精確溯源到原始位置，包含FAISS排序資訊和相似度分數。
""",
    output_key="final_answer"
)


# ================ 基礎工具函式 ================

def get_bailian_embedding(texts: list[str]) -> np.ndarray:
    """調用 LangChain Ollama Embeddings API（被工具函式呼叫）- 支援分批處理"""
    try:
        if not texts:
            raise ValueError("文本列表不能為空")
        
        # 過濾空文本
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("沒有有效的文本內容")
        
        print(f"[向量化] 處理 {len(valid_texts)} 個文本...")

        # 匯入 langchain_community.embeddings
        from langchain_community.embeddings import OllamaEmbeddings
        import numpy as np

        # 每批最多25個文本
        BATCH_SIZE = 25
        all_embeddings = []

        # 建立 Ollama Embeddings 物件
        embeddings = OllamaEmbeddings(model="bge-m3")

        # 分批處理
        for i in range(0, len(valid_texts), BATCH_SIZE):
            batch_texts = valid_texts[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            total_batches = (len(valid_texts) + BATCH_SIZE - 1) // BATCH_SIZE

            print(f"[批次 {batch_num}/{total_batches}] 處理 {len(batch_texts)} 個文本...")

            # 呼叫 Ollama Embeddings 取得向量
            batch_embeddings = embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
            print(f"[批次 {batch_num}] 成功獲取 {len(batch_embeddings)} 個向量")

        # 合併所有批次的向量
        embeddings_array = np.array(all_embeddings, dtype=np.float64)

        # 驗證向量
        if embeddings_array.size == 0:
            raise ValueError("獲取到空的向量")

        print(f"[Ollama向量化] 成功獲取向量，shape: {embeddings_array.shape}")
        print(f"[統計] 總計 {len(all_embeddings)} 個向量，維度: {embeddings_array.shape[1]}")

        return embeddings_array

    except Exception as e:
        print(f"[向量化] 失敗: {str(e)}")
        raise e

# ================ 工具函式 ================
def load_local_image(image_path: str) -> tuple[bytes, str]:
    """加載本地圖片"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"圖片檔案不存在: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        mime_type = 'image/jpeg'
    
    print(f"加載圖片: {image_path}")
    print(f"檔案大小: {len(image_data)} bytes")
    
    return image_data, mime_type

# ================ 智能體定義 ================

# 文檔類型定義
SUPPORTED_DOCUMENT_TYPES = {
    "financial_report": "財務報表",
    "invoice": "發票票據", 
    "contract": "合同文檔",
    "research_paper": "研究報告",
    "business_chart": "商業圖表",
    "receipt": "收據憑證",
    "form": "表單文檔",
    "presentation": "演示文檔",
    "course_material": "課程資料",
    "technical_doc": "技術文檔",
    "other": "其他文檔"
}

# 0. 文檔分類專家
document_classifier = Agent(
    name="document_classifier",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
    ),
    description="文檔類型識別和分類專家",
    instruction=f"""
            你是專業的文檔分類專家。請分析圖像並識別文檔類型。

            支持的文檔類型：
            {json.dumps(SUPPORTED_DOCUMENT_TYPES, ensure_ascii=False, indent=2)}

            任務：
            1. 識別文檔的主要類型
            2. 估算識別的置信度（0-1）
            3. 識別文檔的語言和格式
            4. 檢測文檔的品質和清晰度

            輸出格式（嚴格JSON）：
            {{
                "document_type": "類型代碼（如：course_material）",
                "document_name": "類型中文名稱（如：課程資料）",
                "confidence": 0.95,
                "language": "中文/英文/混合",
                "format": "表格/文本/圖表/混合",
                "quality": "high/medium/low",
                "reasoning": "分類理由和依據"
            }}
            """,
    output_key="document_classification",
)

# 1. 文本提取專家
text_extractor = Agent(
    name="text_extractor",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY
    ),
    description="OCR文本提取專家",
    instruction="""
            你是專業的OCR文本提取專家。請仔細提取圖片中的所有文字並進行結構化標註。

            任務：
            1. 提取所有可見文字，包括表格、標題、正文
            2. 識別文字的類型和重要性
            3. 估算每個文本塊的置信度
            4. 標註文字的大概位置

            輸出格式（嚴格JSON）：
            {{
                "extracted_texts": [
                    {{
                        "text": "具體文字內容",
                        "position": "左上角/中央/右下角/頂部/底部等",
                        "text_type": "title/subtitle/data/label/content/table/header",
                        "confidence": 0.95,
                        "importance": "high/medium/low"
                    }}
                ],
                "total_confidence": 0.88,
                "extraction_summary": "提取概述",
                "text_count": 15,
                "quality_assessment": "提取品質評估"
            }}
            """,
    output_key="text_extraction",
)

# 2. 佈局分析專家
layout_analyzer = Agent(
    name="layout_analyzer", 
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY
    ),
    description="文檔結構和佈局分析專家",
    instruction="""
            你是專業的文檔結構分析專家。請分析文檔的佈局和結構。

            任務：
            1. 識別文檔的整體佈局結構
            2. 檢測表格、圖表、文本塊的位置
            3. 分析資訊的層次關係
            4. 識別關鍵區域和重點內容

            輸出格式（嚴格JSON）：
            {{
                "layout_type": "表格型/報告型/圖表型/混合型/課程型",
                "structure_elements": [
                    {{
                        "element_type": "header/table/chart/text_block/footer/title/section",
                        "position": "位置描述（如：頂部/左上角/中央等）",
                        "size": "large/medium/small",
                        "importance": "high/medium/low",
                        "description": "元素描述"
                    }}
                ],
                "visual_hierarchy": "描述視覺層次和資訊組織方式",
                "key_regions": ["重點區域1", "重點區域2"],
                "layout_complexity": "simple/moderate/complex",
                "analysis_confidence": 0.87
            }}
            """,
    output_key="layout_analysis",
)

# 圖像內容並行提取（OCR + 佈局分析）
image_extractor_agent = ParallelAgent(
    name="image_extractor_workflow", 
    sub_agents=[text_extractor, layout_analyzer],
    description="並行進行OCR文本提取和佈局結構分析",
)

# 3. 內容理解專家  
content_analyzer = Agent(
    name="content_analyzer",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY
    ),
    description="內容理解和資料提取專家",
    instruction="""
            你是專業的內容理解專家。基於前面專家的分析結果，進行深度內容理解和資料提取。

            你可以從 {document_classification} 中獲取文檔分類資訊
            你可以從 {text_extraction} 中獲取提取的文字資訊
            你可以從 {layout_analysis} 中獲取佈局分析資訊

            任務：
            1. 判斷圖像的主要用途和價值
            2. 識別關鍵資訊和重要資料
            3. 提取結構化的鍵值對資訊
            4. 分析資料趨勢或關聯性（如有）
            5. 總結核心要點和發現

            輸出格式（嚴格JSON）：
            {{
                "content_purpose": "文檔的主要用途和目標",
                "key_information": [
                    "關鍵資訊點1",
                    "關鍵資訊點2",
                    "關鍵資訊點3"
                ],
                "key_value_pairs": {{
                    "重要字段1": "值1",
                    "重要字段2": "值2",
                    "日期": "2024-01-15",
                    "數量/金額": "具體數值"
                }},
                "data_insights": [
                    "資料洞察1：趨勢分析",
                    "資料洞察2：關聯發現"
                ],
                "tables_detected": [
                    {{
                        "table_description": "表格描述",
                        "key_data": "表格中的關鍵資料",
                        "importance": "high/medium/low"
                    }}
                ],
                "quality_assessment": {{
                    "information_completeness": "high/medium/low",
                    "data_reliability": "high/medium/low",
                    "analysis_confidence": 0.85
                }},
                "summary": "整體內容總結和核心要點"
            }}
            """,
    output_key="content_analysis",
)

# 完整的多智能體分析工作流
# 流程：文檔分類 → 並行提取(OCR+佈局) → 內容理解 → RAG資料整理

complete_analysis_workflow = SequentialAgent(
    name="complete_analysis_workflow",
    sub_agents=[
        document_classifier,      # 步驟1：文檔分類識別
        image_extractor_agent,    # 步驟2：並行提取(OCR + 佈局分析)
        content_analyzer,         # 步驟3：內容理解和資料提取
    ],
    description="完整的多智能體文檔分析工作流：分類 → 提取 → 理解",
)

# 4. RAG資料整理專家（支援溯源）
rag_data_organizer = Agent(
    name="rag_data_organizer",
    model=LiteLlm(
        model=QWEN_MODEL,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY
    ),
    description="RAG資料整理專家 - 支援精確溯源的多模態資料結構",
    instruction="""
        你是高級RAG資料整理專家，專門負責生成支援精確溯源的多模態RAG資料結構。

        你可以從 session.state 中獲取：
        - 'document_classification': 文檔分類資訊
        - 'text_extraction': OCR文本提取結果 (包含extracted_texts數組)
        - 'layout_analysis': 佈局結構分析結果 (包含structure_elements數組)
        - 'content_analysis': 內容理解和資料提取結果

        **關鍵任務**：
        1. 從text_extraction.extracted_texts中提取**真實文本內容**生成text_chunks
        2. 為每個內容塊添加**溯源資訊**（位置、類型、置信度）
        3. **分類處理**不同類型的內容（文本/表格/圖表）
        4. 構建**可追溯**的資料結構

        輸出格式（嚴格JSON）：
        {{
            "document_summary": "一句話概括圖像內容和價值",
            "document_classification": {{
                "document_type": "從document_classification中提取的類型代碼",
                "document_name": "文檔類型中文名稱", 
                "confidence": 0.95,
                "reasoning": "分類理由"
            }},
            "extracted_text": "所有提取文本的完整拼接內容",
            "key_data": {{
                "從content_analysis.key_value_pairs中提取真實的鍵值對": "真實值",
                "重要資訊": "實際提取的資訊"
            }},
            "text_chunks_with_tracing": [
                {{
                    "chunk_id": "chunk_001",
                    "content": "從extracted_texts中提取的真實文本內容",
                    "source_type": "title/content/table/chart/header/section",
                    "position": "從extracted_texts中獲取的位置信息",
                    "importance": "high/medium/low",
                    "confidence": 0.95,
                    "char_count": 50,
                    "keywords": ["關鍵詞1", "關鍵詞2"]
                }}
            ],
            "table_data": [
                {{
                    "table_id": "table_001", 
                    "description": "表格描述（從tables_detected提取）",
                    "extracted_content": "表格的具體文字內容",
                    "position": "表格在文檔中的位置",
                    "structured_data": {{
                        "列名1": "值1",
                        "列名2": "值2"
                    }},
                    "confidence": 0.9,
                    "importance": "high/medium/low"
                }}
            ],
            "chart_data": [
                {{
                    "chart_id": "chart_001",
                    "description": "圖表描述",
                    "chart_type": "bar/line/pie/flow/other",
                    "extracted_text": "圖表中的文字內容",
                    "position": "圖表位置",
                    "insights": ["圖表洞察1", "圖表洞察2"],
                    "confidence": 0.85
                }}
            ],
            "layout_structure": {{
                "layout_type": "從layout_analysis提取的佈局類型",
                "elements_hierarchy": [
                    {{
                        "element_type": "header/section/table/chart",
                        "position": "位置",
                        "content_summary": "內容概要",
                        "importance": "high/medium/low"
                    }}
                ],
                "reading_order": ["元素1", "元素2", "元素3"]
            }},
            "insights_with_source": [
                {{
                    "insight": "從content_analysis.data_insights提取的真實洞察",
                    "source_evidence": "支持這個洞察的具體文本證據",
                    "confidence": 0.88,
                    "insight_type": "trend/comparison/conclusion/recommendation"
                }}
            ],
            "rag_optimized_chunks": [
                "基於extracted_texts重新組織的語義文本塊1：標題+相關內容",
                "語義文本塊2：特定主題的完整段落",
                "語義文本塊3：表格資料+解釋文字",
                "語義文本塊4：圖表資訊+分析要點"
            ],
            "metadata": {{
                "analysis_quality": "high/medium/low",
                "total_text_blocks": "從extracted_texts統計的實際數量",
                "total_tables": "實際檢測到的表格數量",
                "total_charts": "實際檢測到的圖表數量", 
                "confidence_score": 0.88,
                "processing_timestamp": "當前時間戳",
                "tracing_enabled": true,
                "recommended_use_cases": ["基於實際內容推薦的應用場景"]
            }}
        }}

        **重要**：
        1. text_chunks_with_tracing必須包含從extracted_texts數組中提取的**真實文本內容**
        2. 每個資料塊都要有溯源資訊（位置、類型、置信度）
        3. 分別處理文本、表格、圖表三種不同的資料源
        4. rag_optimized_chunks要重新組織內容，按語義相關性分組
        5. 不要使用模板化內容，要基於實際分析結果生成
        """
)

# 最終整合智能體
upload_and_rag_index = SequentialAgent(
    name="document_rag_analyzer",
    sub_agents=[complete_analysis_workflow, rag_data_organizer, data_loader_agent, vectorization_agent ],
    description="專業文檔分析系統：執行完整分析並生成RAG友好的結構化資料",
)

root_agent = SequentialAgent(
    name = 'qa_agent',
    sub_agents=[upload_and_rag_index, retrieval_agent, answer_generation_agent],
    description="QA智能體：上傳圖像並進行RAG索引，然後進行檢索和回答",
)