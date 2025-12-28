# Parallel RAG Retrieval Optimization

## Vấn đề (Problem)

Thời gian truy xuất RAG (Retrieval-Augmented Generation) quá lâu - khoảng **19-20 giây** cho mỗi query.

## Nguyên nhân (Root Cause)

Trong file `services/orchestrator/app/agents/optimized_orchestrator.py`, method `_perform_rag_retrieval()` đang thực thi các queries **tuần tự (sequential)** thay vì song song (parallel):

```python
# ❌ BEFORE - Sequential execution
for query in queries:
    try:
        result = await self.rag_port.retrieve_context(query, ...)
        all_results.extend(result["retrieved_documents"])
    except Exception:
        continue
```

Khi SmartPlanner tạo ra nhiều rewritten queries (ví dụ 2-3 queries), mỗi query phải chờ query trước đó hoàn thành, dẫn đến thời gian tăng gấp bội.

## Giải pháp (Solution)

Sử dụng `asyncio.gather()` để thực thi tất cả queries **song song**:

```python
# ✅ AFTER - Parallel execution
async def retrieve_single_query(query: str) -> List[Dict[str, Any]]:
    """Helper function to retrieve documents for a single query."""
    try:
        result = await self.rag_port.retrieve_context(
            query, 
            top_k=top_k,
            filters=rag_filters,
            use_rerank=use_rerank,
            need_citation=True,
            include_char_spans=True
        )
        if result and result.get("retrieved_documents"):
            return result["retrieved_documents"]
        return []
    except Exception as e:
        logger.warning(f"Query failed: {query[:50]}... - {e}")
        return []

# Execute all queries in parallel
results_per_query = await asyncio.gather(
    *[retrieve_single_query(q) for q in queries],
    return_exceptions=False
)

# Flatten results from all queries
for docs in results_per_query:
    all_results.extend(docs)
```

## Kết quả (Results)

### Benchmark với query: "Điều 19 quy định gì?"

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **RAG Time** | 19.87s | 9.94s | **50.0% faster** ⚡ |
| **Total Time** | 31.77s | 20.40s | **35.8% faster** ⚡ |

### Chi tiết timing (3 lần test, lấy trung bình):
- **RAG Retrieval**: 9.94s (±0.03s) - rất ổn định
- **Planning**: 3.73s
- **Answer Generation**: 6.74s
- **Total Backend**: 20.40s

## Lợi ích (Benefits)

1. ✅ **Giảm 50% thời gian RAG retrieval** - từ ~20s xuống ~10s
2. ✅ **Giảm 36% thời gian tổng** - từ ~32s xuống ~20s
3. ✅ **Hiệu suất ổn định** - độ lệch chuẩn chỉ 0.03s
4. ✅ **Không thay đổi chất lượng** - vẫn truy xuất được các documents giống nhau
5. ✅ **Scale tốt hơn** - khi có nhiều rewritten queries, thời gian không tăng tuyến tính

## Cách kiểm tra (How to Test)

### 1. Chạy benchmark:
```bash
cd c:\Users\admin\Downloads\Khiem\Chatbot-UIT
conda activate chatbot-uit
python benchmark_parallel_rag.py
```

### 2. Test với nhiều queries khác nhau:
```bash
python test_multiple_queries.py
```

### 3. Test timing chi tiết:
```bash
python test_detailed_timing.py
```

## Files Changed

- `services/orchestrator/app/agents/optimized_orchestrator.py`
  - Method: `_perform_rag_retrieval()` (lines ~791-803)
  - Changed from sequential for loop to parallel asyncio.gather()

## Opportunities for Further Optimization

1. **SmartPlanner reliability** - Sometimes fails and returns N/A for all fields
2. **Graph Reasoning integration** - Context appears even when not requested by planner
3. **Planning parallelization** - Could potentially run Planning + initial RAG in parallel

## Date

December 9, 2025
