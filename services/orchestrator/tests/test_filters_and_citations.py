#!/usr/bin/env python3
"""
Test script for filter extraction and citation functionality.

Tests:
1. ExtractedFilters from SmartPlannerAgent
2. RAGFilters conversion
3. DetailedSource creation
4. Full pipeline with filters
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.adapters.rag_adapter import RAGFilters, RAGServiceAdapter
from app.agents.smart_planner_agent import ExtractedFilters, SmartPlannerAgent
from app.agents.base import DetailedSource, AnswerResult


def test_extracted_filters():
    """Test ExtractedFilters dataclass."""
    print("=" * 60)
    print("TEST 1: ExtractedFilters")
    print("=" * 60)
    
    # Test empty filters
    empty = ExtractedFilters()
    assert empty.is_empty(), "Empty filters should return True for is_empty()"
    assert empty.to_dict() == {}, "Empty filters should return empty dict"
    print("✅ Empty filters work correctly")
    
    # Test with values
    filters = ExtractedFilters(
        doc_types=["regulation", "syllabus"],
        faculties=["CNTT", "KHMT"],
        years=[2023, 2024],
        subjects=["SE101", "IT002"]
    )
    
    assert not filters.is_empty(), "Non-empty filters should return False for is_empty()"
    result = filters.to_dict()
    assert "doc_types" in result
    assert "faculties" in result
    assert "years" in result
    assert "subjects" in result
    print("✅ Non-empty filters work correctly")
    print(f"   Result: {result}")
    
    return True


def test_rag_filters():
    """Test RAGFilters class."""
    print("\n" + "=" * 60)
    print("TEST 2: RAGFilters")
    print("=" * 60)
    
    # Test empty filters
    empty = RAGFilters()
    assert empty.is_empty(), "Empty RAGFilters should return True for is_empty()"
    print("✅ Empty RAGFilters work correctly")
    
    # Test with values
    filters = RAGFilters(
        doc_types=["regulation"],
        faculties=["CNTT"],
        years=[2024],
        language="vi"
    )
    
    assert not filters.is_empty(), "Non-empty RAGFilters should return False for is_empty()"
    result = filters.to_dict()
    assert "doc_types" in result
    assert "faculties" in result
    assert "years" in result
    assert result["language"] == "vi"
    print("✅ Non-empty RAGFilters work correctly")
    print(f"   Result: {result}")
    
    return True


def test_filter_conversion():
    """Test conversion from ExtractedFilters to RAGFilters."""
    print("\n" + "=" * 60)
    print("TEST 3: Filter Conversion")
    print("=" * 60)
    
    extracted = ExtractedFilters(
        doc_types=["regulation"],
        faculties=["CNTT", "KHMT"],
        years=[2024]
    )
    
    # Convert to RAGFilters
    rag_filters = RAGFilters(
        doc_types=extracted.doc_types if extracted.doc_types else None,
        faculties=extracted.faculties if extracted.faculties else None,
        years=extracted.years if extracted.years else None,
        subjects=extracted.subjects if extracted.subjects else None
    )
    
    assert not rag_filters.is_empty()
    result = rag_filters.to_dict()
    assert result["doc_types"] == ["regulation"]
    assert result["faculties"] == ["CNTT", "KHMT"]
    assert result["years"] == [2024]
    print("✅ Filter conversion works correctly")
    print(f"   Result: {result}")
    
    return True


def test_detailed_source():
    """Test DetailedSource dataclass."""
    print("\n" + "=" * 60)
    print("TEST 4: DetailedSource")
    print("=" * 60)
    
    source = DetailedSource(
        title="Quy chế đào tạo đại học",
        doc_id="doc_001",
        chunk_id="chunk_003",
        score=0.92,
        citation_text="Sinh viên phải hoàn thành tối thiểu 120 tín chỉ",
        char_spans=[
            {"start": 0, "end": 47, "text": "Sinh viên phải hoàn thành tối thiểu 120 tín chỉ", "type": "sentence"}
        ],
        highlighted_text=["<em>Sinh viên</em> phải hoàn thành tối thiểu <em>120 tín chỉ</em>"],
        doc_type="regulation",
        faculty="CNTT",
        year=2024,
        subject=None
    )
    
    assert source.title == "Quy chế đào tạo đại học"
    assert source.score == 0.92
    assert source.citation_text is not None
    assert len(source.char_spans) == 1
    assert source.doc_type == "regulation"
    print("✅ DetailedSource works correctly")
    print(f"   Title: {source.title}")
    print(f"   Citation: {source.citation_text}")
    print(f"   Char spans: {len(source.char_spans)} span(s)")
    
    return True


def test_answer_result_with_detailed_sources():
    """Test AnswerResult with detailed_sources."""
    print("\n" + "=" * 60)
    print("TEST 5: AnswerResult with DetailedSources")
    print("=" * 60)
    
    detailed_sources = [
        DetailedSource(
            title="Quy chế đào tạo",
            doc_id="doc_001",
            score=0.9,
            citation_text="Điều 15: Quy định về đăng ký học phần",
            doc_type="regulation"
        ),
        DetailedSource(
            title="Hướng dẫn sinh viên",
            doc_id="doc_002",
            score=0.85,
            citation_text="Sinh viên đăng ký trực tuyến qua portal",
            doc_type="handbook"
        )
    ]
    
    result = AnswerResult(
        query="Làm sao để đăng ký học phần?",
        answer="Để đăng ký học phần, sinh viên truy cập portal...",
        confidence=0.88,
        sources_used=["Quy chế đào tạo", "Hướng dẫn sinh viên"],
        reasoning_steps=["Tìm thông tin về đăng ký", "Tổng hợp từ các nguồn"],
        metadata={"method": "rag"},
        detailed_sources=detailed_sources
    )
    
    assert len(result.sources_used) == 2
    assert len(result.detailed_sources) == 2
    assert result.detailed_sources[0].citation_text == "Điều 15: Quy định về đăng ký học phần"
    print("✅ AnswerResult with detailed_sources works correctly")
    print(f"   Sources: {len(result.sources_used)}")
    print(f"   Detailed sources: {len(result.detailed_sources)}")
    
    return True


def test_filter_extraction_patterns():
    """Test filter extraction from various query patterns."""
    print("\n" + "=" * 60)
    print("TEST 6: Filter Extraction Patterns")
    print("=" * 60)
    
    import re
    
    def extract_filters(query: str) -> ExtractedFilters:
        """Simple filter extraction for testing."""
        query_lower = query.lower()
        filters = ExtractedFilters()
        
        # DOC_TYPES
        doc_type_patterns = {
            "regulation": ["quy chế", "quy định", "điều lệ", "nội quy"],
            "syllabus": ["đề cương", "chương trình đào tạo", "ctđt"],
            "announcement": ["thông báo", "công văn", "hướng dẫn"],
        }
        
        for doc_type, patterns in doc_type_patterns.items():
            if any(p in query_lower for p in patterns):
                filters.doc_types.append(doc_type)
        
        # FACULTIES
        faculty_patterns = {
            "CNTT": ["công nghệ thông tin", "cntt"],
            "KHMT": ["khoa học máy tính", "khmt"],
            "HTTT": ["hệ thống thông tin", "httt"],
        }
        
        for faculty, patterns in faculty_patterns.items():
            if any(p in query_lower for p in patterns):
                filters.faculties.append(faculty)
        
        # YEARS
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, query_lower)
        for y in years:
            year = int(y)
            if year not in filters.years:
                filters.years.append(year)
        
        # SUBJECTS
        subject_pattern = r'\b([A-Z]{2,4}\d{3})\b'
        subjects = re.findall(subject_pattern, query.upper())
        filters.subjects = list(set(subjects))
        
        return filters
    
    test_cases = [
        ("Quy chế đào tạo ngành CNTT năm 2024", 
         {"doc_types": ["regulation"], "faculties": ["CNTT"], "years": [2024]}),
        ("Đề cương môn SE101 khoa học máy tính",
         {"doc_types": ["syllabus"], "faculties": ["KHMT"], "subjects": ["SE101"]}),
        ("Thông báo học phí 2023",
         {"doc_types": ["announcement"], "years": [2023]}),
    ]
    
    all_passed = True
    for query, expected in test_cases:
        result = extract_filters(query)
        result_dict = result.to_dict()
        
        # Check each expected field
        for key, value in expected.items():
            if key not in result_dict or sorted(result_dict[key]) != sorted(value):
                print(f"❌ Failed for query: {query}")
                print(f"   Expected {key}: {value}, Got: {result_dict.get(key)}")
                all_passed = False
                break
        else:
            print(f"✅ Query: {query[:40]}...")
            print(f"   Filters: {result_dict}")
    
    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("FILTER AND CITATION TESTS")
    print("=" * 60)
    
    tests = [
        ("ExtractedFilters", test_extracted_filters),
        ("RAGFilters", test_rag_filters),
        ("Filter Conversion", test_filter_conversion),
        ("DetailedSource", test_detailed_source),
        ("AnswerResult with DetailedSources", test_answer_result_with_detailed_sources),
        ("Filter Extraction Patterns", test_filter_extraction_patterns),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✅ PASS" if p else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
