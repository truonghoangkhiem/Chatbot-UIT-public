#!/usr/bin/env python3
"""Test keyword extraction"""
import re
import sys
sys.path.insert(0, '.')

# Copy the function from graph_reasoning_agent
def extract_keywords_from_query(query: str):
    """Extract meaningful keywords from query for CatRAG search."""
    # Remove common Vietnamese stop words
    stop_words = {
        "là", "gì", "như", "thế", "nào", "và", "hoặc", "hay", 
        "của", "cho", "với", "để", "từ", "đến", "trong", "ngoài",
        "các", "những", "một", "hai", "ba", "có", "không", "được",
        "phải", "cần", "nên", "hỏi", "về", "tôi", "bạn", "em", "anh"
    }
    
    # Split and filter
    words = query.lower().split()
    keywords = [w for w in words if len(w) > 2 and w not in stop_words]
    
    # Also extract compound terms (Vietnamese compound words)
    compound_terms = []
    compound_patterns = [
        r"học phần", r"môn học", r"sinh viên", r"tín chỉ", r"điểm trung bình",
        r"đăng ký", r"xét tốt nghiệp", r"tốt nghiệp", r"buộc thôi học",
        r"cảnh báo học tập", r"kết quả", r"chương trình đào tạo",
        r"tiên quyết", r"học trước", r"môn cơ sở", r"môn chuyên ngành",
        # ENHANCED: Thêm tên môn học phổ biến
        r"nhập môn lập trình", r"lập trình hướng đối tượng",
        r"cấu trúc dữ liệu", r"giải thuật", r"cơ sở dữ liệu",
        r"mạng máy tính", r"hệ điều hành", r"công nghệ phần mềm",
        r"trí tuệ nhân tạo", r"machine learning", r"deep learning",
        # ENHANCED: Thêm ngoại ngữ và quy định
        r"tiếng anh", r"tiếng nhật", r"tiếng pháp", r"ngoại ngữ",
        r"quy định", r"quy chế", r"điều kiện", r"yêu cầu",
        r"miễn học", r"xét miễn", r"chứng chỉ", r"toeic",
        r"đào tạo ngoại ngữ", r"giảng dạy tiếng anh"
    ]
    for pattern in compound_patterns:
        if re.search(pattern, query.lower()):
            compound_terms.append(pattern.replace(r" ", " "))
    
    # ENHANCED: If compound terms found, prioritize them over single keywords
    # This prevents searching with fragmented keywords like "tiếng", "anh" separately
    if compound_terms:
        # Return compound terms first, then add remaining single keywords
        remaining_keywords = [k for k in keywords if not any(k in ct for ct in compound_terms)]
        return list(set(compound_terms + remaining_keywords[:5]))[:15]
    
    # ENHANCED: Extract course names without codes
    # Pattern: "Môn [Tên]" or just "[Tên môn học]"
    course_name_pattern = r"môn\s+([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ\s]+)"
    course_matches = re.findall(course_name_pattern, query, re.IGNORECASE)
    for match in course_matches:
        compound_terms.append(match.strip())
    
    return list(set(keywords + compound_terms))[:15]  # Max 15 keywords

# Test
queries = [
    "quy định về tiếng anh của trường thế nào",
    "điều kiện miễn môn tiếng anh",
    "yêu cầu chứng chỉ toeic để tốt nghiệp",
    "môn nhập môn lập trình có tiên quyết gì",
]

print("=" * 60)
print("Testing keyword extraction")
print("=" * 60)

for q in queries:
    keywords = extract_keywords_from_query(q)
    print(f"\nQuery: {q}")
    print(f"Keywords: {keywords}")
