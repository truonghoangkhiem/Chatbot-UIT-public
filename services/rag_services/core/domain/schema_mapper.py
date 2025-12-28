"""
Schema Mapper for CatRAG Knowledge Graph.

Handles conversion between different schema formats:
1. LLM extraction format (code="MON_HOC_IT003", name="IT003")
2. Standard format (ma_mon="IT003", ten_mon="Lập trình hướng đối tượng")
3. Neo4j label format (MON_HOC, not MonHoc)

This ensures unified schema across the entire ETL pipeline.
"""

import re
from typing import Dict, Any, Optional
from enum import Enum


class StandardLabel(str, Enum):
    """Standardized Neo4j Labels - All UPPER_SNAKE_CASE"""
    MON_HOC = "MON_HOC"
    QUY_DINH = "QUY_DINH"
    DIEU_KIEN = "DIEU_KIEN"
    KHOA = "KHOA"
    NGANH = "NGANH"
    CHUONG_TRINH_DAO_TAO = "CHUONG_TRINH_DAO_TAO"
    SINH_VIEN = "SINH_VIEN"
    KY_HOC = "KY_HOC"
    GIANG_VIEN = "GIANG_VIEN"
    HOC_PHI = "HOC_PHI"


class SchemaMapper:
    """
    Maps between different schema formats.
    
    Ensures:
    - Labels are always UPPER_SNAKE_CASE (MON_HOC, not MonHoc)
    - Property keys are standardized (ma_mon, not code)
    - No prefix in IDs (IT003, not MON_HOC_IT003)
    """
    
    # Mapping from old PascalCase to new UPPER_SNAKE_CASE
    LABEL_MAPPING = {
        "MonHoc": "MON_HOC",
        "QuyDinh": "QUY_DINH",
        "DieuKien": "DIEU_KIEN",
        "Khoa": "KHOA",
        "Nganh": "NGANH",
        "ChuongTrinhDaoTao": "CHUONG_TRINH_DAO_TAO",
        "SinhVien": "SINH_VIEN",
        "KyHoc": "KY_HOC",
        "GiangVien": "GIANG_VIEN",
        "HocPhi": "HOC_PHI",
    }
    
    # Standard property names for each label
    PROPERTY_MAPPING = {
        "MON_HOC": {
            "id_key": "ma_mon",  # Primary key
            "name_key": "ten_mon",  # Display name
            "required": ["ma_mon", "ten_mon", "so_tin_chi"],
        },
        "KHOA": {
            "id_key": "ma_khoa",
            "name_key": "ten_khoa",
            "required": ["ma_khoa", "ten_khoa"],
        },
        "NGANH": {
            "id_key": "ma_nganh",
            "name_key": "ten_nganh",
            "required": ["ma_nganh", "ten_nganh"],
        },
        "QUY_DINH": {
            "id_key": "ma_quy_dinh",
            "name_key": "tieu_de",
            "required": ["ma_quy_dinh", "tieu_de"],
        },
        "DIEU_KIEN": {
            "id_key": "ma_dieu_kien",
            "name_key": "mo_ta",
            "required": ["ma_dieu_kien", "mo_ta"],
        },
    }
    
    @classmethod
    def normalize_label(cls, label: str) -> str:
        """
        Convert any label format to standard UPPER_SNAKE_CASE.
        
        Examples:
            MonHoc → MON_HOC
            mon_hoc → MON_HOC
            MON_HOC → MON_HOC
        """
        # If already in mapping
        if label in cls.LABEL_MAPPING:
            return cls.LABEL_MAPPING[label]
        
        # If already uppercase
        if label.isupper():
            return label
        
        # Try reverse mapping
        for old, new in cls.LABEL_MAPPING.items():
            if label.lower() == new.lower():
                return new
        
        # Default: convert to UPPER_SNAKE_CASE
        return label.upper()
    
    @classmethod
    def extract_clean_id(cls, value: str, label: str) -> str:
        """
        Extract clean ID without prefix.
        
        Examples:
            MON_HOC_IT003 → IT003
            IT003 → IT003
            KHOA_CNTT → CNTT
        """
        # Remove common prefixes
        prefixes = [
            "MON_HOC_",
            "KHOA_",
            "NGANH_",
            "QUY_DINH_",
            "DIEU_KIEN_",
        ]
        
        clean = value
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        
        # Remove extra underscores and parentheses
        clean = clean.replace("(", "").replace(")", "").strip("_")
        
        return clean
    
    @classmethod
    def map_llm_entity_to_standard(cls, entity_data: Dict[str, Any], label: str) -> Dict[str, Any]:
        """
        Convert LLM extraction format to standard format.
        
        Args:
            entity_data: {
                "text": "IT003",
                "type": "MON_HOC", 
                "confidence": 0.95,
                ...
            }
            label: Node label (will be normalized)
            
        Returns:
            {
                "label": "MON_HOC",
                "properties": {
                    "ma_mon": "IT003",
                    "ten_mon": "IT003",  # or from metadata
                    ...
                }
            }
        """
        normalized_label = cls.normalize_label(label)
        
        # Get property mapping for this label
        prop_map = cls.PROPERTY_MAPPING.get(normalized_label, {})
        id_key = prop_map.get("id_key", "id")
        name_key = prop_map.get("name_key", "name")
        
        # Extract clean ID
        text = entity_data.get("text", "")
        clean_id = cls.extract_clean_id(text, normalized_label)
        
        # Build standard properties
        properties = {
            id_key: clean_id,
            name_key: entity_data.get("name", clean_id),
        }
        
        # Add other properties from metadata
        if "metadata" in entity_data:
            metadata = entity_data["metadata"]
            
            # Map common metadata fields
            if normalized_label == "MON_HOC":
                if "credits" in metadata:
                    properties["so_tin_chi"] = metadata["credits"]
                if "title" in metadata:
                    properties["ten_mon"] = metadata["title"]
                if "description" in metadata:
                    properties["mo_ta"] = metadata["description"]
        
        # Ensure required fields
        if normalized_label == "MON_HOC":
            properties.setdefault("so_tin_chi", 4)  # Default credits
            properties.setdefault("ten_mon", clean_id)
        
        return {
            "label": normalized_label,
            "properties": properties
        }
    
    @classmethod
    def map_graph_node_to_standard(cls, category: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert GraphNode format to standard format.
        
        Handles both old format (code, name) and new format (ma_mon, ten_mon).
        """
        normalized_label = cls.normalize_label(category)
        prop_map = cls.PROPERTY_MAPPING.get(normalized_label, {})
        id_key = prop_map.get("id_key", "id")
        name_key = prop_map.get("name_key", "name")
        
        # Extract ID from various possible keys
        id_value = (
            properties.get(id_key) or
            properties.get("code") or
            properties.get("id") or
            properties.get("name")
        )
        
        if id_value:
            id_value = cls.extract_clean_id(str(id_value), normalized_label)
        
        # Extract name
        name_value = (
            properties.get(name_key) or
            properties.get("name") or
            properties.get("title") or
            id_value
        )
        
        # Build standard properties
        standard_props = {
            id_key: id_value,
            name_key: name_value,
        }
        
        # Copy other relevant properties
        for key, value in properties.items():
            if key not in ["code", "id", "name", "title"] and key not in standard_props:
                standard_props[key] = value
        
        return {
            "label": normalized_label,
            "properties": standard_props
        }
    
    @classmethod
    def validate_node_properties(cls, label: str, properties: Dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate that node has all required properties.
        
        Returns:
            (is_valid, missing_properties)
        """
        normalized_label = cls.normalize_label(label)
        prop_map = cls.PROPERTY_MAPPING.get(normalized_label, {})
        required = prop_map.get("required", [])
        
        missing = []
        for prop in required:
            if prop not in properties or not properties[prop]:
                missing.append(prop)
        
        return (len(missing) == 0, missing)


# Convenience functions for common operations

def normalize_mon_hoc_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick helper to normalize MON_HOC properties.
    
    Ensures:
    - ma_mon exists and is clean (no prefix)
    - ten_mon exists
    - so_tin_chi exists (default 4)
    """
    mapper = SchemaMapper()
    result = mapper.map_graph_node_to_standard("MON_HOC", props)
    return result["properties"]


def normalize_khoa_properties(props: Dict[str, Any]) -> Dict[str, Any]:
    """Quick helper to normalize KHOA properties."""
    mapper = SchemaMapper()
    result = mapper.map_graph_node_to_standard("KHOA", props)
    return result["properties"]


def get_standard_id_key(label: str) -> str:
    """Get the standard ID key for a label (e.g., ma_mon for MON_HOC)."""
    normalized = SchemaMapper.normalize_label(label)
    return SchemaMapper.PROPERTY_MAPPING.get(normalized, {}).get("id_key", "id")
