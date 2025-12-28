// ============================================================
// CatRAG Schema Indexes - Week 1 Task A3
// ============================================================
// Purpose: Create indexes for query performance optimization
// Created: November 14, 2025
// Owner: Team A - Infrastructure
// ============================================================

// ============================================================
// 1. Full-Text Search Indexes
// ============================================================

// Full-text index on course names (Vietnamese + English)
CREATE FULLTEXT INDEX mon_hoc_fulltext IF NOT EXISTS
FOR (m:MON_HOC)
ON EACH [m.ten_mon, m.ten_mon_en, m.mo_ta, m.noi_dung];

// Full-text index on department names
CREATE FULLTEXT INDEX khoa_fulltext IF NOT EXISTS
FOR (k:KHOA)
ON EACH [k.ten_khoa, k.ten_khoa_en];

// Full-text index on program names
CREATE FULLTEXT INDEX chuong_trinh_fulltext IF NOT EXISTS
FOR (c:CHUONG_TRINH_DAO_TAO)
ON EACH [c.ten_chuong_trinh, c.mo_ta];

// Full-text index on regulations
CREATE FULLTEXT INDEX quy_dinh_fulltext IF NOT EXISTS
FOR (q:QUY_DINH)
ON EACH [q.tieu_de, q.noi_dung];

// Full-text index on lecturer names
CREATE FULLTEXT INDEX giang_vien_fulltext IF NOT EXISTS
FOR (g:GIANG_VIEN)
ON EACH [g.ten_gv, g.chuyen_nganh];

// ============================================================
// 2. Range Indexes (for filtering and sorting)
// ============================================================

// Index on course credits for filtering
CREATE INDEX mon_hoc_tin_chi_idx IF NOT EXISTS
FOR (m:MON_HOC) ON (m.so_tin_chi);

// Index on course semester for filtering
CREATE INDEX mon_hoc_hoc_ky_idx IF NOT EXISTS
FOR (m:MON_HOC) ON (m.hoc_ky);

// Index on program year for filtering
CREATE INDEX chuong_trinh_nam_idx IF NOT EXISTS
FOR (c:CHUONG_TRINH_DAO_TAO) ON (c.nam_bat_dau);

// Index on regulation status for filtering active regulations
CREATE INDEX quy_dinh_trang_thai_idx IF NOT EXISTS
FOR (q:QUY_DINH) ON (q.trang_thai);

// Index on regulation year for filtering
CREATE INDEX quy_dinh_nam_idx IF NOT EXISTS
FOR (q:QUY_DINH) ON (q.nam_ap_dung);

// Index on semester dates for filtering
CREATE INDEX hoc_ky_ngay_bat_dau_idx IF NOT EXISTS
FOR (h:HOC_KY) ON (h.ngay_bat_dau);

CREATE INDEX hoc_ky_ngay_ket_thuc_idx IF NOT EXISTS
FOR (h:HOC_KY) ON (h.ngay_ket_thuc);

// ============================================================
// 3. Composite Indexes (for complex queries)
// ============================================================

// Composite index for courses by department and semester
CREATE INDEX mon_hoc_khoa_hoc_ky_idx IF NOT EXISTS
FOR (m:MON_HOC) ON (m.khoa, m.hoc_ky);

// Composite index for regulations by type and status
CREATE INDEX quy_dinh_loai_trang_thai_idx IF NOT EXISTS
FOR (q:QUY_DINH) ON (q.loai, q.trang_thai);

// ============================================================
// 4. Property Indexes (for existence checks)
// ============================================================

// Index on entity mention confidence for filtering high-quality extractions
CREATE INDEX entity_mention_confidence_idx IF NOT EXISTS
FOR (e:ENTITY_MENTION) ON (e.confidence);

// Index on entity mention source documents
CREATE INDEX entity_mention_source_idx IF NOT EXISTS
FOR (e:ENTITY_MENTION) ON (e.source_doc);

// Index on tag types for filtering
CREATE INDEX tag_type_idx IF NOT EXISTS
FOR (t:TAG) ON (t.type);

// Index on category parent for hierarchical queries
CREATE INDEX category_parent_idx IF NOT EXISTS
FOR (c:CATEGORY) ON (c.parent_category);

// ============================================================
// 5. Timestamp Indexes (for audit queries)
// ============================================================

// Index on course creation/update timestamps
CREATE INDEX mon_hoc_created_at_idx IF NOT EXISTS
FOR (m:MON_HOC) ON (m.created_at);

CREATE INDEX mon_hoc_updated_at_idx IF NOT EXISTS
FOR (m:MON_HOC) ON (m.updated_at);

// ============================================================
// 6. CatRAG-Specific Indexes
// ============================================================

// Index on relationship similarity scores (for LIEN_QUAN relationships)
// This will be created on relationship properties
// Note: Neo4j 5.15 supports relationship property indexes

// Index on prerequisite minimum grades
CREATE INDEX dieu_kien_loai_idx IF NOT EXISTS
FOR (d:DIEU_KIEN) ON (d.loai);

// ============================================================
// Verification Queries (Run after index creation)
// ============================================================

// Show all indexes
// SHOW INDEXES;

// Count indexes by type
// CALL db.indexes() YIELD name, type, labelsOrTypes, properties
// RETURN type, count(*) as count 
// ORDER BY count DESC;

// Test full-text search
// CALL db.index.fulltext.queryNodes('mon_hoc_fulltext', 'cấu trúc dữ liệu') 
// YIELD node, score 
// RETURN node.ma_mon, node.ten_mon, score 
// ORDER BY score DESC 
// LIMIT 10;

// ============================================================
// Performance Notes
// ============================================================
// 
// 1. Full-text indexes: Use for search queries with CALL db.index.fulltext.queryNodes()
// 2. Range indexes: Automatically used for WHERE clauses with =, <, >, BETWEEN
// 3. Composite indexes: Used when filtering on multiple properties together
// 4. Re-index after bulk data loads for optimal performance
//
// Example usage:
//   // Full-text search
//   CALL db.index.fulltext.queryNodes('mon_hoc_fulltext', 'giải thuật')
//   YIELD node, score
//   RETURN node.ten_mon, score
//   ORDER BY score DESC;
//
//   // Range query (auto-uses index)
//   MATCH (m:MON_HOC)
//   WHERE m.so_tin_chi >= 3 AND m.so_tin_chi <= 4
//   RETURN m.ma_mon, m.ten_mon;
//
// ============================================================
