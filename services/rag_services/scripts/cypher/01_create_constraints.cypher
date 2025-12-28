// ============================================================
// CatRAG Schema Constraints - Week 1 Task A3
// ============================================================
// Purpose: Create uniqueness constraints for all node types
// Created: November 14, 2025
// Owner: Team A - Infrastructure
// ============================================================

// ============================================================
// 1. MON_HOC (Course) Constraints
// ============================================================

// Unique constraint on course code
CREATE CONSTRAINT mon_hoc_ma_mon IF NOT EXISTS
FOR (m:MON_HOC) REQUIRE m.ma_mon IS UNIQUE;

// Note: NODE KEY constraints require Enterprise Edition
// Using UNIQUE constraint instead

// ============================================================
// 2. CHUONG_TRINH_DAO_TAO (Academic Program) Constraints
// ============================================================

// Unique constraint on program code
CREATE CONSTRAINT chuong_trinh_ma IF NOT EXISTS
FOR (c:CHUONG_TRINH_DAO_TAO) REQUIRE c.ma_chuong_trinh IS UNIQUE;

// ============================================================
// 3. KHOA (Department) Constraints
// ============================================================

// Unique constraint on department code
CREATE CONSTRAINT khoa_ma_khoa IF NOT EXISTS
FOR (k:KHOA) REQUIRE k.ma_khoa IS UNIQUE;

// ============================================================
// 4. GIANG_VIEN (Lecturer) Constraints
// ============================================================

// Unique constraint on lecturer ID
CREATE CONSTRAINT giang_vien_ma_gv IF NOT EXISTS
FOR (g:GIANG_VIEN) REQUIRE g.ma_gv IS UNIQUE;

// ============================================================
// 5. QUY_DINH (Regulation) Constraints
// ============================================================

// Unique constraint on regulation code
CREATE CONSTRAINT quy_dinh_ma IF NOT EXISTS
FOR (q:QUY_DINH) REQUIRE q.ma_quy_dinh IS UNIQUE;

// ============================================================
// 6. HOC_KY (Semester) Constraints
// ============================================================

// Unique constraint on semester code
CREATE CONSTRAINT hoc_ky_ma IF NOT EXISTS
FOR (h:HOC_KY) REQUIRE h.ma_hk IS UNIQUE;

// ============================================================
// 7. CATEGORY (Metadata Category) Constraints
// ============================================================

// Unique constraint on category name
CREATE CONSTRAINT category_name IF NOT EXISTS
FOR (cat:CATEGORY) REQUIRE cat.name IS UNIQUE;

// ============================================================
// 8. TAG (Tag/Label) Constraints
// ============================================================

// Note: Using separate unique constraints instead of composite NODE KEY
// Community Edition limitation

// ============================================================
// Property Existence Constraints
// ============================================================

// Note: Property existence constraints (IS NOT NULL) require Enterprise Edition
// In Community Edition, enforce these at the application layer
// Application should validate:
//   - MON_HOC: ten_mon, so_tin_chi must not be null
//   - CHUONG_TRINH_DAO_TAO: ten_chuong_trinh must not be null
//   - KHOA: ten_khoa must not be null

// ============================================================
// Verification Queries (Run after constraints creation)
// ============================================================

// Show all constraints
// SHOW CONSTRAINTS;

// Count constraints by type
// CALL db.constraints() YIELD name, type 
// RETURN type, count(*) as count 
// ORDER BY count DESC;
