# Scripts Organization

This directory contains utility scripts organized by purpose for better maintainability.

## 📁 Directory Structure

```
scripts/
├── setup/          # Setup and initialization scripts
├── etl/            # ETL and data processing scripts
├── demo/           # Demo and testing scripts
├── tools/          # Utility and maintenance tools
└── cypher/         # Cypher query templates
```

---

## 📂 setup/ - Setup & Initialization

**Purpose:** Scripts for initial setup, database initialization, and system configuration.

### Files to Move Here:
- `dev_run.sh` - Development server startup
- `quick_start.sh` - Quick start script
- `start_hybrid_system.sh` - Start hybrid RAG system
- `start_weaviate_system.sh` - Start Weaviate system
- `test_neo4j_connection.py` - Test Neo4j connection
- `create_sample_data.py` - Create sample data for testing

### Usage:
```bash
# Setup environment
./scripts/setup/quick_start.sh

# Test database connections
python scripts/setup/test_neo4j_connection.py

# Initialize with sample data
python scripts/setup/create_sample_data.py
```

---

## 📂 etl/ - ETL & Data Processing

**Purpose:** Scripts for data extraction, transformation, and loading into various systems.

### Files to Move Here:
- `run_etl.py` - **NEW** Main ETL runner for graph building
- `index_crawled_data.py` - Index crawled program data
- `index_quy_dinh.py` - Index regulation PDFs (v1)
- `index_quy_dinh_v2.py` - Index regulation PDFs (v2 - improved)
- `build_graph_from_indexed_data.py` - Build graph from indexed data

### Usage:
```bash
# Index regulations (improved version)
python scripts/etl/index_quy_dinh_v2.py

# Index crawled data
python scripts/etl/index_crawled_data.py

# Run full ETL pipeline
python scripts/etl/run_etl.py --source data/quy_dinh
```

---

## 📂 demo/ - Demos & Testing

**Purpose:** Demo scripts and interactive testing tools.

### Files to Move Here:
- `demo_week2.py` - **NEW** Week 2 implementations demo
- `demo_graphrag_poc.py` - GraphRAG POC demo
- `test_rag_quick.py` - Quick RAG system test

### Usage:
```bash
# Demo Week 2 features
python scripts/demo/demo_week2.py --test all

# Demo GraphRAG
python scripts/demo/demo_graphrag_poc.py

# Quick RAG test
python scripts/demo/test_rag_quick.py
```

---

## 📂 tools/ - Utilities & Maintenance

**Purpose:** Utility scripts for monitoring, viewing data, benchmarking, and maintenance.

### Files to Move Here:
- `view_graph_nodes.py` - View graph nodes in Neo4j
- `view_indexed_data.py` - View indexed data in vector stores
- `check_weaviate_data.py` - Check Weaviate data
- `quick_view.py` - Quick data viewer
- `reset_opensearch.py` - Reset OpenSearch index
- `benchmark_graph_operations.py` - Benchmark graph operations
- `performance_test.py` - Performance testing

### Usage:
```bash
# View graph data
python scripts/tools/view_graph_nodes.py

# Check vector store
python scripts/tools/check_weaviate_data.py

# Benchmark performance
python scripts/tools/benchmark_graph_operations.py

# Reset OpenSearch (CAUTION!)
python scripts/tools/reset_opensearch.py
```

---

## 📂 cypher/ - Cypher Queries

**Purpose:** Reusable Cypher query templates and examples.

### Contents:
- Pre-written Cypher queries for common operations
- Query templates for graph traversal
- Example queries for learning

---

## 🔄 Migration Plan

### Phase 1: Create Symlinks (Backward Compatible)
```bash
# Keep old scripts working while moving files
cd scripts/

# Setup
ln -s setup/quick_start.sh quick_start.sh
ln -s setup/dev_run.sh dev_run.sh

# ETL
ln -s etl/run_etl.py run_etl.py
ln -s etl/index_quy_dinh_v2.py index_quy_dinh_v2.py

# Demo
ln -s demo/demo_week2.py demo_week2.py

# Tools
ln -s tools/view_graph_nodes.py view_graph_nodes.py
```

### Phase 2: Update Documentation
- Update all README files with new paths
- Update QUICK_START_GUIDE.md
- Update weekly plan documentation

### Phase 3: Move Files
```bash
# Move files to new locations
mv demo_week2.py demo/
mv run_etl.py etl/
mv index_quy_dinh_v2.py etl/
# ... etc
```

### Phase 4: Remove Symlinks
```bash
# After confirming everything works
rm quick_start.sh dev_run.sh run_etl.py demo_week2.py view_graph_nodes.py
```

---

## 📋 File Mapping Reference

| Current Location | New Location | Category |
|-----------------|--------------|----------|
| `quick_start.sh` | `setup/quick_start.sh` | Setup |
| `dev_run.sh` | `setup/dev_run.sh` | Setup |
| `start_hybrid_system.sh` | `setup/start_hybrid_system.sh` | Setup |
| `start_weaviate_system.sh` | `setup/start_weaviate_system.sh` | Setup |
| `test_neo4j_connection.py` | `setup/test_neo4j_connection.py` | Setup |
| `create_sample_data.py` | `setup/create_sample_data.py` | Setup |
| `run_etl.py` | `etl/run_etl.py` | ETL |
| `index_crawled_data.py` | `etl/index_crawled_data.py` | ETL |
| `index_quy_dinh.py` | `etl/index_quy_dinh.py` | ETL |
| `index_quy_dinh_v2.py` | `etl/index_quy_dinh_v2.py` | ETL |
| `build_graph_from_indexed_data.py` | `etl/build_graph_from_indexed_data.py` | ETL |
| `demo_week2.py` | `demo/demo_week2.py` | Demo |
| `demo_graphrag_poc.py` | `demo/demo_graphrag_poc.py` | Demo |
| `test_rag_quick.py` | `demo/test_rag_quick.py` | Demo |
| `view_graph_nodes.py` | `tools/view_graph_nodes.py` | Tools |
| `view_indexed_data.py` | `tools/view_indexed_data.py` | Tools |
| `check_weaviate_data.py` | `tools/check_weaviate_data.py` | Tools |
| `quick_view.py` | `tools/quick_view.py` | Tools |
| `reset_opensearch.py` | `tools/reset_opensearch.py` | Tools |
| `benchmark_graph_operations.py` | `tools/benchmark_graph_operations.py` | Tools |
| `performance_test.py` | `tools/performance_test.py` | Tools |

---

## ✅ Benefits of New Structure

1. **Better Organization**: Clear separation of concerns
2. **Easier Navigation**: Find scripts by purpose, not alphabetically
3. **Scalability**: Easy to add new scripts to appropriate category
4. **Documentation**: Each category has clear purpose
5. **Onboarding**: New developers understand structure quickly
6. **Maintenance**: Easier to maintain related scripts together

---

## 🚀 Quick Reference

```bash
# Setup
./scripts/setup/quick_start.sh

# ETL
python scripts/etl/run_etl.py --source data/quy_dinh

# Demo
python scripts/demo/demo_week2.py --test all

# Tools
python scripts/tools/view_graph_nodes.py
```

---

**Last Updated:** November 19, 2025  
**Status:** 🔄 Migration in progress
