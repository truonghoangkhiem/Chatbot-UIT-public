#!/bin/bash

# ============================================================
# CatRAG Graph Initialization Script - Week 1 Task A3
# ============================================================
# Purpose: Run all Cypher scripts to initialize Neo4j schema
# Created: November 14, 2025
# Owner: Team A - Infrastructure
# Usage: ./run_all_init.sh
# ============================================================

set -e  # Exit on error

# Configuration
NEO4J_URI="bolt://localhost:7687"
NEO4J_USER="neo4j"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}CatRAG Graph Initialization${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# Check if Neo4j is running
echo -e "${BLUE}📡 Checking Neo4j connection...${NC}"
if ! curl -s http://localhost:7474 > /dev/null; then
    echo -e "${RED}❌ Error: Neo4j is not running!${NC}"
    echo -e "${YELLOW}Please start Neo4j with:${NC}"
    echo -e "${YELLOW}  docker-compose -f docker/docker-compose.neo4j.yml up -d${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Neo4j is running${NC}"
echo ""

# Function to execute Cypher file
execute_cypher() {
    local file=$1
    local description=$2
    
    echo -e "${BLUE}📝 ${description}...${NC}"
    echo -e "${YELLOW}   File: ${file}${NC}"
    
    # Use cypher-shell to execute the file
    # Install cypher-shell if not available: docker exec neo4j-catrag cypher-shell
    docker exec -i neo4j-catrag cypher-shell \
        -u "$NEO4J_USER" \
        -p "$NEO4J_PASSWORD" \
        -a "$NEO4J_URI" \
        --format plain \
        < "$file"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ${description} completed${NC}"
    else
        echo -e "${RED}❌ Error executing ${file}${NC}"
        exit 1
    fi
    echo ""
}

# Step 1: Create constraints
execute_cypher \
    "$SCRIPT_DIR/01_create_constraints.cypher" \
    "Step 1/3: Creating constraints"

# Step 2: Create indexes
execute_cypher \
    "$SCRIPT_DIR/02_create_indexes.cypher" \
    "Step 2/3: Creating indexes"

# Step 3: Load sample data
execute_cypher \
    "$SCRIPT_DIR/03_load_sample_data.cypher" \
    "Step 3/3: Loading sample data"

# Verification
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}📊 Database Statistics${NC}"
echo -e "${BLUE}============================================================${NC}"

docker exec -i neo4j-catrag cypher-shell \
    -u "$NEO4J_USER" \
    -p "$NEO4J_PASSWORD" \
    -a "$NEO4J_URI" \
    --format plain \
    "MATCH (n) RETURN labels(n)[0] as NodeType, count(*) as Count ORDER BY Count DESC;"

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🔗 Relationship Statistics${NC}"
echo -e "${BLUE}============================================================${NC}"

docker exec -i neo4j-catrag cypher-shell \
    -u "$NEO4J_USER" \
    -p "$NEO4J_PASSWORD" \
    -a "$NEO4J_URI" \
    --format plain \
    "MATCH ()-[r]->() RETURN type(r) as RelType, count(*) as Count ORDER BY Count DESC;"

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}✅ Graph initialization completed successfully!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Access Neo4j Browser: ${BLUE}http://localhost:7474${NC}"
echo -e "  2. Login with username: ${BLUE}neo4j${NC}, password: ${BLUE}[your NEO4J_PASSWORD]${NC}"
echo -e "  3. Test prerequisite query:"
echo -e "     ${YELLOW}MATCH path = (target:MON_HOC {ma_mon: 'SE363'})-[:DIEU_KIEN_TIEN_QUYET*]->(prereq)${NC}"
echo -e "     ${YELLOW}RETURN path;${NC}"
echo -e "  4. Continue with Task A4: GraphRepository POC Enhancement"
echo ""
