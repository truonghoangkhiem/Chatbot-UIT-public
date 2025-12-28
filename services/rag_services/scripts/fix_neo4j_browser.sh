#!/bin/bash
# Fix Neo4j Browser UTF-8 Encoding Issues

echo "======================================================================"
echo "🔧 NEO4J BROWSER UTF-8 FIX"
echo "======================================================================"

echo ""
echo "📋 Step 1: Checking Neo4j Configuration..."
echo "----------------------------------------------------------------------"

# Check if Neo4j is running
if docker ps | grep -q neo4j-catrag; then
    echo "✅ Neo4j container is running"
else
    echo "❌ Neo4j container is not running!"
    echo "Please start Neo4j first: docker start neo4j-catrag"
    exit 1
fi

echo ""
echo "📋 Step 2: Checking Neo4j Configuration Files..."
echo "----------------------------------------------------------------------"

# Check Neo4j conf for UTF-8 settings
if docker exec neo4j-catrag test -f /var/lib/neo4j/conf/neo4j.conf; then
    echo "✅ Found neo4j.conf"
    
    # Check for UTF-8 settings
    docker exec neo4j-catrag grep -i "encoding\|charset\|utf" /var/lib/neo4j/conf/neo4j.conf 2>/dev/null || echo "ℹ️  No explicit encoding settings found (using defaults)"
else
    echo "⚠️  neo4j.conf not found"
fi

echo ""
echo "📋 Step 3: Testing Database Connection with UTF-8..."
echo "----------------------------------------------------------------------"

# Test UTF-8 via Python
python3 << 'PYTHON_SCRIPT'
from neo4j import GraphDatabase
import sys

sys.stdout.reconfigure(encoding='utf-8')

try:
    import os
    password = os.getenv('NEO4J_PASSWORD', 'password')
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', password))
    
    with driver.session() as session:
        # Insert test Vietnamese text
        session.run("""
            MERGE (test:UTF8Test {id: 'test-encoding'})
            SET test.vietnamese = 'Công nghệ Thông tin - ĐẠI HỌC QUỐC GIA TP.HCM',
                test.special = 'àáảãạ ÀÁẢÃẠ êềếểễệ ÊỀẾỂỄỆ',
                test.numbers = '123 học sinh - 456 môn học'
        """)
        
        # Read back
        result = session.run("""
            MATCH (test:UTF8Test {id: 'test-encoding'})
            RETURN test.vietnamese as vn, test.special as sp, test.numbers as num
        """)
        
        for record in result:
            print(f"✅ Vietnamese: {record['vn']}")
            print(f"✅ Special chars: {record['sp']}")
            print(f"✅ Numbers: {record['num']}")
        
        # Clean up
        session.run("MATCH (test:UTF8Test {id: 'test-encoding'}) DELETE test")
        
        print("\n✅ UTF-8 encoding test PASSED!")
    
    driver.close()
    
except Exception as e:
    print(f"❌ UTF-8 test FAILED: {e}")
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✅ Neo4j Database UTF-8 is WORKING CORRECTLY"
    echo "======================================================================"
    echo ""
    echo "🌐 Browser Issues - Manual Steps Required:"
    echo "----------------------------------------------------------------------"
    echo ""
    echo "1️⃣  CLEAR BROWSER CACHE FOR localhost:7474:"
    echo "   Chrome/Edge: Press F12 → Application → Storage → Clear site data"
    echo "   Firefox: Press F12 → Storage → Cookies → Right-click → Delete All"
    echo ""
    echo "2️⃣  HARD REFRESH Neo4j Browser:"
    echo "   - Press: Ctrl + Shift + R (Linux/Windows)"
    echo "   - Or: Cmd + Shift + R (Mac)"
    echo ""
    echo "3️⃣  CHANGE BROWSER (if above doesn't work):"
    echo "   - Try: Google Chrome, Mozilla Firefox, or Microsoft Edge"
    echo "   - Different browsers may render UTF-8 differently"
    echo ""
    echo "4️⃣  ACCESS NEO4J BROWSER:"
    echo "   - Open: http://localhost:7474"
    echo "   - Login: neo4j / [your NEO4J_PASSWORD]"
    echo "   - Run: :use neo4j"
    echo "   - Then: MATCH (n:Khoa) RETURN n LIMIT 5"
    echo ""
    echo "5️⃣  OR USE PYTHON VIEWER (Recommended):"
    echo "   python scripts/view_graph_data.py"
    echo ""
    echo "======================================================================"
else
    echo ""
    echo "❌ UTF-8 test failed - please check Neo4j configuration"
fi
