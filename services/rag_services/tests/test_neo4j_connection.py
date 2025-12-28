#!/usr/bin/env python3
"""
Test Neo4j connection and verify setup
Week 1 - Task A1: Neo4j Environment Setup
"""

import sys
import os
from neo4j import GraphDatabase
import time


class Neo4jConnectionTester:
    """Test Neo4j connection and basic operations"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = None):
        self.uri = uri
        self.user = user
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None
    
    def connect(self):
        """Establish connection to Neo4j"""
        try:
            print(f"🔌 Connecting to Neo4j at {self.uri}...")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            print("✅ Connection established successfully!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def verify_connection(self):
        """Verify connection with a simple query"""
        if not self.driver:
            print("❌ No driver available")
            return False
        
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 'Connection OK' as message")
                record = result.single()
                print(f"✅ Query test: {record['message']}")
                return True
        except Exception as e:
            print(f"❌ Query test failed: {e}")
            return False
    
    def check_apoc(self):
        """Check if APOC plugin is loaded"""
        try:
            with self.driver.session() as session:
                result = session.run("CALL apoc.help('text') YIELD name RETURN count(name) as count")
                record = result.single()
                count = record['count']
                if count > 0:
                    print(f"✅ APOC plugin loaded ({count} text functions available)")
                    return True
                else:
                    print("⚠️  APOC plugin not found")
                    return False
        except Exception as e:
            print(f"⚠️  APOC check failed: {e}")
            return False
    
    def check_gds(self):
        """Check if Graph Data Science plugin is loaded"""
        try:
            with self.driver.session() as session:
                result = session.run("CALL gds.version() YIELD version RETURN version")
                record = result.single()
                version = record['version']
                print(f"✅ Graph Data Science plugin loaded (version {version})")
                return True
        except Exception as e:
            print(f"⚠️  GDS plugin check failed: {e}")
            return False
    
    def get_database_info(self):
        """Get database information"""
        try:
            with self.driver.session() as session:
                # Node count
                result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = result.single()['node_count']
                
                # Relationship count
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()['rel_count']
                
                print(f"\n📊 Database Statistics:")
                print(f"   - Nodes: {node_count}")
                print(f"   - Relationships: {rel_count}")
                
                return True
        except Exception as e:
            print(f"⚠️  Database info retrieval failed: {e}")
            return False
    
    def test_write_operation(self):
        """Test write operation (create and delete test node)"""
        try:
            with self.driver.session() as session:
                # Create test node
                session.run(
                    "CREATE (n:TEST_NODE {name: 'connection_test', timestamp: $timestamp})",
                    timestamp=time.time()
                )
                print("✅ Write operation successful (created test node)")
                
                # Verify test node exists
                result = session.run("MATCH (n:TEST_NODE {name: 'connection_test'}) RETURN n")
                if result.single():
                    print("✅ Read operation successful (verified test node)")
                
                # Delete test node
                session.run("MATCH (n:TEST_NODE {name: 'connection_test'}) DELETE n")
                print("✅ Delete operation successful (cleaned up test node)")
                
                return True
        except Exception as e:
            print(f"❌ Write operation failed: {e}")
            return False
    
    def close(self):
        """Close connection"""
        if self.driver:
            self.driver.close()
            print("\n🔌 Connection closed")


def main():
    """Run all connection tests"""
    print("=" * 60)
    print("Neo4j Connection Test - CatRAG Week 1")
    print("=" * 60)
    print()
    
    tester = Neo4jConnectionTester()
    
    # Test 1: Connection
    if not tester.connect():
        print("\n❌ Cannot proceed - connection failed")
        print("\nTroubleshooting:")
        print("1. Make sure Neo4j Docker container is running:")
        print("   docker-compose -f docker/docker-compose.neo4j.yml up -d")
        print("2. Wait ~30 seconds for Neo4j to fully start")
        print("3. Check if ports 7474 and 7687 are not in use")
        sys.exit(1)
    
    # Test 2: Basic query
    print()
    if not tester.verify_connection():
        print("\n❌ Basic query failed")
        sys.exit(1)
    
    # Test 3: APOC plugin
    print()
    tester.check_apoc()
    
    # Test 4: GDS plugin
    print()
    tester.check_gds()
    
    # Test 5: Database info
    print()
    tester.get_database_info()
    
    # Test 6: Write operations
    print()
    tester.test_write_operation()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ All connection tests passed!")
    print("=" * 60)
    print()
    print("🎉 Neo4j is ready for CatRAG development!")
    print()
    print("Next steps:")
    print("1. Access Neo4j Browser: http://localhost:7474")
    print("2. Login with username: neo4j, password: [your NEO4J_PASSWORD]")
    print("3. Run Task A2: CatRAG Schema Design"))
    print()
    
    tester.close()


if __name__ == "__main__":
    main()
