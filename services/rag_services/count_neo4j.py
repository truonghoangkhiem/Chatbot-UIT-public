#!/usr/bin/env python3
"""Count Neo4j articles"""
import os
from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', os.getenv('NEO4J_PASSWORD', 'password')))
session = driver.session(database='neo4j')

result = session.run('MATCH (a:Article) RETURN count(a) as count')
print('Total articles in Neo4j:', result.single()['count'])

result = session.run('MATCH (n) RETURN labels(n) as labels, count(*) as count ORDER BY count DESC')
print('\nNode counts by label:')
for r in result:
    print(f"  {r['labels']}: {r['count']}")

driver.close()
