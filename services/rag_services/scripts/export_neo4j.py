"""
Export Neo4j graph to JSON format.
"""
import os
from neo4j import GraphDatabase
import json
from datetime import datetime

def export_graph():
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', os.getenv('NEO4J_PASSWORD', 'password')))
    
    with driver.session() as session:
        # Get all nodes
        nodes_result = session.run('''
            MATCH (n)
            RETURN id(n) AS id, labels(n) AS labels, properties(n) AS props
        ''')
        nodes = []
        for record in nodes_result:
            props = {}
            for k, v in record['props'].items():
                if hasattr(v, 'isoformat'):
                    props[k] = v.isoformat()
                else:
                    props[k] = v
            node = {
                'id': record['id'],
                'labels': record['labels'],
                'properties': props
            }
            nodes.append(node)
        
        # Get all relationships
        rels_result = session.run('''
            MATCH (a)-[r]->(b)
            RETURN id(a) AS source_id, id(b) AS target_id, type(r) AS type, properties(r) AS props,
                   a.id AS source_name, b.id AS target_name, 
                   CASE WHEN a.text IS NOT NULL THEN a.text ELSE a.id END AS source_text,
                   CASE WHEN b.text IS NOT NULL THEN b.text ELSE b.id END AS target_text,
                   labels(a)[0] AS source_label, labels(b)[0] AS target_label
        ''')
        relationships = []
        for record in rels_result:
            props = {}
            for k, v in record['props'].items():
                if hasattr(v, 'isoformat'):
                    props[k] = v.isoformat()
                else:
                    props[k] = v
            rel = {
                'source_id': record['source_id'],
                'target_id': record['target_id'],
                'source_name': record['source_name'],
                'target_name': record['target_name'],
                'source_text': record['source_text'],
                'target_text': record['target_text'],
                'source_label': record['source_label'],
                'target_label': record['target_label'],
                'type': record['type'],
                'properties': props
            }
            relationships.append(rel)

    driver.close()

    # Export
    export_data = {
        'exported_at': datetime.now().isoformat(),
        'stats': {
            'nodes': len(nodes),
            'relationships': len(relationships)
        },
        'nodes': nodes,
        'relationships': relationships
    }

    output_path = '../data/extraction_results/neo4j_graph_export.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f'Exported {len(nodes)} nodes, {len(relationships)} relationships')
    print(f'Saved to: {output_path}')
    return export_data

if __name__ == "__main__":
    export_graph()
