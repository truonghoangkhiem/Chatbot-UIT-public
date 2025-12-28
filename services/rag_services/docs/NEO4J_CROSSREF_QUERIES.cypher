// ============================================================================
// Neo4j Browser Queries - Cross-Reference Visualization
// ============================================================================
// Copy-paste these queries into Neo4j Browser (http://localhost:7474)
// to explore the cross-reference network
// ============================================================================

// 1. Overview: Show Article 6 and all articles that reference it
// Article 6 is the most referenced (17 incoming links)
MATCH (target:Article {article_no: 6})<-[r:REFERENCES]-(source:Article)
RETURN target, r, source

// 2. Citation Network: Show all cross-references
// Visualizes the full regulation dependency network
MATCH (source:Article)-[r:REFERENCES]->(target:Article)
RETURN source, r, target

// 3. Hub Analysis: Find the top 5 most referenced articles
// These are "hub" articles that many others depend on
MATCH (a:Article)<-[r:REFERENCES]-()
WITH a, count(r) as ref_count
ORDER BY ref_count DESC
LIMIT 5
RETURN a.article_no as article,
       a.title_vi as title,
       ref_count

// 4. Dependency Check: What happens if Article 6 changes?
// Find all articles (1-2 hops away) that depend on Article 6
MATCH (a:Article {article_no: 6})<-[:REFERENCES*1..2]-(dependent)
RETURN DISTINCT dependent.article_no as article,
                dependent.title_vi as title

// 5. Path Finding: How is Article 18 connected to Article 21?
// Find the shortest citation path between two articles
MATCH path = shortestPath(
  (a:Article {article_no: 18})-[:REFERENCES*]-(b:Article {article_no: 21})
)
RETURN path

// 6. Orphan Detection: Find articles with NO references (in or out)
// These are independent regulations
MATCH (a:Article)
WHERE NOT (a)-[:REFERENCES]-() AND NOT ()-[:REFERENCES]->(a)
RETURN a.article_no as article,
       a.title_vi as title

// 7. Bidirectional References: Find mutual dependencies
// A references B AND B references A
MATCH (a:Article)-[:REFERENCES]->(b:Article)-[:REFERENCES]->(a)
RETURN a.article_no, b.article_no

// 8. Reference Details: Show reference text for Article 16
// See how other articles mention Article 16
MATCH (source)-[r:REFERENCES]->(a:Article {article_no: 16})
RETURN source.article_no as from_article,
       r.reference_text as how_referenced,
       r.reference_type as pattern_type

// 9. Clause-Level References: Find clauses that reference other articles
// More granular than article-level
MATCH (cl:Clause)-[r:REFERENCES]->(a:Article)
RETURN cl.clause_id, r.reference_text, a.article_no

// 10. Full Context: Article with all its referenced articles
// Use for context expansion in RAG queries
MATCH (a:Article {article_no: 16})
OPTIONAL MATCH (a)-[:REFERENCES]->(ref:Article)
RETURN a as main_article,
       collect(ref) as referenced_articles

// ============================================================================
// Advanced Queries
// ============================================================================

// 11. Citation Strength: Count references between article pairs
MATCH (a:Article)-[r:REFERENCES]->(b:Article)
WITH a, b, count(r) as ref_count
WHERE ref_count > 1
RETURN a.article_no, b.article_no, ref_count
ORDER BY ref_count DESC

// 12. Community Detection: Find clusters of related articles
// Articles that reference each other form communities
MATCH (a:Article)-[:REFERENCES]->(b:Article)
WHERE a.article_no < b.article_no  // Avoid duplicates
RETURN a, b
LIMIT 50

// 13. Reference Statistics per Chapter
// Which chapters have most cross-references?
MATCH (ch:Chapter)-[:HAS_ARTICLE]->(a:Article)
OPTIONAL MATCH (a)-[r:REFERENCES]->()
WITH ch, count(DISTINCT a) as articles, count(r) as refs
RETURN ch.title_vi as chapter,
       articles,
       refs,
       CASE WHEN articles > 0 THEN toFloat(refs)/articles ELSE 0 END as avg_refs_per_article
ORDER BY refs DESC

// 14. Deep Dependency Analysis: 3-hop citation chains
// Find complex dependencies (A → B → C → D)
MATCH path = (a:Article)-[:REFERENCES*3..3]->(target:Article)
RETURN [n in nodes(path) | n.article_no] as citation_chain,
       length(path) as depth
LIMIT 10

// 15. Compliance Impact Matrix
// For each article, show how many others depend on it (incoming)
// and how many it depends on (outgoing)
MATCH (a:Article)
OPTIONAL MATCH (a)<-[in:REFERENCES]-()
OPTIONAL MATCH (a)-[out:REFERENCES]->()
WITH a,
     count(DISTINCT in) as incoming_refs,
     count(DISTINCT out) as outgoing_refs
RETURN a.article_no as article,
       a.title_vi as title,
       incoming_refs,
       outgoing_refs,
       incoming_refs + outgoing_refs as total_connectivity
ORDER BY total_connectivity DESC

// ============================================================================
// Visualization Settings (for Neo4j Browser)
// ============================================================================
// Click the "settings" icon in Browser and use these:
//
// Node Colors:
// - Article: #FF6B6B (red)
// - Clause: #4ECDC4 (teal)
//
// Relationship Colors:
// - REFERENCES: #FFE66D (yellow)
// - HAS_ARTICLE: #95E1D3 (green)
//
// Node Size:
// Based on degree (number of connections)
//
// Node Caption:
// Article: article_no + " - " + title_vi
// Clause: clause_no
// ============================================================================
