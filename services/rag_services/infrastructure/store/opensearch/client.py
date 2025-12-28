# store/opensearch/client.py
#
# Description:
# This module provides OpenSearch client functionality for BM25 text search.
# It handles document indexing, searching, and manages the OpenSearch connection.

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, RequestError

from app.config.settings import settings

logger = logging.getLogger(__name__)

class OpenSearchClient:
    """OpenSearch client for BM25 text search and document management."""
    
    def __init__(self):
        """Initialize OpenSearch client with configuration from settings."""
        self.client = OpenSearch(
            hosts=[{
                'host': settings.opensearch_host, 
                'port': settings.opensearch_port
            }],
            http_auth=(settings.opensearch_username, settings.opensearch_password),
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=settings.opensearch_verify_certs,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        self.index_name = settings.opensearch_index
        self._ensure_index()

    def _get_index_mapping(self) -> Dict[str, Any]:
        """Define the index mapping for document storage with Vietnamese support."""
        return {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "text": {
                        "type": "text",
                        "analyzer": "vietnamese_analyzer",
                        "search_analyzer": "vietnamese_search_analyzer",
                        "fields": {
                            "raw": {
                                "type": "keyword",
                                "ignore_above": 512
                            },
                            "standard": {
                                "type": "text",
                                "analyzer": "standard"
                            }
                        }
                    },
                    "title": {
                        "type": "text",
                        "analyzer": "vietnamese_analyzer",
                        "search_analyzer": "vietnamese_search_analyzer"
                    },
                    "doc_type": {
                        "type": "keyword",
                        "doc_values": True
                    },
                    "faculty": {
                        "type": "keyword", 
                        "doc_values": True
                    },
                    "year": {
                        "type": "integer",
                        "doc_values": True
                    },
                    "subject": {
                        "type": "keyword",
                        "doc_values": True
                    },
                    "language": {
                        "type": "keyword",
                        "doc_values": True
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "chunk_size": {"type": "integer"},
                            "char_start": {"type": "integer"},
                            "char_end": {"type": "integer"},
                            "section": {"type": "keyword"},
                            "subsection": {"type": "keyword"}
                        }
                    },
                    "char_spans": {
                        "type": "nested",
                        "properties": {
                            "start": {"type": "integer"},
                            "end": {"type": "integer"},
                            "text": {"type": "text"},
                            "type": {"type": "keyword"}
                        }
                    },
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "max_result_window": 50000,
                "analysis": {
                    "filter": {
                        "vietnamese_stop": {
                            "type": "stop",
                            "stopwords": [
                                "của", "và", "trong", "với", "đã", "được", "có", "là", "từ", 
                                "cho", "về", "theo", "như", "khi", "nếu", "để", "sẽ", "đến",
                                "tại", "các", "những", "một", "này", "đó", "không", "hoặc"
                            ]
                        },
                        "vietnamese_stemmer": {
                            "type": "stemmer",
                            "language": "light_english"
                        },
                        "ascii_folding": {
                            "type": "asciifolding",
                            "preserve_original": "true"
                        }
                    },
                    "char_filter": {
                        "vietnamese_char_filter": {
                            "type": "mapping",
                            "mappings": [
                                "đ => d", "Đ => D",
                                "ă => a", "â => a", "á => a", "à => a", "ả => a", "ã => a", "ạ => a",
                                "ấ => a", "ầ => a", "ẩ => a", "ẫ => a", "ậ => a",
                                "ắ => a", "ằ => a", "ẳ => a", "ẵ => a", "ặ => a"
                            ]
                        }
                    },
                    "analyzer": {
                        "vietnamese_analyzer": {
                            "type": "custom",
                            "char_filter": ["vietnamese_char_filter"],
                            "tokenizer": "standard",  # Changed from icu_tokenizer to standard
                            "filter": [
                                "lowercase",
                                "ascii_folding", 
                                "vietnamese_stop",
                                "vietnamese_stemmer"
                            ]
                        },
                        "vietnamese_search_analyzer": {
                            "type": "custom",
                            "char_filter": ["vietnamese_char_filter"],
                            "tokenizer": "standard",  # Changed from icu_tokenizer to standard
                            "filter": [
                                "lowercase",
                                "ascii_folding",
                                "vietnamese_stop"
                            ]
                        },
                        "keyword_analyzer": {
                            "type": "custom",
                            "tokenizer": "keyword",
                            "filter": ["lowercase", "ascii_folding"]
                        }
                    },
                    "normalizer": {
                        "vietnamese_normalizer": {
                            "type": "custom",
                            "char_filter": ["vietnamese_char_filter"],
                            "filter": ["lowercase", "ascii_folding"]
                        }
                    }
                }
            }
        }

    def _ensure_index(self):
        """Create index if it doesn't exist."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                mapping = self._get_index_mapping()
                self.client.indices.create(
                    index=self.index_name, 
                    body=mapping
                )
                logger.info(f"Created OpenSearch index: {self.index_name}")
            else:
                logger.info(f"OpenSearch index {self.index_name} already exists")
        except Exception as e:
            logger.error(f"Error creating OpenSearch index: {e}")
            raise

    def index_document(self, doc_id: str, chunk_id: str, text: str, 
                      metadata: Optional[Dict[str, Any]] = None,
                      doc_type: Optional[str] = None,
                      faculty: Optional[str] = None,
                      year: Optional[int] = None,
                      subject: Optional[str] = None,
                      language: str = "vi",
                      char_spans: Optional[List[Dict[str, Any]]] = None,
                      title: Optional[str] = None) -> bool:
        """
        Index a single document chunk with enhanced metadata.
        
        Args:
            doc_id: Document identifier
            chunk_id: Chunk identifier within the document
            text: Text content to index
            metadata: Additional metadata
            doc_type: Document type (e.g., "syllabus", "regulation", "guide")
            faculty: Faculty name (e.g., "CNTT", "KHTN", "CTDA")
            year: Academic year
            subject: Subject/course name
            language: Document language (default: "vi")
            char_spans: Character spans for citation
            title: Document title
            
        Returns:
            bool: Success status
        """
        try:
            # Extract char spans from metadata if not provided
            if not char_spans and metadata:
                char_spans = self._extract_char_spans(text, metadata)
            
            document = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": text,
                "title": title or metadata.get("title", "") if metadata else "",
                "doc_type": doc_type or (metadata.get("doc_type") if metadata else "document"),
                "faculty": faculty or (metadata.get("faculty") if metadata else "general"),
                "year": year or (metadata.get("year") if metadata else None),
                "subject": subject or (metadata.get("subject") if metadata else ""),
                "language": language,
                "metadata": self._enrich_metadata(metadata or {}),
                "char_spans": char_spans or [],
                "created_at": datetime.utcnow().isoformat() + "Z",
                "updated_at": datetime.utcnow().isoformat() + "Z"
            }
            
            # Use doc_id + chunk_id as the unique document ID
            elasticsearch_doc_id = f"{doc_id}_{chunk_id}"
            
            response = self.client.index(
                index=self.index_name,
                id=elasticsearch_doc_id,
                body=document
            )
            
            return response.get("result") in ["created", "updated"]
            
        except Exception as e:
            logger.error(f"Error indexing document {doc_id}_{chunk_id}: {e}")
            return False

    def _enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich metadata with char offsets and additional info."""
        enriched = metadata.copy()
        
        # Ensure char positions are integers if they exist
        if "char_start" in enriched:
            enriched["char_start"] = int(enriched["char_start"])
        if "char_end" in enriched:
            enriched["char_end"] = int(enriched["char_end"])
            
        return enriched

    def _extract_char_spans(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract character spans for citation purposes."""
        spans = []
        
        # If we have char start/end in metadata
        if "char_start" in metadata and "char_end" in metadata:
            spans.append({
                "start": int(metadata["char_start"]),
                "end": int(metadata["char_end"]),
                "text": text[:100] + "..." if len(text) > 100 else text,
                "type": "content"
            })
        
        # Extract sentence spans for better citation granularity
        sentences = text.split('. ')
        char_pos = 0
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 10:  # Only meaningful sentences
                spans.append({
                    "start": char_pos,
                    "end": char_pos + len(sentence),
                    "text": sentence.strip(),
                    "type": "sentence"
                })
            char_pos += len(sentence) + 2  # +2 for '. '
            
        return spans

    def bulk_index_documents(self, documents: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Bulk index multiple documents.
        
        Args:
            documents: List of documents to index
            
        Returns:
            Tuple[int, int]: (successful_count, failed_count)
        """
        if not documents:
            return 0, 0
            
        try:
            actions = []
            for doc in documents:
                elasticsearch_doc_id = f"{doc['doc_id']}_{doc['chunk_id']}"
                
                # Extract metadata fields
                metadata = doc.get("metadata", {})
                
                action = {
                    "_index": self.index_name,
                    "_id": elasticsearch_doc_id,
                    "_source": {
                        "doc_id": doc["doc_id"],
                        "chunk_id": doc["chunk_id"],
                        "text": doc["text"],
                        "title": doc.get("title", metadata.get("title", "")),
                        "doc_type": doc.get("doc_type", metadata.get("doc_type", "document")),
                        "faculty": doc.get("faculty", metadata.get("faculty", "general")),
                        "year": doc.get("year", metadata.get("year")),
                        "subject": doc.get("subject", metadata.get("subject", "")),
                        "language": doc.get("language", metadata.get("language", "vi")),
                        "metadata": self._enrich_metadata(metadata),
                        "char_spans": doc.get("char_spans", []),
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "updated_at": datetime.utcnow().isoformat() + "Z"
                    }
                }
                actions.append(action)
            
            # Use the helpers.bulk for efficient bulk indexing
            from opensearchpy import helpers
            success, failed = helpers.bulk(
                self.client,
                actions,
                chunk_size=100,
                request_timeout=60
            )
            
            logger.info(f"Bulk indexed: {success} successful, {len(failed) if isinstance(failed, list) else failed} failed")
            return success, len(failed) if isinstance(failed, list) else failed
            
        except Exception as e:
            logger.error(f"Error in bulk indexing: {e}")
            return 0, len(documents)

    def search(self, query: str, size: int = 10, filters: Optional[Dict[str, Any]] = None,
               doc_types: Optional[List[str]] = None,
               faculties: Optional[List[str]] = None,
               years: Optional[List[int]] = None,
               subjects: Optional[List[str]] = None,
               language: Optional[str] = None,
               include_char_spans: bool = True) -> List[Dict[str, Any]]:
        """
        Perform advanced BM25 search with Vietnamese analyzer and field filters.
        
        Args:
            query: Search query text
            size: Number of results to return
            filters: Optional metadata filters (legacy support)
            doc_types: Filter by document types
            faculties: Filter by faculties
            years: Filter by years
            subjects: Filter by subjects
            language: Filter by language
            include_char_spans: Include character spans in results
            
        Returns:
            List of search results with scores and enhanced metadata
        """
        try:
            # Build enhanced search query
            search_body = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "bool": {
                                    "should": [
                                        {
                                            "multi_match": {
                                                "query": query,
                                                "fields": [
                                                    "text^1.0", 
                                                    "title^1.5"
                                                ],
                                                "type": "best_fields",
                                                "fuzziness": "AUTO"
                                            }
                                        },
                                        {
                                            "match_phrase": {
                                                "text": {
                                                    "query": query,
                                                    "boost": 1.2
                                                }
                                            }
                                        }
                                    ]
                                }
                            }
                        ],
                        "filter": []
                    }
                },
                "size": size,
                "_source": [
                    "doc_id", "chunk_id", "text", "title", "doc_type", 
                    "faculty", "year", "subject", "language", "metadata", "char_spans"
                ],
                "highlight": {
                    "fields": {
                        "text": {
                            "fragment_size": 200,
                            "number_of_fragments": 2,
                            "pre_tags": ["<mark>"],
                            "post_tags": ["</mark>"]
                        },
                        "title": {
                            "fragment_size": 100,
                            "number_of_fragments": 1
                        }
                    }
                }
            }
            
            # Build filter conditions
            filter_conditions = []
            
            # Document type filters
            if doc_types:
                filter_conditions.append({"terms": {"doc_type": doc_types}})
            
            # Faculty filters
            if faculties:
                filter_conditions.append({"terms": {"faculty": faculties}})
                
            # Year filters
            if years:
                filter_conditions.append({"terms": {"year": years}})
                
            # Subject filters
            if subjects:
                filter_conditions.append({"terms": {"subject": subjects}})
                
            # Language filter
            if language:
                filter_conditions.append({"term": {"language": language}})
            
            # Legacy metadata filters (for backward compatibility)
            if filters:
                for key, value in filters.items():
                    if key in ["doc_type", "faculty", "year", "subject", "language"]:
                        # Use direct field access for known fields
                        if isinstance(value, list):
                            filter_conditions.append({"terms": {key: value}})
                        else:
                            filter_conditions.append({"term": {key: value}})
                    else:
                        # Use metadata prefix for unknown fields
                        if isinstance(value, list):
                            filter_conditions.append({"terms": {f"metadata.{key}": value}})
                        else:
                            filter_conditions.append({"term": {f"metadata.{key}": value}})
            
            # Apply filters
            if filter_conditions:
                search_body["query"]["bool"]["filter"] = filter_conditions
                
            # Add aggregations for faceted search
            search_body["aggs"] = {
                "doc_types": {
                    "terms": {"field": "doc_type", "size": 20}
                },
                "faculties": {
                    "terms": {"field": "faculty", "size": 20}
                },
                "years": {
                    "terms": {"field": "year", "size": 10}
                },
                "subjects": {
                    "terms": {"field": "subject", "size": 30}
                }
            }

            response = self.client.search(
                index=self.index_name,
                body=search_body
            )
            
            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                
                result = {
                    "doc_id": source["doc_id"],
                    "chunk_id": source["chunk_id"],
                    "text": source["text"],
                    "title": source.get("title", ""),
                    "doc_type": source.get("doc_type", "document"),
                    "faculty": source.get("faculty", "general"),
                    "year": source.get("year"),
                    "subject": source.get("subject", ""),
                    "language": source.get("language", "vi"),
                    "metadata": source.get("metadata", {}),
                    "bm25_score": hit["_score"],
                    "elasticsearch_id": hit["_id"]
                }
                
                # Add character spans for citation
                if include_char_spans and "char_spans" in source:
                    result["char_spans"] = source["char_spans"]
                
                # Add highlighted text
                if "highlight" in hit:
                    result["highlighted_text"] = hit["highlight"].get("text", [])
                    result["highlighted_title"] = hit["highlight"].get("title", [])
                
                results.append(result)
            
            # Add aggregation results for faceted search
            if "aggregations" in response:
                for result in results:
                    result["facets"] = {
                        "doc_types": response["aggregations"]["doc_types"]["buckets"],
                        "faculties": response["aggregations"]["faculties"]["buckets"],
                        "years": response["aggregations"]["years"]["buckets"],
                        "subjects": response["aggregations"]["subjects"]["buckets"]
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching OpenSearch: {e}")
            return []

    def delete_document(self, doc_id: str, chunk_id: str) -> bool:
        """Delete a specific document chunk."""
        try:
            elasticsearch_doc_id = f"{doc_id}_{chunk_id}"
            self.client.delete(
                index=self.index_name,
                id=elasticsearch_doc_id
            )
            return True
        except NotFoundError:
            logger.warning(f"Document {doc_id}_{chunk_id} not found for deletion")
            return False
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}_{chunk_id}: {e}")
            return False

    def delete_all_documents_for_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for a given document ID."""
        try:
            query = {"query": {"term": {"doc_id": doc_id}}}
            response = self.client.delete_by_query(
                index=self.index_name,
                body=query
            )
            deleted_count = response.get("deleted", 0)
            logger.info(f"Deleted {deleted_count} chunks for doc_id: {doc_id}")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting documents for doc_id {doc_id}: {e}")
            return 0

    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self.client.indices.stats(index=self.index_name)
            return {
                "total_docs": stats["indices"][self.index_name]["total"]["docs"]["count"],
                "index_size": stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
            }
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"total_docs": 0, "index_size": 0}

    def health_check(self) -> bool:
        """Check if OpenSearch is healthy and accessible."""
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {e}")
            return False

# Global instance for reuse
_opensearch_client = None

def get_opensearch_client() -> OpenSearchClient:
    """Get the singleton OpenSearch client instance."""
    global _opensearch_client
    if _opensearch_client is None:
        _opensearch_client = OpenSearchClient()
    return _opensearch_client
