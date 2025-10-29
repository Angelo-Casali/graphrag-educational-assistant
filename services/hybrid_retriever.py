"""
Hybrid Retriever for Educational Knowledge Graph
Combines Cypher search with Node2Vec semantic search for enhanced query processing.
"""

import os
import re
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Import existing services
from services.text2cypher import text2cypher, detect_language, extract_keywords, get_neuroscience_fallback
from services.enhanced_graph_retriever import run_graph_query
from services.node2vec_service import node2vec_service

load_dotenv()

class HybridRetriever:
    """
    Hybrid retriever that combines:
    1. Cypher-based structural search
    2. Node2Vec-based semantic search
    3. Keyword-based filtering
    """
    
    def __init__(self, cypher_weight=0.6, semantic_weight=0.4):
        """
        Initialize hybrid retriever with configurable weights.
        
        Args:
            cypher_weight: Weight for Cypher search results (0.0-1.0)
            semantic_weight: Weight for semantic search results (0.0-1.0)
        """
        self.cypher_weight = cypher_weight
        self.semantic_weight = semantic_weight
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        
        # Initialize Node2Vec service
        self.node2vec_available = node2vec_service.load_or_train_model()
        
        # Cache for performance optimization
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
        # Performance monitoring
        self.performance_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time': 0.0,
            'cypher_success_rate': 0.0,
            'semantic_success_rate': 0.0,
            'query_times': []
        }
        
    def process_query(self, query: str, language: str = "auto", 
                     enhance: bool = True, max_results: int = 10) -> Dict[str, Any]:
        """
        Process query using hybrid retrieval approach with performance optimizations.
        
        Args:
            query: Natural language query
            language: Language of the query
            enhance: Whether to enhance query with keywords
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing hybrid search results and metadata
        """
        start_time = time.time()
        
        # Update performance stats
        self.performance_stats['total_queries'] += 1
        
        # Check cache first
        cache_key = f"{query}_{language}_{enhance}"
        if cache_key in self.query_cache:
            cached_result = self.query_cache[cache_key]
            if time.time() - cached_result['timestamp'] < self.cache_ttl:
                cached_result['from_cache'] = True
                cached_result['processing_time'] = f"{time.time() - start_time:.3f}s"
                self.performance_stats['cache_hits'] += 1
                return cached_result
        
        self.performance_stats['cache_misses'] += 1
        
        # Auto-detect language (new optimized system)
        language = detect_language(query)
        keywords = extract_keywords(query, language)

        # PROPER HYBRID APPROACH: Use direct Cypher + Semantic for best of both worlds
        print(f"🔧 HYBRID MODE: Combining direct Cypher + Semantic search for '{query[:50]}..'")

        # Execute search strategies in parallel for true hybrid approach
        semantic_results = []
        cypher_results = []

        # STEP 1: DIRECT CYPHER GENERATION FIRST 
        print("🔍 STEP 1: Direct Cypher Generation")
        cypher_query = text2cypher(query, self.driver)  # Use direct text2cypher function

        if cypher_query:
            print("  ✅ Direct Cypher query generated successfully")            
            cypher_results = self._execute_cypher_search(cypher_query)
            print(f"  📊 Direct Cypher returned {len(cypher_results)} entities")
        else:
            print("  ❌ Direct Cypher generation failed, using fallback")
            fallback_query = get_neuroscience_fallback(query)
            if fallback_query:
                cypher_results = self._execute_cypher_search(fallback_query)
                cypher_query = fallback_query
                print(f"  🔄 Using neuroscience fallback: {len(cypher_results)} entities")

        # STEP 2: SEMANTIC SEARCH SECOND (vector similarity)
        print("🔎 STEP 2: Semantic Search (Node2Vec)")
        if self.node2vec_available:
            semantic_results = self._execute_semantic_search(query, keywords)
            print(f"  📊 Semantic search returned {len(semantic_results)} entities")
        else:
            print("  ❌ Node2Vec not available, skipping semantic search")
        
        # Combine and rank results
        hybrid_results = self._combine_results(
            cypher_results, 
            semantic_results, 
            keywords,
            max_results
        )
        
        # Prepare response
        result = {
            "original_query": query,
            "detected_language": language,
            "keywords": keywords,
            "cypher_query": cypher_query,
            "cypher_results": cypher_results,
            "semantic_results": semantic_results,
            "hybrid_results": hybrid_results,
            "total_results": len(hybrid_results),
            "processing_time": f"{time.time() - start_time:.3f}s",
            "from_cache": False,
            "timestamp": time.time()
        }
        
        # Update performance statistics
        processing_time = time.time() - start_time
        self.performance_stats['query_times'].append(processing_time)
        
        # Keep only last 100 query times for moving average
        if len(self.performance_stats['query_times']) > 100:
            self.performance_stats['query_times'] = self.performance_stats['query_times'][-100:]
        
        # Calculate average processing time
        self.performance_stats['avg_processing_time'] = sum(self.performance_stats['query_times']) / len(self.performance_stats['query_times'])
        
        # Calculate success rates
        cypher_success = len(cypher_results) > 0 if cypher_query else False
        semantic_success = len(semantic_results) > 0 if self.node2vec_available else False
        
        self.performance_stats['cypher_success_rate'] = (self.performance_stats['cypher_success_rate'] * (self.performance_stats['total_queries'] - 1) + (1 if cypher_success else 0)) / self.performance_stats['total_queries']
        self.performance_stats['semantic_success_rate'] = (self.performance_stats['semantic_success_rate'] * (self.performance_stats['total_queries'] - 1) + (1 if semantic_success else 0)) / self.performance_stats['total_queries']
        
        # Cache the result
        self.query_cache[cache_key] = result
        
        # Clean old cache entries
        self._clean_cache()
        
        return result
    
    def _execute_cypher_search(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute Cypher query and validate results against graph reality."""
        try:
            results = run_graph_query(cypher_query)

            # Format results for hybrid processing
            formatted_results = []
            for i, record in enumerate(results):
                formatted_results.append({
                    "id": f"cypher_{i}",
                    "type": "cypher",
                    "score": 1.0,  # Cypher results get base score
                    "data": record,
                    "explanation": f"Found via Cypher query: {cypher_query[:100]}..."
                })

            # FILTER FOR REALITY: Keep only relationships that actually exist in the graph
            from services.text2cypher import filter_real_relationships
            validated_results = filter_real_relationships(formatted_results, self.driver)

            return validated_results

        except Exception as e:
            print(f"❌ Cypher search error: {e}")
            return []
    
    def _execute_semantic_search(self, query: str, keywords: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Execute Node2Vec-based semantic search with educational content prioritization."""
        try:
            # Find similar nodes using Node2Vec
            similar_nodes = node2vec_service.find_similar_nodes(query, top_k=20)  # Increased for better filtering

            # Find similar edges using Node2Vec (enhanced node-edge-node units)
            similar_edges = node2vec_service.find_similar_edges(query, top_k=15)

            # Get detailed information for each similar node
            semantic_results = []

            # Neuroscience content prioritization
            neuroscience_node_types = {
                'Attention', 'Memory', 'CognitiveLoad', 'ExecutiveControl',
                'WorkingMemory', 'LongTermMemory', 'Neuroplasticity'
            }

            # Process nodes with educational filtering
            for node_id, similarity in similar_nodes:
                node_info = node2vec_service.get_detailed_node_info(node_id)

                if node_info:
                    # Extract node type from the info string
                    node_type = self._extract_node_type(node_info)

                    # Prioritize neuroscience content
                    neuroscience_boost = 1.0
                    if node_type in neuroscience_node_types:
                        neuroscience_boost = 2.0  # Double score for neuroscience nodes
                    elif any(word in node_info.lower() for word in ['attention', 'memory', 'cognitive', 'executive', 'neuroplasticity', 'brain']):
                        neuroscience_boost = 1.5  # Boost for neuroscience-related content

                    # Check if node contains relevant keywords
                    keyword_score = self._calculate_keyword_score(node_info, keywords)

                    # Combine similarity, keyword scores, and neuroscience boost
                    combined_score = (similarity * 0.6) + (keyword_score * 0.3) + (neuroscience_boost * 0.1)

                    # Only include results with reasonable relevance
                    if combined_score > 0.3 or node_type in neuroscience_node_types:
                        semantic_results.append({
                            "id": f"semantic_node_{node_id}",
                            "type": "semantic",
                            "subtype": "node",
                            "score": combined_score,
                            "similarity": similarity,
                            "keyword_score": keyword_score,
                            "neuroscience_boost": neuroscience_boost,
                            "node_type": node_type,
                            "data": {"node_info": node_info, "node_id": node_id},
                            "explanation": f"Neuroscience node (type: {node_type}, similarity: {similarity:.3f})"
                        })

            # Process edges with neuroscience filtering
            for source_node, target_node, similarity, edge_data in similar_edges:
                # Get detailed information for connected nodes
                u_info = node2vec_service.get_detailed_node_info(source_node)
                v_info = node2vec_service.get_detailed_node_info(target_node)

                if u_info and v_info:
                    rel_type = edge_data.get('type', 'unknown')

                    # Only include neuroscience relationships
                    neuroscience_relationships = ['SUPPORTS', 'HINDERS', 'IS_GOVERNED_BY', 'FACILITATES', 'IMPAIRS', 'ENHANCES']
                    if rel_type in neuroscience_relationships:
                        # Create enhanced edge explanation with node-edge-node context
                        edge_text = f"{u_info} --[{rel_type}]--> {v_info}"

                        # Check if edge contains relevant keywords
                        keyword_score = self._calculate_keyword_score(edge_text, keywords)

                        # Neuroscience boost for meaningful relationships
                        neuroscience_boost = 1.5 if rel_type in ['SUPPORTS', 'FACILITATES', 'ENHANCES'] else 1.0

                        # Combine similarity and keyword scores (higher weight for edges)
                        combined_score = (similarity * 0.7) + (keyword_score * 0.2) + (neuroscience_boost * 0.1)

                        # Only include highly relevant edges
                        if combined_score > 0.4:
                            semantic_results.append({
                                "id": f"semantic_edge_{source_node}_{target_node}",
                                "type": "semantic",
                                "subtype": "edge",
                                "score": combined_score,
                                "similarity": similarity,
                                "keyword_score": keyword_score,
                                "neuroscience_boost": neuroscience_boost,
                                "data": {
                                    "source_node": u_info,
                                    "target_node": v_info,
                                    "relationship_type": rel_type,
                                    "edge_properties": edge_data,
                                    "source_node_id": source_node,
                                    "target_node_id": target_node
                                },
                                "explanation": f"Neuroscience relationship '{rel_type}' (similarity: {similarity:.3f})"
                            })

            # Sort by score and return top results
            semantic_results.sort(key=lambda x: x['score'], reverse=True)
            return semantic_results[:15]  # Return top 15 most relevant

        except Exception as e:
            print(f"❌ Semantic search error: {e}")
            return []

# Removed: _generate_semantic_guided_cypher, _template_based_semantic_guided_cypher
# STEP 3 complexity removed - now focusing on simple, effective: direct Cypher + semantic + context re-ranking

    def _extract_entity_from_node_info(self, node_info: str) -> str:
        """Extract the 'name' property value from graph node info for precise Cyper querying"""

        if not node_info:
            return ""

        # Node info format: "NodeLabel: name_property_value concept: ConceptProperty..."
        # Example: "AttentionalFiltering: top-down control concept: Attentional Filtering..."
        # We want to extract: "top-down control" (the name property value)

        try:
            # Find the pattern: "Label: name_value concept:"
            # Extract everything between ": " and " concept:"
            if ': ' in node_info and ' concept:' in node_info:
                start_marker = ': '
                end_marker = ' concept:'

                start_idx = node_info.find(start_marker)
                if start_idx != -1:
                    start_idx += len(start_marker)
                    end_idx = node_info.find(end_marker, start_idx)

                    if end_idx != -1:
                        name_value = node_info[start_idx:end_idx].strip()

                        # Clean the extracted name (remove extra whitespace, trim)
                        name_value = ' '.join(name_value.split())

                        # If it's a reasonable neuroscience term, return it
                        if len(name_value) > 1 and not name_value.lower() in ['the', 'and', 'or', 'but', 'for', 'with', 'without']:
                            return name_value

        except Exception as e:
            print(f"⚠️ Error parsing node info: {e}")

        # Fallback: try alternative parsing if the main pattern fails
        # Some nodes might have different formats
        try:
            import re
            # Alternative pattern: take text between first ": " and " concept:" or similar
            pattern = r':\s*([^:]+?)(?:\s+concept:|\s*$|\|)'
            match = re.search(pattern, node_info)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 1:
                    return candidate
        except:
            pass

        # Final fallback: return the whole cleaned node_info if all else fails
        # But remove the label part if it starts with capital letters
        words = node_info.replace(':', '').split()
        if words and words[0][0].isupper():  # Likely a label, skip it
            return ' '.join(words[1:]).strip()

        return node_info.split(':')[1].strip() if ':' in node_info else node_info

    def _calculate_keyword_score(self, text: str, keywords: Dict[str, List[str]]) -> float:
        """Calculate keyword relevance score for a text."""
        if not keywords or not text:
            return 0.0
        
        text_lower = text.lower()
        score = 0.0
        total_keywords = 0
        
        for category, words in keywords.items():
            for word in words:
                if word.lower() in text_lower:
                    score += 1.0
                    total_keywords += 1
        
        # Normalize score
        return score / max(total_keywords, 1)
    
    def _combine_results(self, cypher_results: List[Dict], semantic_results: List[Dict], 
                        keywords: Dict[str, List[str]], max_results: int) -> List[Dict]:
        """Combine and rank results from both search strategies."""
        all_results = []
        
        # Add Cypher results with weights
        for result in cypher_results:
            # Boost score based on keyword relevance
            keyword_score = self._calculate_keyword_score(str(result['data']), keywords)
            weighted_score = (result['score'] * self.cypher_weight) + (keyword_score * 0.2)
            
            all_results.append({
                **result,
                "weighted_score": weighted_score,
                "relevance_type": "structural"
            })
        
        # Add semantic results with weights
        for result in semantic_results:
            weighted_score = (result['score'] * self.semantic_weight)
            
            all_results.append({
                **result,
                "weighted_score": weighted_score,
                "relevance_type": "semantic"
            })
        
        # Remove duplicates based on content similarity
        unique_results = self._remove_duplicates(all_results)
        
        # Sort by weighted score
        unique_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Return top results
        return unique_results[:max_results]
    
    def _remove_duplicates(self, results: List[Dict], similarity_threshold: float = 0.8) -> List[Dict]:
        """Remove duplicate or highly similar results."""
        unique_results = []

        for result in results:
            # Create content signature for basic deduplication
            content_signature = self._create_content_signature(result)

            # Check for duplicates
            is_duplicate = False
            for existing_result in unique_results:
                existing_signature = self._create_content_signature(existing_result)
                if self._calculate_signature_similarity(content_signature, existing_signature) > similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)

        return unique_results

    def _is_relationship_result(self, result: Dict) -> bool:
        """Check if a result represents a relationship between two nodes."""
        if result['type'] == 'cypher':
            # Check if Cypher result has relationship data (a.name, type(r), b.name)
            data = result.get('data', {})
            return 'a.name' in data and 'type(r)' in data and 'b.name' in data
        elif result['type'] == 'semantic' and result.get('subtype') == 'edge':
            # Semantic edge results represent relationships
            return True
        return False

    def _create_relationship_key(self, result: Dict) -> Optional[str]:
        """Create a normalized key for a relationship to detect reversals."""
        if not self._is_relationship_result(result):
            return None

        if result['type'] == 'cypher':
            data = result.get('data', {})
            source = data.get('a.name', '').lower().strip()
            rel_type = data.get('type(r)', '').upper().strip()
            target = data.get('b.name', '').lower().strip()
        elif result['type'] == 'semantic' and result.get('subtype') == 'edge':
            data = result.get('data', {})
            source = data.get('source_node', '').lower().strip()
            rel_type = data.get('relationship_type', '').upper().strip()
            target = data.get('target_node', '').lower().strip()
        else:
            return None

        # Create normalized key: sort nodes alphabetically to detect reversals
        if source and target and rel_type:
            nodes = sorted([source, target])
            return f"{nodes[0]}|{rel_type}|{nodes[1]}"
        return None

    def _find_reversed_relationship(self, relationship_key: str, seen_relationships: set) -> Optional[str]:
        """Check if a reversed version of this relationship has been seen."""
        if not relationship_key:
            return None

        # relationship_key is already normalized, so any exact match is the reverse
        if relationship_key in seen_relationships:
            return relationship_key

        return None

    def _choose_better_relationship(self, result1: Dict, result2: Dict) -> Dict:
        """Choose the better relationship between two alternatives."""
        # Prefer Cypher results over semantic results
        if result1['type'] == 'cypher' and result2['type'] != 'cypher':
            return result1
        elif result2['type'] == 'cypher' and result1['type'] != 'cypher':
            return result2

        # For same types, prefer higher score
        return result1 if result1.get('score', 0) >= result2.get('score', 0) else result2

    def _is_same_relationship(self, result: Dict, existing_key: str) -> bool:
        """Check if a result represents the same relationship as an existing key."""
        result_key = self._create_relationship_key(result)
        return result_key == existing_key

    def _create_content_signature(self, result: Dict) -> str:
        """Create a content signature for deduplication."""
        if result['type'] == 'cypher':
            # For Cypher results, use the data content
            content = str(result['data'])
        else:
            # For semantic results, use the node info
            content = result['data'].get('node_info', '')
        
        # Create a simple hash-like signature
        words = content.lower().split()
        signature = "_".join(sorted(words[:5]))  # Use first 5 words as signature
        return signature
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two signatures."""
        words1 = set(sig1.split("_"))
        words2 = set(sig2.split("_"))

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _extract_node_type(self, node_info: str) -> str:
        """Extract node type from node information string."""
        # Common neuroscience node type patterns in the info strings
        type_patterns = {
            'Attention': ['Attention', 'attention'],
            'Memory': ['Memory', 'memory'],
            'CognitiveLoad': ['CognitiveLoad', 'cognitive load'],
            'ExecutiveControl': ['ExecutiveControl', 'executive control'],
            'WorkingMemory': ['WorkingMemory', 'working memory'],
            'LongTermMemory': ['LongTermMemory', 'long-term memory', 'long term memory'],
            'Neuroplasticity': ['Neuroplasticity', 'neuroplasticity'],
            'CognitiveControl': ['CognitiveControl', 'cognitive control'],
            'Consolidation': ['Consolidation', 'consolidation'],
            'EmotionalArousal': ['EmotionalArousal', 'emotional arousal'],
            'Motivation': ['Motivation', 'motivation']
        }

        node_info_lower = node_info.lower()
        for node_type, patterns in type_patterns.items():
            if any(pattern in node_info_lower for pattern in patterns):
                return node_type

        # Default to unknown if no pattern matches
        return 'Unknown'
    
    def _clean_cache(self):
        """Clean expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.query_cache.items()
            if current_time - value['timestamp'] > self.cache_ttl
        ]
        
        for key in expired_keys:
            del self.query_cache[key]
    
    def get_explanation(self, result: Dict) -> str:
        """Generate human-readable explanation for a result."""
        result_type = result.get('relevance_type', 'unknown')
        score = result.get('weighted_score', 0.0)
        
        if result_type == 'structural':
            return f"Structural match (Cypher query) with relevance score: {score:.3f}"
        elif result_type == 'semantic':
            similarity = result.get('similarity', 0.0)
            return f"Semantic match (Node2Vec) with similarity: {similarity:.3f}"
        else:
            return f"Hybrid match with relevance score: {score:.3f}"
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get statistics about the retriever performance."""
        return {
            "cache_size": len(self.query_cache),
            "node2vec_available": self.node2vec_available,
            "cypher_weight": self.cypher_weight,
            "semantic_weight": self.semantic_weight,
            "cache_ttl": self.cache_ttl
        }

# Global instance
hybrid_retriever = HybridRetriever()
