#!/usr/bin/env python3
"""
graph_retriever.py - Hybrid Graph Retriever for Educational Knowledge Graph
Combines graph traversal precision with semantic search breadth for comprehensive context retrieval
"""

import asyncio
import logging
import re
import os
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from neo4j import GraphDatabase, Driver
from collections import defaultdict
import time
from functools import lru_cache
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class RetrievedContext:
    """Structured output for retrieved educational context"""
    nodes: List[Dict[str, Any]]
    triples: List[Tuple[str, str, str]]  # (source_name, rel_type, target_name)
    facets: Dict[str, Dict[str, int]]  # {label_counts, rel_counts}
    metadata: Dict[str, Any]  # {graph_count, semantic_count, timings, limits_applied}

class HybridGraphRetriever:
    """Hybrid retriever combining graph traversal with semantic search"""
    
    def __init__(self, neo4j_driver: Driver, use_vectors: bool = False, config: Optional[Dict] = None):
        self.neo4j_driver = neo4j_driver
        self.use_vectors = use_vectors
        
        # Configuration with defaults
        self.config = config or {}
        self.max_nodes = self.config.get('max_nodes', 15)
        self.max_edges = self.config.get('max_edges', 30)
        self.expand_neighbors = self.config.get('expand_neighbors', True)
        self.neighbor_depth = self.config.get('neighbor_depth', 1)
        
        # Node2Vec integration
        self.node2vec_model = None
        self.node_embeddings = None
        self.node_index = None
        self.reverse_index = None
        self.node2vec_loaded = False
        
        # Load Node2Vec model if vectors are enabled
        if self.use_vectors:
            self._load_node2vec_model()
        
        # Neuroscience domain boosts (higher = more relevant)
        self.domain_boosts = {
            # Core motivation concepts
            'MotivationalModulation': 2.5,
            'IntrinsicMotivation': 2.5,
            'ExtrinsicMotivation': 2.5,
            
            # Mindset concepts
            'Mindset': 2.3,
            'GrowthMindset': 2.3,
            'FixedMindset': 2.3,
            
            # Stress and emotions
            'PositiveStressEustress': 2.0,
            'NegativeStressDistress': 2.0,
            'PositiveEmotions': 2.0,
            'NegativeEmotions': 2.0,
            'Emotions': 2.0,
            
            # Cognitive functions
            'ExecutiveFunctions': 2.2,
            'Metacognition': 2.2,
            'CriticalThinking': 2.2,
            'Attention': 2.1,
            'WorkingMemory': 2.1,
            'LongTermMemory': 2.1,
            'Creativity': 1.9,
            
            # Learning processes
            'LearningDevelopment': 1.8,
            'LearningOutcomes': 1.8,
            'MemoryEncoding': 1.7,
            'Consolidation': 1.7,
            'Neuroplasticity': 1.8,
            
            # Processing and control
            'CognitiveLoad': 1.6,
            'CognitiveControl': 1.6,
            'AttentionalControl': 1.6,
            'PrefrontalCortexactivation': 1.5
        }
        
        # Schema typo corrections (none needed for neuroscience graph)
        self.schema_corrections = {}
        
        # Whitelist for neighbor expansion (focus on neuroscience relevance)
        self.expansion_labels = {
            # Motivation
            'MotivationalModulation', 'IntrinsicMotivation', 'ExtrinsicMotivation',
            
            # Mindset
            'Mindset', 'GrowthMindset', 'FixedMindset',
            
            # Emotions and Stress
            'PositiveStressEustress', 'NegativeStressDistress',
            'PositiveEmotions', 'NegativeEmotions', 'Emotions',
            
            # Cognitive Functions
            'ExecutiveFunctions', 'Metacognition', 'CriticalThinking',
            'Attention', 'WorkingMemory', 'LongTermMemory', 'Memory',
            'Creativity', 'executivecontrol',
            
            # Learning and Development
            'LearningDevelopment', 'LearningOutcomes', 'Neuroplasticity',
            'MemoryEncoding', 'Memoryencoding', 'Consolidation',
            
            # Processing
            'CognitiveLoad', 'Cognitiveload', 'CognitiveControl', 'Cognitivecontrol',
            'AttentionalControl', 'PrefrontalCortexactivation', 'Prefrontalcortexactivation',
            
            # Other relevant
            'AffectiveProcesses', 'SocialCognition', 'Communication',
            'ProblemSolving', 'Reasoning', 'BrainAdaptability'
        }
    
    def _load_node2vec_model(self, model_path: str = "models/educational_node2vec"):
        """Load pre-trained Node2Vec model and embeddings"""
        try:
            # Load Node2Vec model
            model_file = f"{model_path}_model.pkl"
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.node2vec_model = pickle.load(f)
                logger.info("Node2Vec model loaded successfully")
            else:
                logger.warning(f"Node2Vec model file not found: {model_file}")
                return False
            
            # Load embeddings and indices
            embeddings_file = f"{model_path}_embeddings.npz"
            if os.path.exists(embeddings_file):
                data = np.load(embeddings_file, allow_pickle=True)
                self.node_embeddings = data['embeddings']
                self.node_index = data['node_index'].item()
                self.reverse_index = data['reverse_index'].item()
                self.node2vec_loaded = True
                logger.info(f"Node2Vec embeddings loaded: {len(self.node_embeddings)} nodes")
                return True
            else:
                logger.warning(f"Node2Vec embeddings file not found: {embeddings_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load Node2Vec model: {e}")
            self.node2vec_loaded = False
            return False
    
    async def retrieve(self, query: str, cypher_result: Dict) -> RetrievedContext:
        """Main retrieval method combining graph and semantic search"""
        start_time = time.time()
        
        try:
            # Phase 1: Graph traversal (precision)
            graph_start = time.time()
            graph_nodes = await self._graph_traversal(cypher_result)
            graph_time = time.time() - graph_start
            
            # Phase 2: Semantic search (breadth, optional)
            semantic_nodes = []
            semantic_time = 0
            if self.use_vectors:
                semantic_start = time.time()
                semantic_nodes = await self._semantic_search(query, graph_nodes)
                semantic_time = time.time() - semantic_start
            
            # Phase 3: Fusion and ranking
            fusion_start = time.time()
            ranked_nodes, triples = self._fuse_results(graph_nodes, semantic_nodes)
            fusion_time = time.time() - fusion_start
            
            # Phase 4: Build final context
            context_start = time.time()
            facets = self._build_facets(ranked_nodes, triples)
            context_time = time.time() - context_start
            
            total_time = time.time() - start_time
            
            metadata = {
                'graph_count': len(graph_nodes),
                'semantic_count': len(semantic_nodes),
                'total_nodes': len(ranked_nodes),
                'total_triples': len(triples),
                'timings': {
                    'graph_traversal': graph_time,
                    'semantic_search': semantic_time,
                    'fusion': fusion_time,
                    'context_building': context_time,
                    'total': total_time
                },
                'limits_applied': {
                    'max_nodes': self.max_nodes,
                    'max_edges': self.max_edges,
                    'use_vectors': self.use_vectors
                }
            }
            
            return RetrievedContext(
                nodes=ranked_nodes[:self.max_nodes],
                triples=triples[:self.max_edges],
                facets=facets,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Fallback to graph-only results
            return await self._fallback_retrieval(cypher_result, str(e))
    
    async def _graph_traversal(self, cypher_result: Dict) -> List[Dict]:
        """Execute Cypher query and retrieve curriculum nodes with neighbor expansion"""
        try:
            cypher_query = cypher_result.get('cypher_query', '')
            if not cypher_query:
                logger.warning("No Cypher query provided for graph traversal")
                return []

            corrected_query = self._apply_schema_corrections(cypher_query)
            alias_labels = self._extract_alias_labels(corrected_query)

            with self.neo4j_driver.session() as session:
                # 1) Run the original query
                result = session.run(corrected_query)
                records = list(result)

                # 2) If it returned nodes -> keep them; if it returned scalars -> reconstruct
                initial_nodes = []
                if records:
                    # detect if rows look like scalars (keys containing '.')
                    sample_keys = list(dict(records[0]).keys())
                    logger.debug(f"First row keys: {sample_keys}")
                    
                    if any("." in k for k in sample_keys):
                        initial_nodes = self._records_to_nodes(records, alias_labels)
                    else:
                        # rows might already contain nodes; normalize each 'n'/'m'/etc. field
                        for rec in records:
                            row = dict(rec)
                            for v in row.values():
                                # If it's a Node (py2neo/neo4j driver), cast to dict of properties
                                try:
                                    props = dict(v)
                                    initial_nodes.append(self._normalize_node(props))
                                except Exception:
                                    pass

                if not initial_nodes:
                    logger.info("No initial nodes found from Cypher query")
                    return []

                # 3) Expand neighbors if enabled
                if self.expand_neighbors:
                    expanded_nodes = await self._expand_neighbors(initial_nodes, session)
                    return expanded_nodes
                else:
                    return initial_nodes

        except Exception as e:
            logger.error(f"Graph traversal failed: {e}")
            return []
    
    def _apply_schema_corrections(self, cypher_query: str) -> str:
        """Apply schema typo corrections to Cypher query"""
        corrected_query = cypher_query
        for typo, correct in self.schema_corrections.items():
            corrected_query = corrected_query.replace(typo, correct)
        return corrected_query
    
    def _extract_alias_labels(self, cypher_query: str) -> dict:
        """
        From patterns like (m:PedagogicalMethodology) or (s:StudentWithSpecialNeeds),
        build {'m': 'PedagogicalMethodology', 's': 'StudentWithSpecialNeeds'}.
        """
        alias_labels = {}
        for alias, label in re.findall(r'\(\s*([A-Za-z_]\w*)\s*:\s*([A-Za-z][A-Za-z0-9_]*)\s*\)', cypher_query):
            alias_labels[alias] = label
        return alias_labels

    def _records_to_nodes(self, records, alias_labels: dict) -> list:
        """
        Turn scalar projections like {'m.name': 'Cooperative Learning', 'm.category': 'Pedagogical Methodology'}
        into normalized node dicts the retriever understands.
        """
        nodes = []
        for rec in records:
            # Neo4j Record -> plain dict
            row = dict(rec)
            # group fields by alias (m, s, lr, etc.)
            buckets = {}
            for key, val in row.items():
                if "." in key:
                    alias, prop = key.split(".", 1)
                    buckets.setdefault(alias, {})[prop] = val

            # build node per alias that has a 'name'
            for alias, props in buckets.items():
                if "name" in props:
                    label = alias_labels.get(alias)
                    node = {
                        "id": f"{label}:{props['name']}" if label else props["name"],
                        "name": props["name"],
                        "category": props.get("category", ""),
                        "labels": [label] if label else [],
                        "description": props.get("description", ""),
                        "rel_type": "",
                        "source_node": {}
                    }
                    nodes.append(node)

        return nodes
    
    async def _expand_neighbors(self, nodes: List[Dict], session) -> List[Dict]:
        """Expand nodes with their educational neighbors (structural + vector-based)"""
        expanded_nodes = []
        seen_node_ids = set()
        
        for node in nodes:
            # Add the original node
            node_id = node.get('id') or node.get('name')
            if node_id and node_id not in seen_node_ids:
                expanded_nodes.append(self._normalize_node(node))
                seen_node_ids.add(node_id)
            
            # Get structural neighbors (direct relationships)
            structural_neighbors = self._get_educational_neighbors(node, session)
            for neighbor in structural_neighbors:
                neighbor_id = neighbor.get('id') or neighbor.get('name')
                if neighbor_id and neighbor_id not in seen_node_ids:
                    neighbor['source'] = 'structural'
                    expanded_nodes.append(self._normalize_node(neighbor))
                    seen_node_ids.add(neighbor_id)
            
            # Get vector-based neighbors (semantic similarity)
            if self.node2vec_loaded and self.use_vectors:
                vector_neighbors = await self._get_vector_neighbors(node, seen_node_ids)
                for neighbor in vector_neighbors:
                    neighbor_id = neighbor.get('id') or neighbor.get('name')
                    if neighbor_id and neighbor_id not in seen_node_ids:
                        neighbor['source'] = 'vector'
                        expanded_nodes.append(self._normalize_node(neighbor))
                        seen_node_ids.add(neighbor_id)
        
        return expanded_nodes
    
    def _get_educational_neighbors(self, node: Dict, session) -> List[Dict]:
        """Get educationally relevant neighbors of a node"""
        try:
            node_labels = node.get('labels', [])
            node_name = node.get('name', '')
            if not node_labels or not node_name:
                return []

            main_label = node_labels[0]

            # Prefer putting the label in MATCH (can't parameterize labels)
            neighbor_query = f"""
            MATCH (source:{main_label} {{name: $node_name}})-[r]-(n)
            WHERE any(l IN labels(n) WHERE l IN $relevant_labels)
            RETURN DISTINCT n, type(r) AS rel_type, source AS source_node
            LIMIT $limit
            """

            result = session.run(
                neighbor_query,
                node_name=node_name,
                relevant_labels=list(self.expansion_labels),
                limit=5
            )

            neighbors = []
            for record in result:
                # n and source_node are Neo4j Node objects — capture labels explicitly
                n_node = record['n']
                s_node = record['source_node']

                neighbor = dict(n_node)
                neighbor['labels'] = list(getattr(n_node, 'labels', []))
                neighbor['rel_type'] = record['rel_type']

                src = dict(s_node)
                src['labels'] = list(getattr(s_node, 'labels', []))
                neighbor['source_node'] = src

                # give a stable id (helps dedup + ranking)
                label_for_id = neighbor['labels'][0] if neighbor.get('labels') else ''
                neighbor['id'] = f"{label_for_id}:{neighbor.get('name','')}"
                neighbors.append(neighbor)

            return neighbors

        except Exception as e:
            logger.error(f"Error getting neighbors for {node.get('name', 'unknown')}: {e}")
            return []
    
    async def _get_vector_neighbors(self, node: Dict, seen_node_ids: set) -> List[Dict]:
        """Get vector-based neighbors using Node2Vec similarity"""
        try:
            node_name = node.get('name', '')
            if not node_name or not self.node2vec_loaded or self.node_embeddings is None:
                return []
            
            # Find similar concepts using Node2Vec
            similar_concepts = self._find_similar_concepts(node_name, top_k=8)
            
            vector_neighbors = []
            for similar_name, similarity_score in similar_concepts:
                if similar_name in seen_node_ids:
                    continue
                
                # Get node details from Neo4j
                node_details = await self._get_node_details(similar_name)
                if node_details:
                    # Only include educationally relevant nodes
                    labels = node_details.get('labels', [])
                    if any(label in self.expansion_labels for label in labels):
                        node_details['vector_similarity'] = similarity_score
                        node_details['rel_type'] = 'VECTOR_SIMILAR'
                        node_details['source_node'] = node
                        vector_neighbors.append(node_details)
            
            return vector_neighbors
            
        except Exception as e:
            logger.error(f"Error getting vector neighbors for {node.get('name', 'unknown')}: {e}")
            return []
    
    def _normalize_node(self, node: Dict) -> Dict:
        """Normalize node data for consistent format"""
        return {
            'id': node.get('id', ''),
            'name': node.get('name', ''),
            'category': node.get('category', ''),
            'labels': node.get('labels', []),
            'description': node.get('description', ''),
            'rel_type': node.get('rel_type', ''),
            'source_node': node.get('source_node', {})
        }
    
    async def _semantic_search(self, query: str, existing_nodes: List[Dict]) -> List[Dict]:
        """Enhanced semantic search using Node2Vec embeddings"""
        try:
            # Get existing node names to avoid duplicates
            existing_names = {node.get('name', '') for node in existing_nodes}
            
            semantic_nodes = []
            
            if self.node2vec_loaded and self.use_vectors:
                # Use Node2Vec for semantic similarity
                semantic_nodes = await self._node2vec_semantic_search(query, existing_names)
            else:
                # Fallback to keyword-based search
                semantic_nodes = await self._keyword_semantic_search(query, existing_names)
            
            return semantic_nodes
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _node2vec_semantic_search(self, query: str, existing_names: set) -> List[Dict]:
        """Node2Vec-based semantic search for educational concepts"""
        try:
            # Extract key concepts from query
            query_concepts = self._extract_query_concepts(query)
            
            semantic_nodes = []
            seen_concepts = set()
            
            for concept in query_concepts:
                if concept in seen_concepts:
                    continue
                seen_concepts.add(concept)
                
                # Find similar concepts using Node2Vec
                similar_concepts = self._find_similar_concepts(concept, top_k=15)
                
                for similar_name, similarity_score in similar_concepts:
                    if similar_name in existing_names:
                        continue
                    
                    # Get node details from Neo4j
                    node_details = await self._get_node_details(similar_name)
                    if node_details:
                        node_details['semantic_score'] = similarity_score
                        node_details['query_concept'] = concept
                        semantic_nodes.append(self._normalize_node(node_details))
            
            # Sort by semantic score and return top results
            semantic_nodes.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
            return semantic_nodes[:20]  # Limit to top 20
            
        except Exception as e:
            logger.error(f"Node2Vec semantic search failed: {e}")
            return []
    
    async def _keyword_semantic_search(self, query: str, existing_names: set) -> List[Dict]:
        """Fallback keyword-based semantic search"""
        try:
            semantic_nodes = []
            
            with self.neo4j_driver.session() as session:
                # Search in node descriptions and names
                semantic_query = """
                MATCH (n)
                WHERE n.description IS NOT NULL OR n.name IS NOT NULL
                AND NOT n.name IN $existing_names
                AND any(label IN labels(n) WHERE label IN $relevant_labels)
                RETURN n, 
                       CASE 
                         WHEN toLower(n.description) CONTAINS toLower($query_text) THEN 0.8
                         WHEN toLower(n.name) CONTAINS toLower($query_text) THEN 0.6
                         ELSE 0.2
                       END as relevance_score
                ORDER BY relevance_score DESC
                LIMIT 10
                """
                
                result = session.run(
                    semantic_query,
                    query_text=query,
                    existing_names=list(existing_names),
                    relevant_labels=list(self.expansion_labels)
                )
                
                for record in result:
                    node = dict(record['n'])
                    node['semantic_score'] = record['relevance_score']
                    semantic_nodes.append(self._normalize_node(node))
            
            return semantic_nodes
            
        except Exception as e:
            logger.error(f"Keyword semantic search failed: {e}")
            return []
    
    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract educational concepts from query for Node2Vec similarity"""
        # Simple concept extraction - can be enhanced with NLP
        concepts = []
        
        # Common educational terms to look for
        educational_terms = [
            'adhd', 'autism', 'blind', 'deaf', 'cognitive', 'physical', 'disability',
            'cooperative', 'learning', 'methodology', 'pedagogical', 'strategy',
            'motivation', 'attention', 'visual', 'hearing', 'impairment'
        ]
        
        query_lower = query.lower()
        for term in educational_terms:
            if term in query_lower:
                concepts.append(term)
        
        # If no specific terms found, use the whole query
        if not concepts:
            concepts = [query]
        
        return concepts
    
    def _find_similar_concepts(self, concept: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar concepts using Node2Vec embeddings"""
        if not self.node2vec_loaded or self.node_embeddings is None:
            return []
        
        try:
            # Try to find exact match first
            if concept in self.node_index:
                concept_idx = self.node_index[concept]
                concept_embedding = self.node_embeddings[concept_idx].reshape(1, -1)
                
                # Calculate cosine similarities
                similarities = cosine_similarity(concept_embedding, self.node_embeddings)[0]
                
                # Get top-k similar concepts
                similar_indices = np.argsort(similarities)[::-1][1:top_k+1]  # Exclude self
                
                results = []
                for idx in similar_indices:
                    similar_name = self.reverse_index[idx]
                    similarity_score = similarities[idx]
                    results.append((similar_name, similarity_score))
                
                return results
            
            # If no exact match, try fuzzy matching
            return self._fuzzy_concept_search(concept, top_k)
            
        except Exception as e:
            logger.error(f"Error finding similar concepts for '{concept}': {e}")
            return []
    
    def _fuzzy_concept_search(self, concept: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Fuzzy search for concepts when exact match not found"""
        try:
            # Find concepts that contain the search term
            matching_concepts = []
            for node_name in self.node_index.keys():
                if concept.lower() in node_name.lower():
                    matching_concepts.append(node_name)
            
            if not matching_concepts:
                return []
            
            # Get similarities for all matching concepts
            all_similarities = []
            for match_concept in matching_concepts:
                if match_concept in self.node_index:
                    concept_idx = self.node_index[match_concept]
                    concept_embedding = self.node_embeddings[concept_idx].reshape(1, -1)
                    similarities = cosine_similarity(concept_embedding, self.node_embeddings)[0]
                    
                    # Get top similarities for this concept
                    similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
                    for idx in similar_indices:
                        similar_name = self.reverse_index[idx]
                        similarity_score = similarities[idx]
                        all_similarities.append((similar_name, similarity_score))
            
            # Sort and deduplicate
            all_similarities.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            results = []
            for name, score in all_similarities:
                if name not in seen:
                    results.append((name, score))
                    seen.add(name)
                    if len(results) >= top_k:
                        break
            
            return results
            
        except Exception as e:
            logger.error(f"Fuzzy concept search failed: {e}")
            return []
    
    async def _get_node_details(self, node_name: str) -> Optional[Dict]:
        """Get full node details from Neo4j"""
        try:
            with self.neo4j_driver.session() as session:
                query = """
                MATCH (n {name: $node_name})
                RETURN n, labels(n) as labels
                LIMIT 1
                """
                
                result = session.run(query, node_name=node_name)
                record = result.single()
                
                if record:
                    node = dict(record['n'])
                    node['labels'] = record['labels']
                    return node
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting node details for '{node_name}': {e}")
            return None
    
    def _fuse_results(self, graph_nodes: List[Dict], semantic_nodes: List[Dict]) -> Tuple[List[Dict], List[Tuple[str, str, str]]]:
        """Fuse and rank results from multiple sources (graph, structural, vector, semantic)"""
        # Combine all nodes
        all_nodes = []
        
        # Add graph nodes (already have source assigned)
        for node in graph_nodes:
            source = node.get('source', 'graph')
            node['rank_score'] = self._calculate_rank_score(node, source=source)
            all_nodes.append(node)
        
        # Add semantic nodes with lower priority
        for node in semantic_nodes:
            node['source'] = 'semantic'
            node['rank_score'] = self._calculate_rank_score(node, source='semantic')
            all_nodes.append(node)
        
        # Deduplicate by node name/id, keeping highest scoring version
        unique_nodes = {}
        for node in all_nodes:
            node_id = node.get('id') or node.get('name', '')
            if node_id:
                if node_id not in unique_nodes or node['rank_score'] > unique_nodes[node_id]['rank_score']:
                    unique_nodes[node_id] = node
        
        # Sort by rank score
        ranked_nodes = sorted(unique_nodes.values(), key=lambda x: x['rank_score'], reverse=True)
        
        # Extract triples from relationships
        triples = self._extract_triples(ranked_nodes)
        
        return ranked_nodes, triples
    
    def _calculate_rank_score(self, node: Dict, source: str = 'graph') -> float:
        """Calculate ranking score for a node with Node2Vec enhancements"""
        # Base score by source
        if source == 'graph':
            base_score = 1.0
        elif source == 'structural':
            base_score = 0.8
        elif source == 'vector':
            base_score = 0.6
        else:  # semantic
            base_score = 0.5
        
        # Apply domain boosts
        labels = node.get('labels', [])
        domain_boost = max([self.domain_boosts.get(l, 1.0) for l in labels], default=1.0)
        
        # Apply semantic score if available
        semantic_score = node.get('semantic_score', 1.0)
        
        # Apply vector similarity boost if available
        vector_boost = node.get('vector_similarity', 1.0)
        
        # Calculate final score
        final_score = base_score * domain_boost * semantic_score * vector_boost
        
        return final_score
    
    def _extract_triples(self, nodes: List[Dict]) -> List[Tuple[str, str, str]]:
        """Extract relationship triples from nodes"""
        triples = []
        
        for node in nodes:
            # Extract triples from neighbor relationships
            if 'rel_type' in node and 'source_node' in node:
                source_name = node['source_node'].get('name', '')
                rel_type = node['rel_type']
                target_name = node.get('name', '')
                
                if source_name and rel_type and target_name:
                    triples.append((source_name, rel_type, target_name))
        
        return triples
    
    def _build_facets(self, nodes: List[Dict], triples: List[Tuple[str, str, str]]) -> Dict[str, Dict[str, int]]:
        """Build facet summaries for the retrieved context"""
        # Count by label
        label_counts = defaultdict(int)
        for node in nodes:
            labels = node.get('labels', [])
            for label in labels:
                label_counts[label] += 1
        
        # Count by relationship type
        rel_counts = defaultdict(int)
        for source, rel_type, target in triples:
            rel_counts[rel_type] += 1
        
        return {
            'label_counts': dict(label_counts),
            'rel_counts': dict(rel_counts)
        }
    
    async def _fallback_retrieval(self, cypher_result: Dict, error: str) -> RetrievedContext:
        """Fallback retrieval when main retrieval fails"""
        logger.warning(f"Using fallback retrieval due to error: {error}")
        
        # Return basic context from cypher result
        nodes = cypher_result.get('results', [])
        normalized_nodes = [self._normalize_node(node) for node in nodes]
        
        return RetrievedContext(
            nodes=normalized_nodes[:self.max_nodes],
            triples=[],
            facets={'label_counts': {}, 'rel_counts': {}},
            metadata={
                'graph_count': len(nodes),
                'semantic_count': 0,
                'total_nodes': len(normalized_nodes),
                'total_triples': 0,
                'error': error,
                'fallback_used': True,
                'timings': {'total': 0}
            }
        )

# Integration helper for your existing pipeline
class EnhancedMultilingualText2Cypher:
    """Enhanced version that includes hybrid retrieval"""
    
    def __init__(self, use_vectors: bool = False, config: Optional[Dict] = None):
        from multilingual_text2cypher import MultilingualText2Cypher
        from config import config as app_config
        
        self.text2cypher = MultilingualText2Cypher()
        self.graph_retriever = HybridGraphRetriever(
            neo4j_driver=self.text2cypher.pipeline.converter.schema_extractor.driver,
            use_vectors=use_vectors,
            config=config
        )
    
    async def process_query_with_retrieval(self, query: str) -> Dict:
        """Process query with full hybrid retrieval pipeline"""
        # Step 1: Text2Cypher (existing functionality)
        cypher_result = self.text2cypher.process_query(query, execute=True)
        
        # Step 2: Hybrid Retrieval (new functionality)
        retrieval_result = await self.graph_retriever.retrieve(query, cypher_result)
        
        # Step 3: Build Educational Context (uses real retrieval)
        try:
            from context_builder import EducationalContextBuilder
            context_builder = EducationalContextBuilder()
            
            # Convert retrieval_result to dict format expected by context_builder
            # Convert triples (tuples) to dicts with keys: relationship, source, target
            triples_as_dicts = [
                {'relationship': rel_type, 'source': source, 'target': target}
                for source, rel_type, target in retrieval_result.triples
            ]
            
            retrieval_dict = {
                'nodes': retrieval_result.nodes,
                'triples': triples_as_dicts,
                'metadata': retrieval_result.metadata
            }
            
            educational_context_obj = await context_builder.build_context(
                retrieval_dict,
                query,
                {'educational_context': 'special_needs'}  # can be enriched from multilingual processor
            )
            
            # Convert to dict for display/serialization
            educational_context_dict = asdict(educational_context_obj)
            
            # Clean up the confidence fields - convert from "ConfidenceLevel.HIGH" to "HIGH"
            if 'confidence_assessment' in educational_context_dict:
                conf = str(educational_context_dict['confidence_assessment'])
                educational_context_dict['confidence_assessment'] = conf.replace('ConfidenceLevel.', '')
            
            for methodology in educational_context_dict.get('primary_methodologies', []):
                if 'confidence' in methodology and methodology['confidence']:
                    conf = str(methodology['confidence'])
                    methodology['confidence'] = conf.replace('ConfidenceLevel.', '')
            
            for methodology in educational_context_dict.get('supporting_methodologies', []):
                if 'confidence' in methodology and methodology['confidence']:
                    conf = str(methodology['confidence'])
                    methodology['confidence'] = conf.replace('ConfidenceLevel.', '')
            
        except Exception as e:
            logger.error(f"Context building failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            educational_context_obj = None
            educational_context_dict = {}

        # Step 4: Generate LLM Response (add this missing step!)
        llm_response_result = {}
        if educational_context_obj:
            try:
                from llm_chain import EducationalResponseGenerator
                from config import config

                response_generator = EducationalResponseGenerator(
                    openai_api_key=config.openai.api_key,
                    language="italian",
                    temperature=0.7
                )

                llm_response_result = await response_generator.generate_response(
                    educational_context_obj,
                    query
                )

                # Close the generator
                # Note: EducationalResponseGenerator doesn't have a close method

            except Exception as e:
                logger.error(f"LLM response generation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

                # Generate fallback response
                llm_response_result = {
                    'response': f"""Mi dispiace, ho riscontrato un problema nella generazione della risposta completa.

Tuttavia, ecco un riassunto delle metodologie neuroscientifiche identificate per la tua domanda "{query}":

**Metodologie Raccomandate:**
{chr(10).join([f"- {m['name']}" for m in educational_context_dict.get('primary_methodologies', [])])}

**Confidenza:** {educational_context_dict.get('confidence_assessment', 'Bassa')}

**Prossimi Passi:**
- Consulta risorse specializzate in neuroscienze educative
- Considera formazione per insegnati su approccio neuroscientifico
- Collabora con esperti di didattica inclusiva

Se hai bisogno di ulteriore assistenza, non esitare a chiedere.""",
                    'confidence': 'LOW',
                    'metadata': {'fallback': True, 'error': str(e)}
                }
        else:
            # No context available - major fallback
            llm_response_result = {
                'response': f"""Mi dispiace, non sono riuscito a identificare metodologie specifiche per la tua domanda.

Tuttavia, posso suggerire un approccio generale basato su neuroscienze:

**Raccomandazioni Generiche:**
- Crea un ambiente di apprendimento positivo che favorisca la motivazione intrinseca
- Adatta le metodologie alle caratteristiche individuali degli studenti
- Utilizza un approccio multisensoriale per migliorare l'apprendimento

**Per supporto più specifico:** consulta specialisti di pedagogia o psicologia educativa.

Riprova con una domanda più specifica sui bisogni educativi degli studenti.""",
                'confidence': 'VERY_LOW',
                'metadata': {'fallback': True, 'reason': 'no_context'}
            }

        return {
            'original_query': query,
            'cypher_result': cypher_result,
            'retrieval_result': retrieval_result,
            'combined_context': self._build_context(retrieval_result),
            'educational_context': educational_context_dict,  # Dict for display
            'educational_context_obj': educational_context_obj,  # Object for LLM chain
            'llm_response': llm_response_result  # Add the missing LLM response!
        }
    
    def _build_context(self, retrieval_result: RetrievedContext) -> str:
        """Build human-readable context from retrieval results"""
        context_parts = []
        
        # Add node summaries
        if retrieval_result.nodes:
            context_parts.append("## Educational Context")
            for i, node in enumerate(retrieval_result.nodes[:10], 1):
                context_parts.append(f"{i}. **{node.get('name', 'Unknown')}** ({node.get('category', 'Unknown')})")
        
        # Add relationship summaries
        if retrieval_result.triples:
            context_parts.append("\n## Key Relationships")
            for source, rel_type, target in retrieval_result.triples[:10]:
                context_parts.append(f"- {source} → {rel_type} → {target}")
        
        # Add facet summary
        if retrieval_result.facets:
            context_parts.append("\n## Summary")
            label_counts = retrieval_result.facets.get('label_counts', {})
            if label_counts:
                context_parts.append("**Node Types:** " + ", ".join([f"{label} ({count})" for label, count in label_counts.items()]))
        
        return "\n".join(context_parts)
    
    def close(self):
        """Close all connections"""
        self.text2cypher.close()

# Testing function
async def test_hybrid_retriever():
    """Test the hybrid retriever with sample queries"""
    from config import config
    
    # Initialize enhanced processor with Node2Vec
    processor = EnhancedMultilingualText2Cypher(use_vectors=True)
    
    # Test queries
    test_queries = [
        "Ci sono strategie per i ragazzi ipovedenti?",
        "Il mio studente ha l'ADHD, cosa posso fare?",
        "Metodologie per studenti senza motivazione personale?",
        "Come aiutare studenti con disturbi dello spettro autistico?",
        "Esistono tecniche per includere studenti con disabilità fisica?"
    ]
    
    try:
        for query in test_queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print('='*80)
            
            result = await processor.process_query_with_retrieval(query)
            
            # Show Cypher result
            cypher_result = result['cypher_result']
            print(f"\n🔍 Generated Cypher: {cypher_result['cypher_query']}")
            print(f"✅ Valid: {cypher_result['metadata'].get('is_valid', False)}")
            
            # Show retrieval result
            retrieval_result = result['retrieval_result']
            print(f"\n📊 Retrieved {len(retrieval_result.nodes)} nodes")
            print(f"🔗 Retrieved {len(retrieval_result.triples)} relationships")
            
            # Show metadata
            metadata = retrieval_result.metadata
            print(f"\n⏱️  Timings: {metadata.get('timings', {})}")
            print(f"📈 Counts: Graph={metadata.get('graph_count', 0)}, Semantic={metadata.get('semantic_count', 0)}")
            
            # Show sample nodes
            if retrieval_result.nodes:
                print(f"\n📝 Sample Nodes:")
                for i, node in enumerate(retrieval_result.nodes[:3], 1):
                    print(f"  {i}. {node.get('name', 'Unknown')} ({node.get('category', 'Unknown')})")
            
            # Show sample triples
            if retrieval_result.triples:
                print(f"\n🔗 Sample Relationships:")
                for i, (source, rel, target) in enumerate(retrieval_result.triples[:3], 1):
                    print(f"  {i}. {source} → {rel} → {target}")
    
    finally:
        processor.close()

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_hybrid_retriever())
