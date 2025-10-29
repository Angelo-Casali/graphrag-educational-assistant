"""
Enhanced Context Builder for Hybrid Retriever Results
Integrates both Cypher and semantic search results for LLM consumption.
"""

from typing import Dict, List, Any, Optional
import json

def build_context(records: list) -> str:
    """
    Legacy context builder for backward compatibility.
    """
    if not records:
        return "No relevant information found in the knowledge graph for your query. This could mean:\n1. The information doesn't exist in the database\n2. The query might need to be rephrased\n3. The generated Cypher query might need adjustment"
    
    context = "Relevant information from the knowledge graph:\n\n"
    for i, record in enumerate(records, 1):
        context += f"Record {i}:\n"
        # Format the record in a more readable way
        for key, value in record.items():
            context += f"  {key}: {value}\n"
        context += "\n"
        
    return context

def build_hybrid_context(hybrid_result: Dict[str, Any]) -> str:
    """
    Build context from hybrid retriever results for LLM consumption with relevance filtering.

    Args:
        hybrid_result: Result from hybrid_retriever.process_query()

    Returns:
        Formatted context string suitable for LLM input
    """
    if not hybrid_result or hybrid_result.get('total_results', 0) == 0:
        return "No relevant information found in the knowledge graph for your query. This could mean:\n1. The information doesn't exist in the database\n2. The query might need to be rephrased\n3. The search strategy might need adjustment"

    context_parts = []

    # Add query metadata
    context_parts.append(f"📚 **Query Analysis**")
    context_parts.append(f"- Original Query: {hybrid_result['original_query']}")
    context_parts.append(f"- Detected Language: {hybrid_result['detected_language']}")
    context_parts.append(f"- Total Results Found: {hybrid_result['total_results']}")
    context_parts.append(f"- Processing Time: {hybrid_result['processing_time']}")
    context_parts.append("")

    # Add extracted keywords
    if hybrid_result.get('keywords'):
        context_parts.append(f"🔑 **Extracted Keywords**")
        for category, words in hybrid_result['keywords'].items():
            if words:
                context_parts.append(f"- {category.title()}: {', '.join(words)}")
        context_parts.append("")

    # Add generated Cypher query
    if hybrid_result.get('cypher_query'):
        context_parts.append(f"🔧 **Generated Cypher Query**")
        context_parts.append(f"```cypher\n{hybrid_result['cypher_query']}\n```")
        context_parts.append("")

    # Filter and prioritize results
    filtered_results = _filter_relevant_results(hybrid_result.get('hybrid_results', []))

    # Group results by type for better organization
    structural_results = []
    semantic_node_results = []
    semantic_edge_results = []

    for item in filtered_results:
        relevance_type = item.get('relevance_type', 'unknown')
        if relevance_type == 'structural':
            structural_results.append(item)
        elif relevance_type == 'semantic':
            subtype = item.get('subtype', 'node')
            if subtype == 'node':
                semantic_node_results.append(item)
            else:
                semantic_edge_results.append(item)

    # Only include sections with relevant results
    has_content = False

    # Add structural results (Cypher-based) - highest priority
    if structural_results:
        context_parts.append(f"**🏗️ Structural Search Results (Direct Database Matches)**")
        context_parts.append("")
        for i, item in enumerate(structural_results[:5], 1):  # Limit to top 5
            score = item.get('weighted_score', 0.0)
            context_parts.append(f"**Result {i} (Relevance: {score:.3f})**")

            data = item.get('data', {})
            if isinstance(data, dict):
                for key, value in data.items():
                    if key != 'id' and value:
                        context_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
            else:
                context_parts.append(f"- Data: {str(data)[:200]}...")

            context_parts.append("")
        has_content = True

    # Add semantic edge results - high priority (relationships)
    if semantic_edge_results:
        context_parts.append(f"**🔗 Semantic Relationship Results (Educational Connections)**")
        context_parts.append("")
        for i, item in enumerate(semantic_edge_results[:3], 1):  # Limit to top 3
            score = item.get('weighted_score', 0.0)
            similarity = item.get('similarity', 0.0)

            context_parts.append(f"**Result {i} (Relevance: {score:.3f})**")
            context_parts.append(f"- Similarity Score: {similarity:.3f}")

            data = item.get('data', {})
            source = data.get('source_node', 'N/A')
            target = data.get('target_node', 'N/A')
            rel_type = data.get('relationship_type', 'N/A')

            context_parts.append(f"- Relationship: {source[:50]} --[{rel_type}]--> {target[:50]}")

            # Add edge properties if available
            edge_props = data.get('edge_properties', {})
            if edge_props:
                for key, value in edge_props.items():
                    if key not in ['type', 'element_id'] and value:
                        context_parts.append(f"- {key.replace('_', ' ').title()}: {value}")

            context_parts.append("")
        has_content = True

    # Add semantic node results - medium priority (only educational nodes)
    educational_nodes = [item for item in semantic_node_results
                        if item.get('node_type') in ['PedagogicalMethodology', 'LearnerProfile', 'ClassClimate', 'LearningSetting', 'Technologie']]

    if educational_nodes:
        context_parts.append(f"**🧠 Educational Content Results (Relevant Teaching Resources)**")
        context_parts.append("")
        for i, item in enumerate(educational_nodes[:3], 1):  # Limit to top 3
            score = item.get('weighted_score', 0.0)
            similarity = item.get('similarity', 0.0)
            node_type = item.get('node_type', 'Unknown')

            context_parts.append(f"**Result {i} (Relevance: {score:.3f})**")
            context_parts.append(f"- Similarity Score: {similarity:.3f}")
            context_parts.append(f"- Content Type: {node_type}")

            node_info = item.get('data', {}).get('node_info', 'N/A')
            context_parts.append(f"- Information: {node_info}")

            context_parts.append("")
        has_content = True

    # Add search strategy information only if we have content
    if has_content:
        context_parts.append(f"📊 **Search Strategy Information**")
        context_parts.append(f"- Cypher Results: {len(hybrid_result.get('cypher_results', []))}")
        context_parts.append(f"- Semantic Node Results: {len(semantic_node_results)}")
        context_parts.append(f"- Semantic Edge Results: {len(semantic_edge_results)}")
        context_parts.append(f"- Search Weights - Cypher: {hybrid_result.get('cypher_weight', 0.6)}, Semantic: {hybrid_result.get('semantic_weight', 0.4)}")
        context_parts.append("")

        # Add cache information
        if hybrid_result.get('from_cache', False):
            context_parts.append(f"💾 **Performance Note**: Results retrieved from cache for faster response.")
            context_parts.append("")

    return "\n".join(context_parts)

def _filter_relevant_results(hybrid_results: List[Dict]) -> List[Dict]:
    """
    Filter and prioritize results based on educational relevance.

    Args:
        hybrid_results: Raw hybrid search results

    Returns:
        Filtered and prioritized results
    """
    if not hybrid_results:
        return []

    filtered_results = []

    for item in hybrid_results:
        relevance_type = item.get('relevance_type', 'unknown')
        score = item.get('weighted_score', 0.0)

        # Always include high-scoring structural results
        if relevance_type == 'structural' and score > 0.5:
            filtered_results.append(item)
            continue

        # For semantic results, apply stricter filtering
        if relevance_type == 'semantic':
            subtype = item.get('subtype', 'node')

            if subtype == 'edge':
                # Include all educational relationship edges with reasonable scores
                if score > 0.4:
                    filtered_results.append(item)

            elif subtype == 'node':
                # Only include educational node types with good scores
                node_type = item.get('node_type', 'Unknown')
                education_boost = item.get('education_boost', 1.0)

                # Prioritize educational content
                if node_type in ['PedagogicalMethodology', 'LearnerProfile', 'ClassClimate', 'LearningSetting', 'Technologie']:
                    if score > 0.3 or education_boost > 1.5:
                        filtered_results.append(item)
                # Include other potentially relevant content with higher thresholds
                elif score > 0.6:
                    filtered_results.append(item)

    # Sort by score and return top results
    filtered_results.sort(key=lambda x: x.get('weighted_score', 0.0), reverse=True)
    return filtered_results[:10]  # Return top 10 most relevant

def build_compact_context(hybrid_result: Dict[str, Any], max_items: int = 5) -> str:
    """
    Build a compact context with only the most relevant results.
    
    Args:
        hybrid_result: Result from hybrid_retriever.process_query()
        max_items: Maximum number of items to include
        
    Returns:
        Compact formatted context string
    """
    if not hybrid_result or hybrid_result.get('total_results', 0) == 0:
        return "No relevant information found in the knowledge graph."
    
    context_parts = []
    
    # Add basic query info
    context_parts.append(f"Query: {hybrid_result['original_query']}")
    context_parts.append(f"Language: {hybrid_result['detected_language']}")
    context_parts.append(f"Results: {hybrid_result['total_results']}")
    context_parts.append("")
    
    # Get top results
    top_results = hybrid_result.get('hybrid_results', [])[:max_items]
    
    for i, item in enumerate(top_results, 1):
        relevance_type = item.get('relevance_type', 'unknown')
        score = item.get('weighted_score', 0.0)
        
        context_parts.append(f"**Result {i}** (Score: {score:.3f}, Type: {relevance_type})")
        
        if relevance_type == 'structural':
            data = item.get('data', {})
            if isinstance(data, dict) and 'name' in data:
                context_parts.append(f"- {data.get('name', 'N/A')}: {data.get('concept', data.get('description', 'N/A'))}")
        
        elif relevance_type == 'semantic':
            data = item.get('data', {})
            if data.get('subtype') == 'node':
                node_info = data.get('node_info', 'N/A')
                context_parts.append(f"- Node: {node_info}")
            else:
                source = data.get('source_node', 'N/A')
                target = data.get('target_node', 'N/A')
                rel_type = data.get('relationship_type', 'N/A')
                context_parts.append(f"- Relationship: {source} --[{rel_type}]--> {target}")
        
        context_parts.append("")
    
    return "\n".join(context_parts)

def extract_key_information(hybrid_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key information from hybrid results for structured processing.
    
    Args:
        hybrid_result: Result from hybrid_retriever.process_query()
        
    Returns:
        Dictionary with structured key information
    """
    if not hybrid_result or hybrid_result.get('total_results', 0) == 0:
        return {"found_information": False}
    
    key_info = {
        "found_information": True,
        "query": hybrid_result['original_query'],
        "language": hybrid_result['detected_language'],
        "total_results": hybrid_result['total_results'],
        "keywords": hybrid_result.get('keywords', {}),
        "cypher_query": hybrid_result.get('cypher_query'),
        "structural_results": [],
        "semantic_results": [],
        "methodologies": [],
        "resources": [],
        "learner_profiles": []
    }
    
    for item in hybrid_result.get('hybrid_results', []):
        relevance_type = item.get('relevance_type', 'unknown')
        data = item.get('data', {})
        
        if relevance_type == 'structural':
            key_info['structural_results'].append({
                "score": item.get('weighted_score', 0.0),
                "data": data
            })
            
            # Extract specific types of information
            if isinstance(data, dict):
                if data.get('type') == 'methodology':
                    key_info['methodologies'].append(data)
                elif data.get('type') == 'resource':
                    key_info['resources'].append(data)
                elif 'profile' in data.get('name', '').lower():
                    key_info['learner_profiles'].append(data)
        
        elif relevance_type == 'semantic':
            key_info['semantic_results'].append({
                "score": item.get('weighted_score', 0.0),
                "similarity": item.get('similarity', 0.0),
                "subtype": item.get('subtype'),
                "data": data
            })
    
    return key_info
