#!/usr/bin/env python3
"""
OPTIMIZED CYPHER GENERATION - Multilingual Neural Science
Core functions: text2cypher, detect_language, extract_keywords, validate_cypher
Enhanced: Directional relationship awareness to prevent wrong relationship directions
"""

import os, re
from typing import Optional, Tuple, Dict, List, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

def get_dynamic_schema(driver) -> str:
    """Retrieve schema automatically from Neo4j"""
    try:
        with driver.session() as session:
            # Get available nodes and relationships
            node_query = """
            MATCH (n)
            WITH labels(n) as nodeLabels, count(*) as cnt
            RETURN nodeLabels, cnt
            ORDER BY cnt DESC
            LIMIT 20
            """
            node_result = session.run(node_query)
            all_node_labels = set()
            for record in node_result:
                labels = record["nodeLabels"]
                all_node_labels.update(labels)
            node_list = sorted(list(all_node_labels))

            rel_query = """
            MATCH ()-[r]-()
            RETURN distinct type(r) as relType
            ORDER BY relType
            LIMIT 15
            """
            rel_result = session.run(rel_query)
            all_rel_types = sorted([record["relType"] for record in rel_result])

            # Format for LLM
            schema_text = f"""
DYNAMIC NODES: {', '.join(node_list)}
DYNAMIC RELATIONSHIPS: {', '.join(all_rel_types)}

Use only these existing nodes and relationships in your Cypher query.
"""

            return schema_text.strip()

    except Exception as e:
        print(f"❌ Schema retrieval failed: {e}, using fallback")
        return """
NODES: Attention, Memory, ExecutiveControl, CognitiveLoad, WorkingMemory, SustainedAttention, Neuroplasticity
RELATIONSHIPS: SUPPORTS, HINDERS, CAUSES, ALIGNS_WITH, FACILITATES, IMPAIRS

Use these neuroscience nodes and relationships in your Cypher query.
"""

# FLEXIBLE CONTENT-BASED PROMPT
DYNAMIC_PROMPT = PromptTemplate.from_template("""
TASK: Generate very flexible Cypher queries using broad content matching. Return ONLY Cypher code.

{schema}

CRITICAL RULES:
1. Use GENERIC patterns: (a)-[r]-(b)
2. Use BROAD OR conditions: WHERE ... OR ... OR ...
3. Use SINGLE keywords, not combinations
4. Always include: RETURN a.name, type(r), b.name LIMIT 15

FLEXIBLE EXAMPLES:
USER: "What is intrinsic motivation?"
CYPHER: MATCH (a)-[r]-(b) WHERE toLower(a.name) CONTAINS 'motivation' OR toLower(b.name) CONTAINS 'motivation' RETURN a.name, type(r), b.name LIMIT 15

USER: "Connection between attention and memory"
CYPHER: MATCH (a)-[r]-(b) WHERE toLower(a.name) CONTAINS 'attention' OR toLower(b.name) CONTAINS 'attention' OR toLower(a.name) CONTAINS 'memory' OR toLower(b.name) CONTAINS 'memory' RETURN a.name, type(r), b.name LIMIT 15

YOUR OUTPUT (only Cypher, no explanations):
{query}

CYPHER:""")

def get_neuroscience_fallback(query: str) -> Optional[str]:
    """Rule-based fallbacks for neuroscience teacher queries"""
    q = query.lower()

    if any(word in q for word in ['motivation', 'motivazione', 'intrinsic', 'extrinsic']):
        return "MATCH (a)-[r]-(b) WHERE toLower(a.name) CONTAINS 'motivation' OR toLower(b.name) CONTAINS 'motivation' RETURN a.name, type(r), b.name LIMIT 15"

    if any(word in q for word in ['stress', 'eustress', 'distress', 'anxiety']):
        return "MATCH (a)-[r]-(b) WHERE toLower(a.name) CONTAINS 'stress' OR toLower(b.name) CONTAINS 'stress' RETURN a.name, type(r), b.name LIMIT 15"

    if any(word in q for word in ['learning', 'apprendimento', 'learn', 'imparare']):
        return "MATCH (a)-[r]-(b) WHERE toLower(a.name) CONTAINS 'learning' OR toLower(b.name) CONTAINS 'learning' RETURN a.name, type(r), b.name LIMIT 15"

    return "MATCH (a)-[r]-(b) WHERE a.name IS NOT NULL AND b.name IS NOT NULL RETURN a.name, type(r), b.name LIMIT 15"

def detect_language(query: str) -> str:
    """Ultra-fast language detection"""
    if not query: return "xx"
    q_lower = query.lower()
    italian_markers = ["come", "perché", "studenti", "insegnanti"]
    english_markers = ["how", "why", "students", "teachers"]
    italian_count = sum(1 for marker in italian_markers if marker in q_lower)
    english_count = sum(1 for marker in english_markers if marker in q_lower)
    return "it" if italian_count > english_count else "en"

def query_verified_relationships(keywords: Dict[str, List[str]], driver) -> List[str]:
    """
    Query the graph for VERIFIED directional relationships involving query concepts.
    This provides the LLM with real relationship directions to prevent bidirectional hallucinations.
    """
    from services.enhanced_graph_retriever import run_graph_query

    # Flatten all keywords into search concepts
    search_concepts = []
    for concept_list in keywords.values():
        search_concepts.extend(concept_list)

    # Remove duplicates, limit to prevent explosion
    search_concepts = list(set(search_concepts[:5]))  # Max 5 concepts for directional context

    if not search_concepts:
        print("⚠️ No concepts found for directional relationship verification")
        return []

    # Build Cypher to get verified directional relationships
    concepts_str = "', '".join([c.replace("'", "''") for c in search_concepts])

    # Query for relationships in BOTH directions to understand actual graph structure
    query = f"""
    MATCH (a)-[r]->(b)
    WHERE toLower(a.name) IN ['{concepts_str}']
       OR toLower(b.name) IN ['{concepts_str}']
    RETURN DISTINCT a.name as source, type(r) as relationship, b.name as target
    ORDER BY source, relationship
    LIMIT 10
    """

    try:
        results = run_graph_query(query, quiet=True)  # Silent to avoid validation noise
        verified_rels = []

        for record in results:
            source = record['source']
            relationship = record['relationship']
            target = record['target']

            # Format as directional arrow: "Source --RELATION--> Target"
            verified_rel = f"{source} --{relationship}--> {target}"
            verified_rels.append(verified_rel)

        return verified_rels

    except Exception as e:
        print(f"❌ Error querying verified relationships: {e}")
        return []

def calculate_directional_confidence(keywords: Dict[str, List[str]], verified_relationships: List[str], driver) -> float:
    """
    Calculate confidence in our directional understanding (0.0 - 1.0)
    Higher confidence = Can be more restrictive with directions
    Lower confidence = Need broader, more flexible querying
    """
    # Factor 1: Relationship Density (0.35 weight)
    relationship_count = len(verified_relationships)
    density_score = min(relationship_count / 5.0, 1.0)  # Max at 5+ relationships

    # Factor 2: Concept Coverage (0.35 weight)
    key_concepts = []
    for concept_list in keywords.values():
        key_concepts.extend(concept_list)
    key_concepts = list(set(key_concepts))  # Remove duplicates

    if not key_concepts:
        return 0.0

    # Count how many query concepts appear in verified relationships
    covered_concepts_count = 0
    rel_text = " ".join(verified_relationships).lower()

    for concept in key_concepts:
        if concept.lower() in rel_text:
            covered_concepts_count += 1

    coverage_score = min(covered_concepts_count / len(key_concepts), 1.0)

    # Factor 3: Connection Complexity (0.20 weight) - How interconnected the concepts are
    complexity_score = 0.0
    try:
        # Count unique nodes and relationships to assess complexity
        all_nodes = set()
        all_rel_types = set()

        for rel in verified_relationships:
            # Parse "Source --REL--> Target" format
            if " --" in rel and "--> " in rel:
                parts = rel.split(" --")
                if len(parts) == 2:
                    source = parts[0].strip()
                    rel_target = parts[1].split("--> ")
                    if len(rel_target) == 2:
                        rel_type = rel_target[0].strip()
                        target = rel_target[1].strip()

                        all_nodes.add(source)
                        all_nodes.add(target)
                        all_rel_types.add(rel_type)

        node_count = len(all_nodes)
        rel_type_count = len(all_rel_types)

        # Higher complexity = more trust in our directional understanding
        complexity_score = min((node_count + rel_type_count) / 10.0, 1.0)

    except Exception as e:
        print(f"⚠️ Error calculating complexity: {e}")
        complexity_score = 0.2  # Default moderate complexity

    # Factor 4: Neuroscience Specificity (0.10 weight)
    neuroscience_specific_rels = ['SUPPORTS', 'HINDERS', 'FACILITATES', 'IMPAIRS', 'ENHANCES',
                                  'IS_INFLUENCED_BY', 'CAUSES', 'ALIGNS_WITH', 'ENHANCES_LEARNING']
    neuroscience_score = 0.0

    for rel in verified_relationships:
        for neuro_rel in neuroscience_specific_rels:
            if neuro_rel in rel:
                neuroscience_score = 0.8  # Neuroscience-specific relationships increase confidence
                break

    if neuroscience_score == 0.0:
        neuroscience_score = 0.2  # Some basic relationships

    # Weighted confidence score
    confidence = (
        density_score * 0.35 +
        coverage_score * 0.35 +
        complexity_score * 0.20 +
        neuroscience_score * 0.10
    )

    confidence = min(confidence, 1.0)
    print(f"🎯 Directional confidence: {confidence:.2f} (density: {density_score:.2f}, coverage: {coverage_score:.2f})")

    return confidence

def extract_keywords(query: str, language: str = None) -> Dict[str, List[str]]:
    """Extract neuroscience keywords grouped by cognitive domain"""
    if not query: return {}
    if language is None:
        language = detect_language(query)
    q_lower = query.lower()

    # Basic neuroscience keywords mapping
    NEUROSCIENCE_KEYWORDS = {
        "attention": ["attention", "focus", "concentration"],
        "memory": ["memory", "remember", "recall"],
        "executive": ["executive", "control", "planning"],
        "emotional": ["emotional", "motivation", "stress", "emotion"],
        "learning": ["learning", "plasticity", "neuroplasticity"]
    }

    result = {}
    for domain, keywords in NEUROSCIENCE_KEYWORDS.items():
        found = [kw for kw in keywords if kw in q_lower]
        if found:
            result[domain] = found
    return result

def clean_cypher(query: str) -> str:
    """Extract pure Cypher code"""
    if not query: return ""
    query = query.replace("```cypher", "").replace("```", "").replace("`", "").strip()
    query = query.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")

    cypher_pattern = r'(MATCH.*?LIMIT \d+)'
    match = re.search(cypher_pattern, query, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    cypher_pattern2 = r'(MATCH.*?RETURN.*?(?:LIMIT \d+)?(?:\s*WHERE.*)?)'
    match = re.search(cypher_pattern2, query, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    lines = query.split('\n')
    cypher_lines = []
    for line in lines:
        line_upper = line.upper()
        if any(keyword in line_upper for keyword in ['MATCH', 'RETURN', 'WHERE', 'LIMIT']):
            cypher_lines.append(line.strip())

    result = ' '.join(cypher_lines[:3]).strip()
    result = re.sub(r'^(?:CYPHER|OUTPUT|Result|Query):\s*', '', result, flags=re.IGNORECASE)
    result = re.sub(r'^.*?:\s*(MATCH)', r'\1', result, flags=re.IGNORECASE)
    return result

def validate_cypher(query: str) -> Tuple[bool, str]:
    """Basic validation for Cypher queries"""
    if not query or len(query) < 10: return False, "Too short"
    if not query.upper().startswith('MATCH'): return False, "Must start with MATCH"
    if 'RETURN' not in query.upper(): return False, "Missing RETURN"
    return True, "Valid"

def validate_relationship_direction(source_name: str, rel_type: str, target_name: str, driver) -> bool:
    """
    Validate that a specific directional relationship exists in the graph.
    This filters out hallucinations while keeping real relationships.
    """
    from services.enhanced_graph_retriever import run_graph_query

    try:
        # Query to check if this exact relationship direction exists
        validation_query = f"""
        MATCH (a)-[r:{rel_type}]->(b)
        WHERE toLower(a.name) = toLower('{source_name.replace("'", "''")}')
          AND toLower(b.name) = toLower('{target_name.replace("'", "''")}')
        RETURN count(r) > 0 as exists
        """

        results = run_graph_query(validation_query, quiet=True)
        if results and len(results) > 0:
            return results[0].get('exists', False)

        return False

    except Exception as e:
        print(f"⚠️ Error validating relationship {source_name}--{rel_type}-->{target_name}: {e}")
        return False

def filter_real_relationships(cypher_results: List[Dict], driver) -> List[Dict]:
    """
    Filter Cypher results to only include relationships that actually exist in the graph.
    Removes hallucinations while preserving real directional relationships.
    """
    if not cypher_results:
        return []

    validated_results = []
    filtered_count = 0

    for result in cypher_results:
        try:
            # Parse the result to get relationship components
            data = result.get('data', {})

            # Standard Cypher result format: {'a.name': 'source', 'type(r)': 'REL_TYPE', 'b.name': 'target'}
            source_name = data.get('a.name', '').strip()
            rel_type = data.get('type(r)', '').strip()
            target_name = data.get('b.name', '').strip()

            if source_name and rel_type and target_name:
                # Validate this relationship actually exists
                if validate_relationship_direction(source_name, rel_type, target_name, driver):
                    validated_results.append(result)
                else:
                    filtered_count += 1
                    print(f"🚫 Filtered hallucination: {source_name}--{rel_type}-->{target_name} doesn't exist in graph")

        except Exception as e:
            print(f"⚠️ Error parsing relationship result: {e}")
            filtered_count += 1

    print(f"✅ Kept {len(validated_results)} real relationships, filtered {filtered_count} hallucinations")

    return validated_results

def text2cypher(query: str, driver=None) -> Optional[str]:
    """Main function: Generate Cypher with broad coverage, validate directionally for accuracy"""
    if not query or not query.strip(): return None

    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key: raise ValueError("OpenAI API key required")

        if driver is None:
            from neo4j import GraphDatabase
            neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
            neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
            print("🔌 Created new Neo4j driver for schema retrieval")

        # STEP 1: Get dynamic schema
        dynamic_schema = get_dynamic_schema(driver)
        print("📊 Using dynamic schema for broad content matching")

        # STEP 2: Generate broad, flexible Cypher for MAXIMUM coverage
        cypher = None

        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

            # MAXIMUM COVERAGE APPROACH: Generate broadly, validate specifically
            print("🔄 Using broad flexible generation for maximum coverage")
            chain = DYNAMIC_PROMPT | llm
            result = chain.invoke({"query": query, "schema": dynamic_schema})

            cypher = str(result).strip() if result else ""
            print("✅ Generated with broad coverage approach")

        except Exception as primary_error:
            print(f"⚠️ Primary provider failed ({str(primary_error)[:100]}), trying fallback")

            try:
                openrouter_key = os.getenv("OPENROUTER_API_KEY")
                if openrouter_key:
                    os.environ["OPENAI_API_KEY"] = openrouter_key
                    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
                    llm = ChatOpenAI(model="openai/gpt-oss-20b:free", temperature=0.0)
                    chain = DYNAMIC_PROMPT | llm
                    result = chain.invoke({"query": query, "schema": dynamic_schema})
                    cypher = str(result).strip() if result else ""
                    print("✅ Generated with OpenRouter fallback")
                else:
                    print("⚠️ No OpenRouter fallback key available")
            except Exception as fallback_error:
                print(f"❌ All LLM providers failed. Primary: {str(primary_error)[:50]}, Fallback: {str(fallback_error)[:50]}")

        if not cypher:
            print("❌ No LLM provider generated a response")

        cypher = clean_cypher(cypher)
        is_valid, msg = validate_cypher(cypher)

        if is_valid:
            print("🎯 Generated Cypher with maximum coverage - direction validation will happen post-execution")
            return cypher
        else:
            print(f"⚠️ Validation failed: {msg}")

    except Exception as e:
        print(f"❌ Generation failed: {e}")

    fallback = get_neuroscience_fallback(query)
    if fallback:
        print("🔄 Using neuroscience fallback with directional awareness")
        return fallback

    return None
