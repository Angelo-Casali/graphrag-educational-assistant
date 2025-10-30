#!/usr/bin/env python3
"""
OPTIMIZED CYPHER GENERATION - Multilingual Neural Science
Core functions: text2cypher, detect_language, extract_keywords, validate_cypher
Enhanced: Directional relationship awareness to prevent wrong relationship directions
"""

import os, re
import logging
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DYNAMIC SCHEMA FUNCTIONS (From services/enhanced_graph_retriever.py)
# ============================================================================

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

    if any(word in q for word in ['stress', 'ansia', 'anxiety', 'eustress', 'distress']):
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
    from enhanced_graph_retriever import run_graph_query

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
    from enhanced_graph_retriever import run_graph_query

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

@dataclass
class SchemaInfo:
    """Data class to hold Neo4j schema information"""
    node_labels: List[str]
    relationship_types: List[str]
    node_properties: Dict[str, List[str]]
    sample_nodes: Dict[str, List[Dict]]

class Neo4jSchemaExtractor:
    """Extract schema information from Neo4j database"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the Neo4j connection"""
        self.driver.close()
    
    def extract_schema(self) -> SchemaInfo:
        """Extract comprehensive schema information from the database"""
        with self.driver.session() as session:
            # Get node labels
            node_labels = session.run("CALL db.labels()").value()
            
            # Get relationship types
            relationship_types = session.run("CALL db.relationshipTypes()").value()
            
            # Get node properties for each label
            node_properties = {}
            sample_nodes = {}
            
            for label in node_labels:
                # Get properties for this label
                props_query = f"MATCH (n:{label}) RETURN DISTINCT keys(n) as props LIMIT 1"
                props_result = session.run(props_query).single()
                if props_result:
                    node_properties[label] = props_result["props"]
                else:
                    node_properties[label] = []
                
                # Get sample nodes for this label
                sample_query = f"MATCH (n:{label}) RETURN n LIMIT 3"
                samples = []
                for record in session.run(sample_query):
                    node_dict = dict(record["n"])
                    samples.append(node_dict)
                sample_nodes[label] = samples
        
        return SchemaInfo(
            node_labels=node_labels,
            relationship_types=relationship_types,
            node_properties=node_properties,
            sample_nodes=sample_nodes
        )

class Text2CypherConverter:
    """Main class for converting natural language to Cypher queries"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
        self.schema_extractor = Neo4jSchemaExtractor(neo4j_uri, neo4j_user, neo4j_password)
        self.schema_info = self.schema_extractor.extract_schema()
        
        # Initialize OpenAI LLM with GPT-4o-mini (128K context)
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=0.1,  # Low temperature for more consistent outputs
            max_tokens=500
        )
        
        # Create the prompt template and chain using LCEL
        self.prompt_template = self._create_prompt_template()
        self.output_parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create a comprehensive prompt template for text2cypher conversion"""
        
        # Build schema description
        schema_desc = self._build_schema_description()
        
        # Create few-shot examples based on the actual schema
        examples = self._create_few_shot_examples()
        
        # Build template without f-string to avoid variable interpolation issues
        template = """You are a Neo4j Cypher query expert for an educational knowledge graph system.

""" + schema_desc + """

IMPORTANT RULES:
1. Always use EXACT node labels and property names from the schema above
2. Use case-insensitive matching with toLower() for text searches
3. Return LIMIT 20 unless specified otherwise
4. For "what" questions, return node properties
5. For "how many" questions, use COUNT()
6. Use SUGGESTS relationship for positive recommendations
7. Use NO_SUGGESTS relationship for negative recommendations
8. Always return meaningful node properties like name and category

EXAMPLES:
""" + examples + """

QUERY PATTERNS:
- Motivation concepts: MATCH (m:MotivationalModulation)-[r:IS_CONTRASTED_WITH|SUPPORTS]->(o:LearningOutcomes) RETURN m.name, type(r), o.name
- Stress and learning: MATCH (s:PositiveStressEustress|NegativeStressDistress)-[r:SUPPORTS|UNDERMINES]->(l:LearningDevelopment)
- Mindset patterns: MATCH (m:GrowthMindset|FixedMindset)-[r:SUPPORTS|UNDERMINES]->(ld:LearningDevelopment) RETURN m.name, type(r), ld.name
- Emotions and cognition: MATCH (e:PositiveEmotions|NegativeEmotions)-[r:ENHANCE|INTERFERE_WITH]->(c:CognitiveProcesses)
- Metacognition: MATCH (m:Metacognition)-[r:STRENGTHENS|ENABLES]->(s:SelfRegulation) RETURN m.name, type(r), s.name
- Attention processes: MATCH (a:Attention)-[r:SUPPORTS|IS_IMPAIRED_BY]->(ld:LearningDevelopment) RETURN a.name, type(r), ld.name
- Memory systems: MATCH (m:WorkingMemory|LongTermMemory)-[r:FACILITATES|ENHANCES]->(learn:LearningOutcomes) RETURN m.name, type(r), learn.name
- Executive functions: MATCH (e:ExecutiveFunctions)-[r:ESSENTIAL_FOR|REQUIRES|IMPAIRED_BY]->(o:LearningOutcomes) RETURN e.name, type(r), o.name
- Critical thinking: MATCH (c:CriticalThinking)-[r:BUILDS_ON|DEPENDS_ON]->(ecf:ExecutiveFunctions) RETURN c.name, type(r), ecf.name
- Creativity and cognition: MATCH (cr:Creativity)-[r:SUPPORTS|ENHANCES]->(cf:CognitiveFlexibility) RETURN cr.name, type(r), cf.name
- Learned helplessness: MATCH (lh:LearnedHelplessness)-[r:LEADS_TO|REDUCES]->(m:MotivationalModulation) RETURN lh.name, type(r), m.name
- Neuroplasticity: MATCH (np:Neuroplasticity)-[r:SUPPORTED_BY|ENHANCES]->(ld:LearningDevelopment) RETURN np.name, type(r), ld.name
- Cognitive load: MATCH (cl:Cognitiveload|CognitiveLoad)-[r:INCREASES|IMPARS]->(wm:WorkingMemory|Attention) RETURN cl.name, type(r), wm.name

Convert this natural language question to a Cypher query:
Question: {question}

Cypher Query:"""

        return ChatPromptTemplate.from_messages([
            ("system", "You are a Neo4j Cypher query expert for a neuroscience knowledge graph system."),
            ("human", template)
        ])
    
    def _build_schema_description(self) -> str:
        """Build a comprehensive schema description"""
        desc = "DATABASE SCHEMA:\n"
        desc += "=" * 50 + "\n\n"
        
        desc += "NODE LABELS:\n"
        for label in self.schema_info.node_labels:
            desc += f"- {label}\n"
            if label in self.schema_info.node_properties:
                props = self.schema_info.node_properties[label]
                desc += f"  Properties: {', '.join(props)}\n"
            
            # Add sample data without dictionary representation
            if label in self.schema_info.sample_nodes and self.schema_info.sample_nodes[label]:
                sample = self.schema_info.sample_nodes[label][0]
                # Format sample data safely without single quotes
                sample_str = ", ".join([f"{k}: {v}" for k, v in sample.items()])
                desc += f"  Example: {sample_str}\n"
            desc += "\n"
        
        desc += "RELATIONSHIP TYPES:\n"
        for rel_type in self.schema_info.relationship_types:
            desc += f"- {rel_type}\n"
        
        return desc
    
    def _create_few_shot_examples(self) -> str:
        """Create comprehensive few-shot examples based on neuroscience knowledge graph"""
        examples = """
Question: "What is the difference between intrinsic and extrinsic motivation?"
Cypher: MATCH (i:AcademicMotivation)-[r:IS_LINKED_WITH]-(e:AcademicMotivation) WHERE i.name CONTAINS "Avoidance" AND e.name CONTAINS "avoidance" RETURN i.name, type(r), e.name LIMIT 15

Question: "Can stress sometimes be positive for learning?"
Cypher: MATCH (p:AffectiveMotivationalProcesses)-[r:SUPPORTS]->(l:LearningDevelopment) WHERE p.name CONTAINS "stress" RETURN p.name, type(r), l.name LIMIT 10

Question: "What does growth mindset mean?"
Cypher: MATCH (g:GrowthMindset)-[r:SUPPORTS|ENHANCES]->(o) RETURN g.name, type(r), o.name, labels(o) LIMIT 15

Question: "How do emotions affect student learning?"
Cypher: MATCH (e:AffectiveProcesses)-[r:ENHANCE|INTERFERE_WITH]->(c:CognitiveProcesses) RETURN e.name, type(r), c.name LIMIT 10

Question: "What is metacognition and why is it important?"
Cypher: MATCH (m:Metacognition)-[r:STRENGTHENS|ENHANCES]->(o) RETURN m.name, type(r), o.name, labels(o) LIMIT 15

Question: "How can I encourage intrinsic motivation in students?"
Cypher: MATCH (i:AffectiveMotivationalProcesses)-[r:SUPPORTS]->(l:LearningOutcomes) WHERE i.name CONTAINS "intrinsic" RETURN i.name, type(r), l.name LIMIT 10

Question: "How does stress influence motivation?"
Cypher: MATCH (s:PositiveStressEustress)-[r:AFFECTS|SUPPORTS]->(m:AcademicMotivation) RETURN s.name, type(r), m.name LIMIT 10

Question: "What's the link between emotions and mindset?"
Cypher: MATCH (e:AffectiveProcesses)-[r:ARE_LINKED_WITH]->(g:GrowthMindset) RETURN e.name, type(r), g.name LIMIT 10

Question: "What cognitive processes support memory encoding?"
Cypher: MATCH (p:CognitiveProcesses)-[r:FACILITATES|ENHANCES]->(m:MemorySystems) RETURN p.name, labels(p), type(r), m.name LIMIT 10

Question: "How does attention modulate learning?"
Cypher: MATCH (a:Attention)-[r:SUPPORTS|ENHANCES|AFFECTS]->(o:LearningOutcomes) RETURN a.name, type(r), o.name, labels(o) LIMIT 15

Question: "What factors impair executive functions?"
Cypher: MATCH (f:Attention)-[r:IS_IMPAIRED_BY]->(e:ExecutiveFunctions) RETURN f.name, labels(f), type(r), e.name LIMIT 10

Question: "How does working memory affect learning?"
Cypher: MATCH (w:WorkingMemory)-[r:ENHANCES|SUPPORTS|AFFECTS]->(o:LearningOutcomes) RETURN w.name, type(r), o.name, labels(o) LIMIT 15

Question: "What enhances critical thinking?"
Cypher: MATCH (n:CognitiveProcesses)-[r:ENHANCES|SUPPORTS|STRENGTHENS]->(c:CriticalThinking) RETURN n.name, labels(n), type(r), c.name LIMIT 10

Question: "How many types of attention are in the database?"
Cypher: MATCH (a:Attention) RETURN COUNT(DISTINCT a.name) as attention_types

Question: "What is the relationship between creativity and cognitive flexibility?"
Cypher: MATCH (c:Creativity)-[r:SUPPORTS|ENHANCES]->(cf:CognitiveFlexibility) RETURN c.name, type(r), cf.name LIMIT 10

Question: "How does learned helplessness affect motivation?"
Cypher: MATCH (lh:LearnedHelplessness)-[r:REDUCES|AFFECTS]->(m:AcademicMotivation) RETURN lh.name, type(r), m.name LIMIT 10

Question: "What helps overcome negative stress for learning?"
Cypher: MATCH (ns:NegativeStressDistress)-[r:IS_LINKED_WITH]->(adapt:AdaptiveCoping) RETURN ns.name, type(r), adapt.name LIMIT 10

Question: "How does positive stress enhance performance?"
Cypher: MATCH (ps:PositiveStressEustress)-[r:SUPPORTS|ENHANCES]->(o:AcademicWorkOutcomes) RETURN ps.name, type(r), o.name LIMIT 10

Question: "What is the impact of fixed mindset on learning?"
Cypher: MATCH (fm:FixedMindset)-[r:UNDERMINES]->(ld:LearningDevelopment) RETURN fm.name, type(r), ld.name LIMIT 10

Question: "How do executive functions support critical thinking?"
Cypher: MATCH (ef:ExecutiveFunctions)-[r:ESSENTIAL_FOR|REQUIRES]->(ct:CriticalThinking) RETURN ef.name, type(r), ct.name LIMIT 15

Question: "What role does memory play in creativity?"
Cypher: MATCH (mem:MemorySystems)-[r:SUPPORTS|FACILITATES]->(cr:Creativity) RETURN mem.name, type(r), cr.name LIMIT 10

Question: "How does neuroplasticity support learning?"
Cypher: MATCH (np:Neuroplasticity)-[r:SUPPORTED_BY]->(ld:LearningDevelopment) RETURN np.name, type(r), ld.name LIMIT 10

Question: "What impairs working memory and attention?"
Cypher: MATCH (cogla:Cognitiveload|CognitiveLoad)-[r:INCREASES|IMPAIRS]->(wm:WorkingMemory|Attention) RETURN cogla.name, type(r), wm.name LIMIT 10

Question: "How do emotions modulate attention?"
Cypher: MATCH (em:AffectiveProcesses)-[r:MODULATES|ENHANCE|INTERFERE_WITH]->(att:Attention) RETURN em.name, type(r), att.name LIMIT 15

Question: "What cognitive biases affect learning?"
Cypher: MATCH (bias:CognitiveBiases)-[r:REINFORCES|DISTORTS|AFFECTS]->(learn:LearningOutcomes) RETURN bias.name, type(r), learn.name LIMIT 10

Question: "How does mindfulness improve attention?"
Cypher: MATCH (mindf:EducationalClinicalInterventions)-[r:TRAINED_BY|STRENGTHENED_THROUGH]->(att:Attention) WHERE mindf.name CONTAINS "Mindfulness" RETURN mindf.name, type(r), att.name LIMIT 10

Question: "What is the relationship between ADHD and attention?"
Cypher: MATCH (adhd:AffectiveBiologicalConstraints)-[r:IMPAIRED_BY|IS_IMPAIRED_IN]->(att:Attention) WHERE adhd.name CONTAINS "ADHD" RETURN adhd.name, type(r), att.name LIMIT 10

Question: "How do motor coordination issues affect learning?"
Cypher: MATCH (motor:CognitiveControl|ExecutiveFunctions)-[r:AFFECTS]->(learn:LearningOutcomes) RETURN motor.name, type(r), learn.name LIMIT 10

Question: "What supports resilience in learning?"
Cypher: MATCH (resil:Resilience)-[r:PROMOTES|FOSTERS]->(learn:LearningOutcomes) RETURN resil.name, type(r), learn.name LIMIT 10

Question: "How does sleep affect memory consolidation?"
Cypher: MATCH (slee:PsychoBiologicalfactor)-[r:SUPPORTED_BY]->(mem:MemoryStabilization) WHERE slee.name CONTAINS "sleep" RETURN slee.name, type(r), mem.name LIMIT 10

Question: "Qual è la differenza tra motivazione intrinseca ed estrinseca?"
Cypher: MATCH (i:AcademicMotivation)-[r:IS_LINKED_WITH]->(e:AcademicMotivation) WHERE i.name CONTAINS "Avoidance" RETURN i.name, type(r), e.name LIMIT 10

Question: "Come aiuta lo stress l'apprendimento?"
Cypher: MATCH (p:PositiveStressEustress|AffectiveMotivationalProcesses)-[r:SUPPORTS|AFFECTS]->(l:LearningDevelopment) RETURN p.name, type(r), l.name LIMIT 10

Question: "Che cos'è il pensiero critico?"
Cypher: MATCH (c:CriticalThinking)-[r:ARE_STRENGTHENED_THROUGH]->(o) RETURN c.name, type(r), o.name, labels(o) LIMIT 15

Question: "Come funziona la memoria di lavoro?"
Cypher: MATCH (w:WorkingMemory)-[r:ENHANCES|AFFECTS]->(l:LearningOutcomes) RETURN w.name, type(r), l.name LIMIT 15
"""
        return examples.strip()
    
    def convert(self, question: str) -> Tuple[str, Dict]:
        """Convert natural language question to Cypher query"""
        try:
            with get_openai_callback() as cb:
                # Generate Cypher query using LCEL
                result = self.chain.invoke({"question": question})
                
                # Clean the result
                cypher_query = self._clean_cypher_query(result)
                
                # Repair common issues based on audit findings
                cypher_query = self._repair_cypher(cypher_query)
                
                # Validate the query
                is_valid, validation_error = self._validate_cypher(cypher_query)
                
                metadata = {
                    "tokens_used": cb.total_tokens,
                    "cost": cb.total_cost,
                    "is_valid": is_valid,
                    "validation_error": validation_error,
                    "original_question": question
                }
                
                return cypher_query, metadata
                
        except Exception as e:
            logger.error(f"Error converting question to Cypher: {e}")
            return "", {"error": str(e), "is_valid": False}
    
    def _clean_cypher_query(self, raw_query: str) -> str:
        """Clean and format the generated Cypher query"""
        query = raw_query.strip()
        
        # Remove markdown code blocks (```cypher ... ``` or ``` ... ```)
        # Pattern 1: ```cypher\nMATCH...\n```
        if query.startswith('```cypher'):
            query = query[9:].strip()  # Remove ```cypher
        elif query.startswith('```'):
            query = query[3:].strip()  # Remove ```
        
        # Remove closing ```
        if query.endswith('```'):
            query = query[:-3].strip()
        
        # Remove any explanatory text before/after the query
        lines = query.split('\n')
        cypher_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and explanatory text
            if (line and 
                not line.startswith('Question:') and 
                not line.startswith('Answer:') and
                not line.startswith('Explanation:') and
                not line.startswith('Note:') and
                not line.startswith('```')):  # Skip any remaining code fence markers
                cypher_lines.append(line)
        
        query = ' '.join(cypher_lines)
        
        # Remove common prefixes
        prefixes_to_remove = [
            "Cypher Query:",
            "Cypher:",
            "Query:",
            "The Cypher query is:",
            "Here's the Cypher query:",
            "Here is the Cypher query:"
        ]
        
        for prefix in prefixes_to_remove:
            if query.startswith(prefix):
                query = query[len(prefix):].strip()
        
        return query
    
    def _repair_cypher(self, query: str) -> str:
        """Repair common Cypher query issues based on actual data patterns"""
        import re
        
        # 1) Force default pattern for StudentWithSpecialNeeds (validated pattern)
        # Change PedagogicalStrategy -> PedagogicalMethodology (our audit shows this is the correct pattern)
        query = re.sub(
            r'(\(s:StudentWithSpecialNeeds\)\s*-\s*\[r:\s*SUGGESTS\s*\]->\s*\()[a-zA-Z]*:PedagogicalStrategy(\))',
            r'\1m:PedagogicalMethodology\2',
            query
        )
        query = re.sub(
            r'(\(s:StudentWithSpecialNeeds\)\s*-\s*\[r:\s*NO_SUGGESTS\s*\]->\s*\()[a-zA-Z]*:PedagogicalStrategy(\))',
            r'\1m:PedagogicalMethodology\2',
            query
        )
        
        # 2) Remove subject-based constraints that don't exist in schema
        query = re.sub(r'AND\s+m\.name\s*=\s*"[Ss]cience[s]?"', '', query)
        query = re.sub(r'AND\s+m\.subject\s*=\s*"[^"]*"', '', query)
        query = re.sub(r'WHERE\s+m\.name\s*=\s*"[Ss]cience[s]?"\s+AND\s+', 'WHERE ', query)
        
        # 3) Fix bracketed cognitive disability cases: equality -> contains
        query = re.sub(
            r's\.name\s*=\s*"Cognitive disability \[mild, moderate, severe\]"',
            'toLower(s.name) CONTAINS "cognitive disability"',
            query,
            flags=re.IGNORECASE
        )
        
        # 4) Convert strict equality to case-insensitive for better matching
        # But preserve the synonym patterns we already have working
        pattern = r'WHERE\s+s\.name\s*=\s*"([^"]+)"(?!\s+OR\s+s\.name)'
        def make_case_insensitive(match):
            term = match.group(1)
            # Don't change if it's already part of a synonym list
            return f'WHERE toLower(s.name) = toLower("{term}")'
        query = re.sub(pattern, make_case_insensitive, query)
        
        # 5) Fix newlines in string literals
        query = query.replace("Assessment \nand evaluation", "Assessment and evaluation")
        query = query.replace("\n", " ")
        
        # 5.1) Fix missing closing parenthesis in WHERE clauses
        # Pattern: WHERE ... CONTAINS toLower("term" RETURN -> WHERE ... CONTAINS toLower("term") RETURN
        query = re.sub(r'CONTAINS toLower\("([^"]+)"\s+RETURN', r'CONTAINS toLower("\1") RETURN', query)
        
        # 5.2) Fix missing closing parenthesis in AND clauses
        # Pattern: AND toLower(s.name) CONTAINS toLower("term" RETURN -> AND toLower(s.name) CONTAINS toLower("term") RETURN
        query = re.sub(r'AND toLower\([^)]+\) CONTAINS toLower\("([^"]+)"\s+RETURN', r'AND toLower(s.name) CONTAINS toLower("\1") RETURN', query)
        
        # 5.3) Fix missing closing parenthesis in complex WHERE clauses
        # Pattern: CONTAINS toLower("term" AND -> CONTAINS toLower("term") AND
        query = re.sub(r'CONTAINS toLower\("([^"]+)"\s+AND\s+', r'CONTAINS toLower("\1") AND ', query)
        
        # 5.4) Fix missing closing parenthesis before RETURN
        # Pattern: CONTAINS toLower("term" RETURN -> CONTAINS toLower("term") RETURN
        query = re.sub(r'CONTAINS toLower\("([^"]+)"\s+RETURN', r'CONTAINS toLower("\1") RETURN', query)
        
        # 5.5) Fix the specific pattern from the error log
        # Pattern: toLower("Cognitive disability" RETURN -> toLower("Cognitive disability") RETURN
        query = re.sub(r'toLower\("([^"]+)"\s+RETURN', r'toLower("\1") RETURN', query)
        
        # 5.6) Fix nested CONTAINS without closing parenthesis before AND/RETURN
        # Pattern: CONTAINS toLower("term" AND -> CONTAINS toLower("term") AND
        query = re.sub(r'CONTAINS\s+toLower\("([^"]+)"\s+(AND|RETURN)', r'CONTAINS toLower("\1") \2', query)
        
        # 5.7) Fix complex WHERE with AND + CONTAINS missing closing parenthesis
        # Pattern: AND toLower(s.name) CONTAINS toLower("term" RETURN -> AND toLower(s.name) CONTAINS toLower("term") RETURN
        query = re.sub(
            r'AND\s+toLower\(([^)]+)\)\s+CONTAINS\s+toLower\("([^"]+)"\s+(AND|RETURN)',
            r'AND toLower(\1) CONTAINS toLower("\2") \3',
            query
        )
        
        # 6) Add synonym expansion for critical SEN terms
        query = self._expand_sen_synonyms(query)
        
        # 7) ChatGPT's specific fixes for remaining issues
        
        # A) Global Alias harmonization: fix any stray p.* to m.* when m is bound
        if re.search(r'\(\s*m\s*:\s*PedagogicalMethodology\s*\)', query) and not re.search(r'\(\s*p\s*:', query):
            # Replace ANY occurrence of 'p.' (not just right after RETURN) with 'm.'
            query = re.sub(r'(?<![A-Za-z0-9_])p\.', 'm.', query)

        # Optional: if LearningResource is the target, normalize p.* -> lr.*
        if re.search(r'\(\s*lr\s*:\s*LearningResource\s*\)', query) and not re.search(r'\(\s*p\s*:', query):
            query = re.sub(r'(?<![A-Za-z0-9_])p\.', 'lr.', query)
        
        # B) Cognitive OR-chain scrub: if we already have CONTAINS("cognitive disability"), drop any trailing OR s.name = "...Cognitive..."
        # First: Insert CONTAINS for cognitive disability patterns
        query = re.sub(
            r'WHERE\s+s\.name\s*=\s*"[^"]*[Cc]ognitive[^"]*"(?:\s+OR\s+s\.name\s*=\s*"[^"]*[Cc]ognitive[^"]*")*',
            'WHERE toLower(s.name) CONTAINS toLower("cognitive disability")',
            query
        )
        
        # Second: Clean up any remaining cognitive OR fragments after CONTAINS
        if re.search(r'toLower\(s\.name\)\s*CONTAINS\s*toLower\("cognitive disability"\)', query, flags=re.IGNORECASE):
            # Remove " OR s.name = "<anything with Cognitive ...>"
            query = re.sub(r'\s*OR\s*s\.name\s*=\s*"[^"]*(?i:cognitive)[^"]*"', '', query, flags=re.IGNORECASE)
            # Also remove accidental concatenation like ...CONTAINS("...") s.name = ...
            query = re.sub(r'\)\s*s\.name\s*=\s*"[^"]*(?i:cognitive)[^"]*"', ')', query, flags=re.IGNORECASE)
        
        # C) Fix Physical disability on wrong property
        query = re.sub(
            r's\.category\s*=\s*"(?i:physical disability)"',
            'toLower(s.name) CONTAINS toLower("Physical disability")',
            query,
            flags=re.IGNORECASE
        )
        
        # D) Relax LearningResource category when too strict
        # If we ask LR with a hard category filter, prefer a tolerant match
        query = re.sub(
            r'lr\.category\s*=\s*"(?i:assessment and evaluation)"',
            'toLower(lr.category) CONTAINS toLower("assessment")',
            query,
            flags=re.IGNORECASE
        )
        
        # E) Final hygiene: collapse duplicate spaces, fix WHERE ... AND/OR punctuation issues
        query = re.sub(r'\s+', ' ', query).strip()
        query = re.sub(r'WHERE\s+(AND|OR)\s+', 'WHERE ', query, flags=re.IGNORECASE)
        query = re.sub(r'\s+(AND|OR)\s+(RETURN|LIMIT)\b', r' \2', query, flags=re.IGNORECASE)
        
        return query
    
    def _expand_sen_synonyms(self, query: str) -> str:
        """Expand neuroscience terms with synonyms for better matching (Italian + English)"""
        import re
        
        # Comprehensive Italian-English synonym mappings for neuroscience knowledge graph
        NEUROSCIENCE_SYNONYMS = {
            # Motivation (Italian + English)
            "motivazione intrinseca": ["Intrinsic", "intrinsic motivation", "internal motivation"],
            "motivazione estrinseca": ["Extrinsic", "extrinsic motivation", "external motivation"],
            "intrinsic motivation": ["Intrinsic", "internal motivation", "autonomous motivation"],
            "extrinsic motivation": ["Extrinsic", "external motivation", "reward-based motivation"],
            "motivazione": ["Intrinsic", "Extrinsic", "motivation"],
            
            # Stress (Italian + English)
            "stress positivo": ["stress (positive or negative)", "positive stress", "eustress"],
            "stress negativo": ["stress (positive or negative)", "negative stress", "distress"],
            "eustress": ["stress (positive or negative)", "beneficial stress"],
            "distress": ["stress (positive or negative)", "harmful stress"],
            "stress": ["stress (positive or negative)"],
            
            # Mindset (Italian + English)
            "mentalità di crescita": ["Growth", "growth mindset", "malleable mindset"],
            "mentalità fissa": ["Fixed", "fixed mindset", "entity mindset"],
            "mindset crescita": ["Growth", "growth mindset"],
            "mindset fisso": ["Fixed", "fixed mindset"],
            "growth mindset": ["Growth", "malleable mindset", "incremental mindset"],
            "fixed mindset": ["Fixed", "entity mindset", "static mindset"],
            
            # Metacognition (Italian + English)
            "metacognizione": ["Metacognition", "self-awareness", "thinking about thinking"],
            "metacognition": ["Metacognition", "MetacognitiveMonitoring", "self-awareness"],
            
            # Memory (Italian + English)
            "memoria di lavoro": ["short-term / working memory", "working memory", "WM"],
            "memoria a lungo termine": ["long-term memory", "LTM"],
            "working memory": ["short-term / working memory", "WM"],
            "memoria": ["short-term / working memory", "long-term memory"],
            "memory": ["Memory", "WorkingMemory", "LongTermMemory"],
            
            # Executive Functions (Italian + English)
            "funzioni esecutive": ["Self-regulation", "Planning", "executive functions", "cognitive control"],
            "executive function": ["ExecutiveFunctions", "cognitive control", "EF"],
            
            # Attention (Italian + English)
            "attenzione": ["selective", "divided", "Sustained", "Focused", "attention", "focus"],
            "attenzione selettiva": ["selective", "selective attention"],
            "attention": ["Attention", "focus", "concentration", "selective attention"],
            
            # Emotions (Italian + English)
            "emozioni": ["stress (positive or negative)", "emotions"],
            "emozioni positive": ["stress (positive or negative)", "positive emotions"],
            "emozioni negative": ["stress (positive or negative)", "negative emotions"],
            "emotions": ["PositiveEmotions", "NegativeEmotions"],
            
            # Creativity (Italian + English)
            "creatività": ["Creativity", "divergent thinking", "innovative thinking"],
            "creativity": ["Creativity", "divergent thinking"],
            
            # Critical Thinking (Italian + English)
            "pensiero critico": ["Critical thinking", "analytical thinking"],
            "critical thinking": ["CriticalThinking"]
        }
        
        # Look for WHERE clauses with single term matching
        pattern = r'WHERE\s+toLower\((s|n|m|e|a)\.name\)\s*=\s*toLower\("([^"]+)"\)'

        def expand_synonyms(match):
            alias = match.group(1)
            term = match.group(2).lower()

            # Check if this term has synonyms
            for key, synonyms in NEUROSCIENCE_SYNONYMS.items():
                if key in term or any(term in syn.lower() for syn in synonyms):
                    # Create IN clause with all synonyms for broad matching
                    synonym_values = ', '.join([f'toLower("{syn}")' for syn in synonyms])
                    return f'WHERE toLower({alias}.name) IN [{synonym_values}]'

            # If no synonyms found, use tolerant CONTAINS matching
            return f'WHERE toLower({alias}.name) CONTAINS toLower("{match.group(2)}")'
        
        return re.sub(pattern, expand_synonyms, query)
    
    def _validate_cypher(self, cypher_query: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated Cypher query syntax"""
        if not cypher_query or cypher_query.strip() == "":
            return False, "Empty query"
        
        try:
            # Basic syntax validation
            with self.schema_extractor.driver.session() as session:
                # Try to explain the query (doesn't execute it)
                session.run(f"EXPLAIN {cypher_query}")
                return True, None
                
        except Exception as e:
            return False, str(e)
    
    def execute_query(self, cypher_query: str) -> Tuple[List[Dict], Optional[str]]:
        """Execute the Cypher query and return results"""
        try:
            with self.schema_extractor.driver.session() as session:
                result = session.run(cypher_query)
                records = []
                for record in result:
                    records.append(dict(record))
                return records, None
                
        except Exception as e:
            logger.error(f"Error executing Cypher query: {e}")
            return [], str(e)
    
    def close(self):
        """Close all connections"""
        self.schema_extractor.close()

class Text2CypherPipeline:
    """Complete pipeline for text2cypher conversion and execution"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str):
        self.converter = Text2CypherConverter(neo4j_uri, neo4j_user, neo4j_password, openai_api_key)
    
    def process_question(self, question: str, execute: bool = True) -> Dict:
        """Process a natural language question end-to-end"""
        # Convert to Cypher
        cypher_query, metadata = self.converter.convert(question)
        
        result = {
            "question": question,
            "cypher_query": cypher_query,
            "metadata": metadata,
            "results": [],
            "execution_error": None
        }
        
        # Execute if requested and query is valid
        if execute and metadata.get("is_valid", False):
            results, execution_error = self.converter.execute_query(cypher_query)
            
            # Retry-on-empty orchestrator: if query is valid but returns 0 rows, try widened version
            if not results and not execution_error:
                # Create a more tolerant version of the query
                widened_query = cypher_query
                
                # Make name matching more tolerant
                widened_query = re.sub(r' s\.name IN \[', ' toLower(s.name) IN [', widened_query)
                widened_query = re.sub(r' s\.name = ', ' toLower(s.name) CONTAINS toLower(', widened_query)
                widened_query = re.sub(r' lr\.category = ', ' toLower(lr.category) CONTAINS toLower(', widened_query)
                
                # Only retry if the query actually changed
                if widened_query != cypher_query:
                    retry_results, retry_error = self.converter.execute_query(widened_query)
                    if retry_results:  # If retry succeeded, use its results
                        results = retry_results
                        execution_error = retry_error
                        result["retry_query"] = widened_query
                        result["retry_success"] = True
                    else:
                        result["retry_query"] = widened_query
                        result["retry_success"] = False
            
            result["results"] = results
            result["execution_error"] = execution_error
        
        return result
    
    def close(self):
        """Close all connections"""
        self.converter.close()

# Example usage and testing functions
def main():
    """Example usage of the Text2Cypher module"""
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "your_password"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    # Initialize pipeline
    pipeline = Text2CypherPipeline(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)
    
    # Test questions
    test_questions = [
        "What teaching methods help students with ADHD?",
        "What methodologies should be avoided for students with autism?",
        "How many different teaching methodologies are available?",
        "What technologies support interactive learning?",
        "Show me learning environment factors for better acoustics"
    ]
    
    try:
        for question in test_questions:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print('='*60)
            
            result = pipeline.process_question(question)
            
            print(f"Generated Cypher: {result['cypher_query']}")
            print(f"Valid: {result['metadata'].get('is_valid', False)}")
            
            if result['execution_error']:
                print(f"Execution Error: {result['execution_error']}")
            else:
                print(f"Results ({len(result['results'])} records):")
                for i, record in enumerate(result['results'][:3]):  # Show first 3
                    print(f"  {i+1}. {record}")
                if len(result['results']) > 3:
                    print(f"  ... and {len(result['results']) - 3} more")
    
    finally:
        pipeline.close()

if __name__ == "__main__":
    main()
