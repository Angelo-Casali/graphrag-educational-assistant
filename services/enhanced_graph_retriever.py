#!/usr/bin/env python3
"""
enhanced_graph_retriever.py - Enhanced graph schema retrieval and management
Compatible with AuraDB and based on kg_neuro_neo4j.json
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from neo4j import GraphDatabase
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class SchemaInfo:
    """Enhanced schema information for neuroscience knowledge graph"""
    node_labels: List[str]
    relationship_types: List[str]
    node_properties: Dict[str, List[str]]
    sample_nodes: Dict[str, List[Dict]]
    node_categories: Dict[str, List[str]]
    relationship_mapping: Dict[str, str]

class EnhancedGraphRetriever:
    """Enhanced graph schema retriever for AuraDB compatibility"""

    def __init__(self, uri: str = None, user: str = None, password: str = None,
                 use_json_schema: bool = True, json_file: str = "kg_neuro_neo4j.json"):
        """
        Initialize the enhanced graph retriever
        
        Args:
            uri: Neo4j/AuraDB URI
            user: Database username
            password: Database password
            use_json_schema: Whether to use JSON schema instead of database query
            json_file: Path to JSON schema file
        """
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USERNAME")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.use_json_schema = use_json_schema
        self.json_file = json_file
        
        # Initialize schema info
        self.schema_info = None
        self.driver = None
        
        # Load schema
        if self.use_json_schema:
            self._load_schema_from_json()
        else:
            self._load_schema_from_database()
    
    def _load_schema_from_json(self):
        """Load schema from concepts_neo4j.json file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract schema information
            node_labels = []
            node_properties = {}
            sample_nodes = {}
            node_categories = {}
            relationship_types = set()
            relationship_mapping = {}
            
            # Process nodes
            for node in data.get('nodes', []):
                label = node.get('label')
                if label and label not in node_labels:
                    node_labels.append(label)
                
                # Store properties
                if label:
                    properties = list(node.get('properties', {}).keys())
                    node_properties[label] = properties
                    
                    # Store sample node
                    if label not in sample_nodes:
                        sample_nodes[label] = []
                    sample_nodes[label].append(node.get('properties', {}))
                    
                    # Store category
                    category = node.get('category', '')
                    if label not in node_categories:
                        node_categories[label] = []
                    if category and category not in node_categories[label]:
                        node_categories[label].append(category)
            
            # Process relationships
            for rel in data.get('relationships', []):
                rel_type = rel.get('type')
                if rel_type:
                    relationship_types.add(rel_type)
                    
                    # Map relationship types to standard ones
                    standard_mapping = self._map_relationship_type(rel_type)
                    relationship_mapping[rel_type] = standard_mapping
            
            self.schema_info = SchemaInfo(
                node_labels=node_labels,
                relationship_types=list(relationship_types),
                node_properties=node_properties,
                sample_nodes=sample_nodes,
                node_categories=node_categories,
                relationship_mapping=relationship_mapping
            )
            
            logger.info(f"Loaded schema from JSON: {len(node_labels)} node labels, {len(relationship_types)} relationship types")
            
        except Exception as e:
            logger.error(f"Error loading schema from JSON: {e}")
            raise
    
    def _load_schema_from_database(self):
        """Load schema directly from AuraDB"""
        try:
            if not all([self.uri, self.user, self.password]):
                raise ValueError("Database credentials required for direct schema loading")
            
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            
            with self.driver.session() as session:
                # Get node labels
                node_labels = session.run("CALL db.labels()").value()
                
                # Get relationship types
                relationship_types = session.run("CALL db.relationshipTypes()").value()
                
                # Get node properties for each label
                node_properties = {}
                sample_nodes = {}
                node_categories = {}
                
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
                
                self.schema_info = SchemaInfo(
                    node_labels=node_labels,
                    relationship_types=relationship_types,
                    node_properties=node_properties,
                    sample_nodes=sample_nodes,
                    node_categories=node_categories,
                    relationship_mapping={}
                )
                
                logger.info(f"Loaded schema from database: {len(node_labels)} node labels, {len(relationship_types)} relationship types")
                
        except Exception as e:
            logger.error(f"Error loading schema from database: {e}")
            raise
    
    def _map_relationship_type(self, rel_type: str) -> str:
        """Map relationship types to semantically meaningful standard ones"""
        mapping = {
            # ALIGNS_WITH (neutral recommendation)
            'SUGGESTS': 'ALIGNS_WITH',
            'OFFERS': 'ALIGNS_WITH',
            
            # SUPPORTS (positive effect)
            'SUPPORTS': 'SUPPORTS',
            'FACILITATES': 'SUPPORTS',
            'ENABLES': 'SUPPORTS',
            'PROMOTES': 'SUPPORTS',
            'IMPROVES': 'SUPPORTS',
            'FOSTERS': 'SUPPORTS',
            'STIMULATES': 'SUPPORTS',
            
            # HINDERS (negative effect)
            'NO_SUGGESTS': 'HINDERS',
            'HINDERS': 'HINDERS',
            'LIMITS': 'HINDERS',
            'REDUCES': 'HINDERS',
            'IMPAIRS': 'HINDERS',
            'DISRUPTS': 'HINDERS',
            'REMOVES': 'HINDERS',
            'DISTRACTS': 'HINDERS',
            'IS_NOT_SUITABLE_FOR': 'HINDERS',
            'NO_SUGGEST': 'HINDERS',
            
            # CAUSES (causal risk)
            'CAUSES': 'CAUSES',
            'CAUSES_DISCOMFORT_IN': 'CAUSES',
            'INCREASES_RISK_OF': 'CAUSES',
            'MAY_INCREASE': 'CAUSES',
            'INCREASES': 'CAUSES',
            
            # Additional mappings for specific educational contexts
            'IMPROVES': 'SUPPORTS',
            'PROMOTES': 'SUPPORTS',
            'STIMULATES': 'SUPPORTS',
            'LIMITS': 'HINDERS',
            'REDUCES': 'HINDERS',
            'DISRUPTS': 'HINDERS',
            'DISTRACTS': 'HINDERS'
        }
        
        return mapping.get(rel_type, 'ALIGNS_WITH')  # Default to neutral recommendation
    
    def get_schema_description(self) -> str:
        """Build comprehensive schema description for LLM prompts"""
        if not self.schema_info:
            return "No schema available"
        
        desc = "DATABASE SCHEMA:\n"
        desc += "=" * 50 + "\n\n"
        
        desc += "NODE LABELS:\n"
        for label in self.schema_info.node_labels:
            desc += f"- {label}\n"
            if label in self.schema_info.node_properties:
                props = self.schema_info.node_properties[label]
                desc += f"  Properties: {', '.join(props)}\n"
            
            # Add sample data
            if label in self.schema_info.sample_nodes and self.schema_info.sample_nodes[label]:
                sample = self.schema_info.sample_nodes[label][0]
                # Format sample data safely
                sample_str = ", ".join([f"{k}: {v}" for k, v in sample.items() if k != 'id'])
                desc += f"  Example: {sample_str}\n"
            
            # Add categories
            if label in self.schema_info.node_categories:
                categories = self.schema_info.node_categories[label]
                desc += f"  Categories: {', '.join(categories)}\n"
            desc += "\n"
        
        desc += "RELATIONSHIP TYPES:\n"
        for rel_type in self.schema_info.relationship_types:
            desc += f"- {rel_type}\n"
        desc += "\n"
        
        desc += "STANDARD RELATIONSHIP MAPPINGS:\n"
        for original, mapped in self.schema_info.relationship_mapping.items():
            desc += f"- {original} → {mapped}\n"
        
        desc += "\nSEMANTIC RELATIONSHIP MEANINGS:\n"
        desc += "- ALIGNS_WITH: Neutral recommendation or suggestion\n"
        desc += "- SUPPORTS: Positive effect or facilitation\n"
        desc += "- HINDERS: Negative effect or obstruction\n"
        desc += "- CAUSES: Causal relationship or risk\n"
        
        return desc
    
    def get_few_shot_examples(self) -> str:
        """Create few-shot examples based on actual schema data"""
        examples = """
Question: "How does working memory affect attention?"
Cypher: MATCH (wm:WorkingMemory)-[r:AFFECTS]->(a:Attention) RETURN wm.name, a.name, type(r) LIMIT 10

Question: "What factors impair sustained attention?"
Cypher: MATCH (f:CognitiveLoad)-[r:INCREASES]->(a:Attention) WHERE a.name = 'sustained' RETURN f.name, f.concept LIMIT 10

Question: "How many types of attention are represented in the graph?"
Cypher: MATCH (a:Attention) RETURN COUNT(a) as attention_count

Question: "What cognitive processes support memory consolidation?"
Cypher: MATCH (p:Consolidation)-[r:SUPPORTED_BY]->(b:Neuroplasticity) RETURN p.name, b.name, type(r) LIMIT 10

Question: "Show me how emotional arousal affects memory formation"
Cypher: MATCH (e:AmygdalahippocampusInteraction)-[r:STRENGTHENED_THROUGH]->(m:LongtermMemoryFormation) WHERE e.name CONTAINS 'arousal' RETURN e.name, m.name LIMIT 10

Question: "What factors cause cognitive fatigue during attention tasks?"
Cypher: MATCH (f:AffectiveBiologicalConstraint)-[r:IMPAIRED_BY]->(a:AttentionalControl) WHERE f.name CONTAINS 'fatigue' RETURN f.name, a.name LIMIT 10

Question: "What role does sleep play in neuroplasticity?"
Cypher: MATCH (s:PsychobiologicalFactor)-[r:SUPPORTED_BY]->(n:Neuroplasticity) WHERE s.name CONTAINS 'sleep' RETURN s.name, n.name LIMIT 10

Question: "What environmental factors impair executive control?"
Cypher: MATCH (e:EnvironmentalFactor)-[r:IS_IMPAIRED_BY]->(ex:ExecutiveControl) WHERE ex.name = 'Self-regulation' RETURN e.name, ex.name LIMIT 10

Question: "What cognitive strategies help with divided attention?"
Cypher: MATCH (cs:ResourceAllocation)-[r:DEPENDS_ON]->(a:Attention) WHERE a.name = 'divided' RETURN cs.name, a.name LIMIT 10

Question: "What factors increase cognitive load during multitasking?"
Cypher: MATCH (m:CognitiveLoad)-[r:INCREASES]->(t:Attention) WHERE t.name CONTAINS 'multitasking' RETURN m.name, t.name LIMIT 10
"""
        return examples.strip()
    
    def get_italian_mappings(self) -> Dict[str, str]:
        """Get Italian to English mappings for educational terms"""
        mappings = {
            # Special Educational Needs - Map to actual LearnerProfile nodes
            "ipovedenti": "Blind",
            "disabilità uditive": "Deaf", 
            "disabilità": "Physical disability",
            "dislessia": "Dyslexia",
            "ADHD": "Adhd",
            "deficit di attenzione": "Attention Deficit",
            "autismo": "Autism spectrum disorder",
            "motivazione": "Lack of motivation",
            "eccellenza": "Excellence in some or all subjects",
            "difficoltà cognitive": "Cognitive disability [mild, moderate, severe]",
            
            # UDL (Universal Design for Learning)
            "UDL": "Universal Design for Learning",
            "Universal Design for Learning": "Universal Design for Learning",
            "progettazione universale": "universal design",
            "linee guida UDL": "UDL guidelines",
            "principi UDL": "UDL principles",
            "strategie UDL": "UDL strategies",
            
            # Teaching and Learning Methods - Map to actual PedagogicalMethodology nodes
            "apprendimento cooperativo": "Cooperative learning",
            "flipped classroom": "Flipped Classroom",
            "game based learning": "GameBasedLearning",
            "debate": "Debate",
            "project based learning": "Project based learning",
            "role playing": "Role Playing, Debate",
            "station rotation": "Station Rotation",
            "stem": "Stem",
            "peertopeereducation": "Peertopeereducation",
            
            # Teaching Approaches - Map to actual PedagogicalApproach nodes
            "lezioni frontali": "Long frontal lessons",
            "lezioni frontali lunghe": "Long Frontal lessons",
            "supporti visivi, alternative bilingue": "Visual supports, bilingual alternatives",
            
            # Class Climate - Map to actual ClassClimate nodes
            "coesivo": "Cohesive",
            "diviso in gruppi": "Split in groups",
            "con elementi disturbanti": "With disruptive elements",
            "motivato": "Motivated",
            "divario di genere": "Gender gap",
            
            # Environmental Factors - Map to actual LearningSetting nodes
            "illuminazione": "Lighting",
            "colori": "Colour",
            "acustica": "Acoustics",
            "arredi": "Furniture",
            "texture": "Textures",
            "odori": "Smells",
            
            # Technologies - Map to actual Technologie nodes
            "lavagna interattiva": "Interactive Board",
            "wifi": "WiFi Access",
            "computer lab": "Computer Lab",
            "notebook": "Notebook Access",
            
            # Generic terms
            "studenti": "students",
            "ragazzi": "students",
            "bambini": "children",
            "insegnare": "teach",
            "adattare": "adapt",
            "aiutare": "help",
            "favorire": "promote",
            "facilitare": "facilitate",
            "progettare": "design",
            "integrare": "integrate",
            "utilizzare": "use",
            "supportare": "support",
            "raggiungere": "achieve",
            "migliorare": "improve"
        }
        return mappings
    
    def close(self):
        """Close database connections"""
        if self.driver:
            self.driver.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Global schema instance for caching
_schema_instance = None

def get_enhanced_schema() -> EnhancedGraphRetriever:
    """Get cached enhanced schema instance"""
    global _schema_instance
    if _schema_instance is None:
        _schema_instance = EnhancedGraphRetriever()
    return _schema_instance

def get_schema_description() -> str:
    """Get schema description for LLM prompts"""
    schema = get_enhanced_schema()
    return schema.get_schema_description()

def get_few_shot_examples() -> str:
    """Get few-shot examples for LLM prompts"""
    schema = get_enhanced_schema()
    return schema.get_few_shot_examples()

def get_italian_mappings() -> Dict[str, str]:
    """Get Italian to English mappings"""
    schema = get_enhanced_schema()
    return schema.get_italian_mappings()

def get_schema():
    """Backward compatibility function - returns basic schema visualization"""
    try:
        # Try to get enhanced schema first
        schema = get_enhanced_schema()
        return schema.get_schema_description()
    except:
        # Fallback to basic schema if enhanced fails
        try:
            driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
            )
            with driver.session() as session:
                result = session.run("CALL db.schema.visualization()")
                return result.single().value()
        except Exception as e:
            print("❌ Neo4j schema error:", e)
            return None

def run_graph_query(cypher_query: str, quiet: bool = False) -> list:
    """Execute Cypher query against Neo4j database

    Args:
        cypher_query: Cypher query to execute
        quiet: If True, suppress success messages (for relationship validation)
    """
    if not cypher_query or not cypher_query.strip():
        if not quiet:
            print("⚠️ Empty Cypher query provided")
        return []

    try:
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

        with driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]
            if not quiet:
                print(f"✅ Successfully executed Cypher query, returned {len(records)} records")
            return records

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Neo4j query error: {error_msg}")

        # Provide more specific feedback based on error type (only for non-quiet mode)
        if not quiet:
            if "SyntaxError" in error_msg:
                print(f"💡 The generated Cypher query has a syntax error. Please check the query: {cypher_query}")
            elif "NoSuchProperty" in error_msg:
                print(f"💡 The query references a property that doesn't exist in the graph schema.")
            elif "NodeNotFound" in error_msg:
                print(f"💡 The query references a node label that doesn't exist in the graph schema.")

        return []
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    # Test the enhanced schema retriever
    try:
        with EnhancedGraphRetriever() as retriever:
            print("Schema Description:")
            print(retriever.get_schema_description())
            
            print("\nFew-shot Examples:")
            print(retriever.get_few_shot_examples())
            
            print("\nItalian Mappings:")
            mappings = retriever.get_italian_mappings()
            for italian, english in mappings.items():
                print(f"{italian} -> {english}")
                
    except Exception as e:
        print(f"Error: {e}")
