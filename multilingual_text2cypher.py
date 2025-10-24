#!/usr/bin/env python3
"""
multilingual_text2cypher.py - Multilingual text2cypher for Neuroscience Knowledge Graph
Maps Italian teacher queries to neuroscience concepts in Neo4j
"""

import re
from typing import Dict, List, Tuple, Optional
from text2cypher import Text2CypherPipeline
from config import config
import logging

logger = logging.getLogger(__name__)

class MultilingualText2Cypher:
    """Multilingual text2cypher for Italian neuroscience educational queries"""
    
    def __init__(self):
        # Use config.py which loads from .env
        self.pipeline = Text2CypherPipeline(
            config.neo4j.uri,
            config.neo4j.user,
            config.neo4j.password,
            config.openai.api_key
        )
        
        # Italian → English mappings for NEUROSCIENCE knowledge graph
        # Based on actual node names in kg_neuro_neo4j.json
        self.italian_terms = {
            # ============ MOTIVATION ============
            # Label: MotivationalModulation, IntrinsicMotivation, ExtrinsicMotivation
            "motivazione intrinseca": "Intrinsic motivation",
            "intrinseca": "Intrinsic",
            "motivazione estrinseca": "Extrinsic motivation",
            "estrinseca": "Extrinsic",
            "motivazione": "motivation",
            "motivazionale": "motivational",
            
            # ============ MINDSET ============
            # Label: Mindset, GrowthMindset, FixedMindset
            "mentalità di crescita": "Growth mindset",
            "mentalità fissa": "Fixed mindset",
            "mindset di crescita": "Growth mindset",
            "mindset fisso": "Fixed mindset",
            "crescita": "Growth",
            "fisso": "Fixed",
            "mentalità": "mindset",
            
            # ============ STRESS & EMOTIONS ============
            # Label: Emotions, PositiveStressEustress, NegativeStressDistress
            "stress positivo": "positive stress",
            "stress negativo": "negative stress",
            "eustress": "eustress",
            "distress": "distress",
            "stress": "stress",
            "emozioni positive": "positive emotions",
            "emozioni negative": "negative emotions",
            "emozioni": "emotions",
            "ansia": "anxiety",
            
            # ============ ATTENTION ============
            # Label: Attention (various types)
            "attenzione": "attention",
            "attenzione selettiva": "selective attention",
            "attenzione divisa": "divided attention",
            "attenzione sostenuta": "sustained attention",
            "attenzione focalizzata": "focused attention",
            "concentrazione": "concentration",
            "focus": "focus",
            
            # ============ MEMORY ============
            # Label: Memory, WorkingMemory, LongTermMemory
            "memoria di lavoro": "working memory",
            "memoria a breve termine": "short-term memory",
            "memoria a lungo termine": "long-term memory",
            "memoria": "memory",
            "codifica": "encoding",
            "consolidamento": "consolidation",
            "recupero": "retrieval",
            
            # ============ EXECUTIVE FUNCTIONS ============
            # Label: ExecutiveFunctions, executivecontrol
            "funzioni esecutive": "executive functions",
            "controllo esecutivo": "executive control",
            "autoregolazione": "self-regulation",
            "pianificazione": "planning",
            "controllo inibitorio": "inhibitory control",
            "flessibilità cognitiva": "cognitive flexibility",
            
            # ============ METACOGNITION ============
            # Label: Metacognition
            "metacognizione": "metacognition",
            "consapevolezza metacognitiva": "metacognitive awareness",
            "monitoraggio metacognitivo": "metacognitive monitoring",
            "autoriflessione": "self-reflection",
            
            # ============ LEARNING & COGNITION ============
            # Label: CognitiveLoad, LearningOutcomes, etc.
            "carico cognitivo": "cognitive load",
            "apprendimento": "learning",
            "apprendimento profondo": "deep learning",
            "comprensione profonda": "deep understanding",
            "elaborazione profonda": "deep processing",
            "elaborazione": "processing",
            
            # ============ CREATIVITY ============
            # Label: Creativity
            "creatività": "creativity",
            "pensiero divergente": "divergent thinking",
            "pensiero creativo": "creative thinking",
            "innovazione": "innovation",
            
            # ============ CRITICAL THINKING ============
            # Label: CriticalThinking
            "pensiero critico": "critical thinking",
            "ragionamento": "reasoning",
            "analisi": "analysis",
            "valutazione": "evaluation",
            "problem solving": "problem solving",
            "risoluzione problemi": "problem solving",
            
            # ============ NEUROPLASTICITY & BRAIN ============
            # Label: Neuroplasticity, BrainAdaptability
            "neuroplasticità": "neuroplasticity",
            "plasticità cerebrale": "brain plasticity",
            "adattabilità cerebrale": "brain adaptability",
            
            # ============ TEACHING & LEARNING CONTEXT ============
            "studenti": "students",
            "insegnamento": "teaching",
            "insegnare": "teach",
            "apprendere": "learn",
            "facilitare": "facilitate",
            "supportare": "support",
            "migliorare": "improve",
            "sviluppare": "develop",
            "incoraggiare": "encourage",
            "promuovere": "promote",
            
            # ============ COMMON QUESTION WORDS ============
            "cos'è": "what is",
            "qual è": "what is",
            "quali sono": "what are",
            "come": "how",
            "perché": "why",
            "quando": "when",
            "dove": "where",
            "differenza": "difference",
            "tra": "between",
            "relazione": "relationship",
            "collegamento": "connection",
            "influenza": "influence",
            "effetto": "effect",
            "impatto": "impact",
        }
        
        # Query type patterns for neuroscience topics
        self.query_patterns = {
            "motivation_queries": [
                "motivazione", "intrinseca", "estrinseca", "motivation"
            ],
            "stress_queries": [
                "stress", "ansia", "anxiety", "eustress", "distress"
            ],
            "mindset_queries": [
                "mentalità", "mindset", "crescita", "growth", "fisso", "fixed"
            ],
            "memory_queries": [
                "memoria", "memory", "working", "lungo termine", "codifica"
            ],
            "attention_queries": [
                "attenzione", "attention", "focus", "concentrazione"
            ],
            "metacognition_queries": [
                "metacognizione", "metacognition", "autoriflessione"
            ],
            "executive_function_queries": [
                "funzioni esecutive", "executive", "autoregolazione", "pianificazione"
            ],
            "creativity_queries": [
                "creatività", "creativity", "pensiero divergente", "innovazione"
            ],
            "critical_thinking_queries": [
                "pensiero critico", "critical thinking", "ragionamento", "problem solving"
            ]
        }
    
    def detect_language(self, query: str) -> str:
        """Detect if query is in Italian or English"""
        italian_indicators = [
            # Question words
            "come", "cosa", "quali", "qual", "che", "perché", "quando", "dove",
            # Common verbs
            "posso", "sono", "può", "hanno", "è",
            # Common nouns
            "studenti", "apprendimento", "insegnare", "differenza"
        ]
        
        query_lower = query.lower()
        italian_count = sum(1 for word in italian_indicators if word in query_lower)
        
        return "italian" if italian_count >= 1 else "english"
    
    def detect_query_type(self, query: str) -> List[str]:
        """Detect the type of neuroscience query"""
        query_lower = query.lower()
        detected_types = []
        
        for pattern_type, keywords in self.query_patterns.items():
            if any(keyword.lower() in query_lower for keyword in keywords):
                detected_types.append(pattern_type)
        
        return detected_types if detected_types else ["general"]
    
    def enhance_italian_query(self, italian_query: str) -> str:
        """Translate Italian neuroscience terms to English for Neo4j matching"""
        enhanced_query = italian_query
        
        # Replace Italian terms with English equivalents (longer terms first)
        sorted_terms = sorted(self.italian_terms.items(), key=lambda x: len(x[0]), reverse=True)
        
        for italian_term, english_term in sorted_terms:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(italian_term), re.IGNORECASE)
            enhanced_query = pattern.sub(english_term, enhanced_query)
        
        # Detect query types for context
        query_types = self.detect_query_type(italian_query)
        
        # Add neuroscience context prefix
        if "motivation_queries" in query_types:
            context_prefix = "Neuroscience query about motivation: "
        elif "stress_queries" in query_types:
            context_prefix = "Neuroscience query about stress and emotions: "
        elif "mindset_queries" in query_types:
            context_prefix = "Neuroscience query about mindset: "
        elif "memory_queries" in query_types:
            context_prefix = "Neuroscience query about memory: "
        elif "attention_queries" in query_types:
            context_prefix = "Neuroscience query about attention: "
        elif "executive_function_queries" in query_types:
            context_prefix = "Neuroscience query about executive functions: "
        else:
            context_prefix = "Neuroscience educational query: "
        
        enhanced_query = context_prefix + enhanced_query
        
        return enhanced_query
    
    def process_query(self, query: str, execute: bool = True) -> Dict:
        """Process query with neuroscience-specific multilingual support"""
        original_query = query
        language = self.detect_language(query)
        query_types = self.detect_query_type(query)
        
        logger.info(f"Processing {language} query with types {query_types}: {query[:50]}...")
        
        # Enhance Italian queries for better Neo4j mapping
        if language == "italian":
            enhanced_query = self.enhance_italian_query(query)
            logger.info(f"Enhanced query: {enhanced_query[:100]}...")
        else:
            enhanced_query = query
        
        # Process with text2cypher pipeline
        result = self.pipeline.process_question(enhanced_query, execute=execute)
        
        # Add multilingual metadata
        result.update({
            "original_query": original_query,
            "detected_language": language,
            "detected_query_types": query_types,
            "enhanced_query": enhanced_query if language == "italian" else None,
            "multilingual_processing": True,
            "neuroscience_context": True
        })
        
        return result
    
    def batch_process_queries(self, queries: List[str], execute: bool = False) -> List[Dict]:
        """Process multiple neuroscience queries efficiently"""
        results = []
        
        # Statistics tracking
        language_stats = {"italian": 0, "english": 0}
        type_stats = {}
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            result = self.process_query(query, execute=execute)
            results.append(result)
            
            # Update statistics
            lang = result.get('detected_language', 'unknown')
            language_stats[lang] = language_stats.get(lang, 0) + 1
            
            for query_type in result.get('detected_query_types', []):
                type_stats[query_type] = type_stats.get(query_type, 0) + 1
        
        # Log statistics
        logger.info(f"Batch processing complete:")
        logger.info(f"  Language distribution: {language_stats}")
        logger.info(f"  Query type distribution: {type_stats}")
        
        return results
    
    def close(self):
        """Close pipeline connections"""
        self.pipeline.close()

# Test function for neuroscience queries
def test_neuroscience_italian_queries():
    """Test Italian neuroscience queries"""
    
    # Validate configuration
    is_valid, errors = config.validate()
    if not is_valid:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    
    # Neuroscience test queries in Italian
    test_queries = [
        # Motivation
        "Qual è la differenza tra motivazione intrinseca ed estrinseca?",
        "Come posso incoraggiare la motivazione intrinseca negli studenti?",
        
        # Mindset
        "Cos'è la mentalità di crescita?",
        "Qual è la differenza tra mentalità di crescita e mentalità fissa?",
        
        # Stress
        "Lo stress può essere positivo per l'apprendimento?",
        "Come influisce lo stress sull'apprendimento?",
        
        # Memory
        "Cos'è la memoria di lavoro?",
        "Come posso migliorare la memoria di lavoro degli studenti?",
        
        # Attention
        "Come funziona l'attenzione selettiva?",
        "Quali fattori influenzano l'attenzione degli studenti?",
        
        # Executive Functions
        "Cosa sono le funzioni esecutive?",
        "Come posso supportare lo sviluppo delle funzioni esecutive?",
        
        # Metacognition
        "Cos'è la metacognizione e perché è importante?",
        
        # Critical Thinking
        "Come posso sviluppare il pensiero critico negli studenti?"
    ]
    
    multilingual_processor = MultilingualText2Cypher()
    
    try:
        print("🧠 Testing Neuroscience Italian Teacher Queries")
        print("=" * 100)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🇮🇹 TEST {i}: {query}")
            print("-" * 80)
            
            result = multilingual_processor.process_query(query, execute=False)
            
            print(f"Language: {result['detected_language']}")
            print(f"Query Types: {', '.join(result.get('detected_query_types', ['general']))}")
            if result.get('enhanced_query'):
                print(f"Enhanced: {result['enhanced_query'][:120]}...")
            print(f"Cypher: {result['cypher_query'][:150]}...")
            print(f"Valid: {result['metadata'].get('is_valid', False)}")
            
            if result['metadata'].get('validation_error'):
                print(f"⚠️ Error: {result['metadata']['validation_error'][:100]}...")
    
    finally:
        multilingual_processor.close()

if __name__ == "__main__":
    test_neuroscience_italian_queries()
