#!/usr/bin/env python3
"""
Explore the current neuroscience knowledge graph content
Extracts all node labels, relationships, and concepts to understand what Text2Cypher should know about
"""

from neo4j import GraphDatabase
import json
from typing import List, Dict, Any
from collections import defaultdict, Counter

def connect_to_neo4j():
    """Connect to Neo4j database"""
    try:
        from config import config
        driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.user, config.neo4j.password)
        )
        return driver
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return None

def explore_neuroscience_concepts(driver):
    """Comprehensive exploration of neuroscience knowledge graph"""

    print("🧠 EXPLORING NEUROSCIENCE KNOWLEDGE GRAPH")
    print("="*80)

    findings = {}

    with driver.session() as session:
        # 1. Get all node labels
        print("\n📝 PHASE 1: Node Labels")
        node_labels = session.run("CALL db.labels()").value()
        findings["node_labels"] = node_labels
        print(f"Found {len(node_labels)} node labels: {', '.join(node_labels)}")

        # 2. Get relationship types
        print("\n🔗 PHASE 2: Relationship Types")
        rel_types = session.run("CALL db.relationshipTypes()").value()
        findings["relationship_types"] = rel_types
        print(f"Found {len(rel_types)} relationship types: {', '.join(rel_types)}")

        # 3. Examine nodes by category
        print("\n🧬 PHASE 3: Neuroscience Concepts by Label")
        neuro_concepts = {}

        for label in node_labels:
            concepts = []
            result = session.run(f"MATCH (n:{label}) RETURN n.name as name, labels(n) as labels LIMIT 50")
            for record in result:
                concepts.append({
                    'name': record['name'],
                    'labels': record['labels']
                })

            neuro_concepts[label] = concepts
            print(f"\n{label} ({len(concepts)} concepts):")
            if concepts:
                # Show first 5 examples
                for i, concept in enumerate(concepts[:5], 1):
                    print(f"  {i}. {concept['name']} - labels: {concept['labels']}")
                if len(concepts) > 5:
                    print(f"  ... and {len(concepts) - 5} more")

        findings["concepts_by_label"] = neuro_concepts

        # 4. Get relationship patterns
        print("\n🌐 PHASE 4: Key Relationship Patterns")
        relationships = []
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN DISTINCT type(r) as relationship,
                   collect(DISTINCT labels(a)[0]) as source_types,
                   collect(DISTINCT labels(b)[0]) as target_types
            LIMIT 20
        """)

        for record in result:
            relationships.append({
                'relationship': record['relationship'],
                'source_types': record['source_types'],
                'target_types': record['target_types']
            })

        findings["relationship_patterns"] = relationships
        print("\nKey relationship patterns:")
        for rel in relationships:
            print(f"  {rel['relationship']}: {rel['source_types']} → {rel['target_types']}")

        # 5. Identify core neuroscience categories
        print("\n🎯 PHASE 5: Neuroscience Knowledge Areas")
        neuro_categories = defaultdict(list)
        motivation_count = 0
        emotion_count = 0
        cognition_count = 0
        learning_count = 0

        for label, concepts in neuro_concepts.items():
            # Categorize concepts
            if any(keyword in label.lower() for keyword in ['motivation', 'modulation']):
                neuro_categories['Motivation'].extend([c['name'] for c in concepts])
                motivation_count += len(concepts)
            elif any(keyword in label.lower() for keyword in ['emotion', 'stress']):
                neuro_categories['Emotions & Stress'].extend([c['name'] for c in concepts])
                emotion_count += len(concepts)
            elif any(keyword in label.lower() for keyword in ['cognit', 'memory', 'attention', 'executive', 'thinking']):
                neuro_categories['Cognition'].extend([c['name'] for c in concepts])
                cognition_count += len(concepts)
            elif any(keyword in label.lower() for keyword in ['learn', 'develop', 'pedagog', 'teaching']):
                neuro_categories['Learning & Development'].extend([c['name'] for c in concepts])
                learning_count += len(concepts)
            else:
                neuro_categories['Other'].extend([c['name'] for c in concepts])

        findings["neuroscience_categories"] = neuro_categories

        print(f"\nKnowledge Distribution:")
        print(f"  🧠 Motivation Concepts: {motivation_count}")
        print(f"  😊 Emotions & Stress: {emotion_count}")
        print(f"  🧮 Cognition: {cognition_count}")
        print(f"  📚 Learning & Development: {learning_count}")
        print(f"  ❓ Other: {len(neuro_categories['Other'])}")

        print(f"\nSample concepts from each category:")
        for category, concepts in neuro_categories.items():
            if concepts:
                print(f"  {category}: {', '.join(concepts[:3])}(...)")

        findings["counts"] = {
            'motivation': motivation_count,
            'emotions_stress': emotion_count,
            'cognition': cognition_count,
            'learning_development': learning_count,
            'total_concepts': sum(len(concepts) for concepts in neuro_concepts.values())
        }

    return findings

def generate_text2cypher_expansions(findings):
    """Generate the expansion data for Text2Cypher"""

    print("\n\n📚 GENERATING TEXT2CYPHER EXPANSIONS")
    print("="*80)

    # Extract key concepts from findings
    expansion_data = {
        'node_labels': findings['node_labels'],
        'relationship_types': findings['relationship_types'],
        'motivation_concepts': findings['neuroscience_categories'].get('Motivation', []),
        'emotion_stress_concepts': findings['neuroscience_categories'].get('Emotions & Stress', []),
        'cognition_concepts': findings['neuroscience_categories'].get('Cognition', []),
        'learning_concepts': findings['neuroscience_categories'].get('Learning & Development', []),
        'relationship_patterns': findings['relationship_patterns']
    }

    # Save to file for reference
    with open('neuroscience_graph_expansion.json', 'w', encoding='utf-8') as f:
        json.dump(expansion_data, f, indent=2, ensure_ascii=False)

    print("✅ Expansion data saved to 'neuroscience_graph_expansion.json'")

    # Generate improved query patterns
    query_patterns = generate_query_patterns(expansion_data)
    print("\n🔍 Key Query Patterns Identified:")
    for pattern in query_patterns[:10]:  # Show first 10
        print(f"  • {pattern}")

    return expansion_data, query_patterns

def generate_query_patterns(expansion_data):
    """Generate improved query patterns for Text2Cypher"""

    patterns = []

    # Motivation patterns
    if expansion_data['motivation_concepts']:
        patterns.extend([
            "MATCH (m:MotivationalModulation)-[r:SUPPORTS|ENHANCES]->(l:LearningDevelopment) RETURN m.name, type(r), l.name",
            "MATCH (motivation)-[r:CONTRASTS_WITH|DIFFERENTIATES_FROM]->(motivation2) WHERE type(r) IN ['IS_CONTRASTED_WITH', 'CONTRASTS_WITH'] RETURN motivation.name, type(r), motivation2.name"
        ])

    # Stress/Motivation interaction
    if expansion_data['emotion_stress_concepts']:
        patterns.extend([
            "MATCH (s:PositiveStressEustress|NegativeStressDistress)-[r:INFLUENCES|AFFECTS]->(m:MotivationalModulation) RETURN s.name, type(r), m.name",
            "MATCH (emotion)-[r:SUPPORTS|IMPAIRS]->(cognition) WHERE type(r) IN ['ENHANCES', 'IMPAIRS', 'SUPPORTS'] RETURN emotion.name, type(r), cognition.name"
        ])

    # Failure/Mindset patterns
    patterns.extend([
        "MATCH (f:LearnedHelplessness)-[r:IS_LINKED_TO|IS_CONTRASTED_WITH]->(g:GrowthMindset) RETURN f.name, type(r), g.name",
        "MATCH (mindset)-[r:PROMOTES|ENABLES]->(outcome) WHERE type(r) IN ['ENHANCES', 'SUPPORTS', 'PROMOTES'] RETURN mindset.name, type(r), outcome.name"
    ])

    return patterns

def main():
    """Main exploration function"""

    print("🧠 Neuroscience Graph Content Exploration Tool")
    print("Discovering what Text2Cypher needs to learn...\n")

    # Connect to Neo4j
    driver = connect_to_neo4j()
    if not driver:
        return

    try:
        # Explore the graph
        findings = explore_neuroscience_concepts(driver)

        # Generate Text2Cypher expansions
        expansion_data, query_patterns = generate_text2cypher_expansions(findings)

        print("\n\n🎯 SUMMARY:")
        print(f"   📊 Total node labels: {len(findings['node_labels'])}")
        print(f"   🔗 Total relationship types: {len(findings['relationship_types'])}")
        print(f"   🧠 Total neuroscience concepts: {findings['counts']['total_concepts']}")

        print("\n🎁 Ready to expand Text2Cypher with:")
        print(f"   📝 {len(query_patterns)} improved query patterns")
        print(f"   🏷️  {len(expansion_data['node_labels'])} node labels to recognize")
        print(f"   🌐 {len(expansion_data['relationship_patterns'])} relationship patterns")

        print("\n✅ EXPLORATION COMPLETE!")

    finally:
        driver.close()

if __name__ == "__main__":
    main()
