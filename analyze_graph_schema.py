#!/usr/bin/env python3
"""
analyze_graph_schema.py - Analyze the neuroscience graph schema
Shows node labels, relationship types, and sample data to understand schema structure
"""

from config import config
from neo4j import GraphDatabase

def analyze_neuroscience_graph():
    """Analyze the structure of the neuroscience knowledge graph"""

    driver = GraphDatabase.driver(config.neo4j.uri, auth=(config.neo4j.user, config.neo4j.password))

    try:
        with driver.session() as session:
            print("🧠 NEUROSCIENCE GRAPH SCHEMA ANALYSIS")
            print("=" * 80)

            # Check what node labels exist
            result = session.run('CALL db.labels() YIELD label RETURN label ORDER BY label')
            labels = [record['label'] for record in result]

            print(f"\n🧠 NODE LABELS ({len(labels)} total):")
            print("-" * 40)
            for i, label in enumerate(labels, 1):
                count_result = session.run(f'MATCH (n:{label}) RETURN count(n) as count')
                count = count_result.single()['count']
                print(".2f")

            # Check what relationship types exist
            result = session.run('CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType')
            rel_types = [record['relationshipType'] for record in result]

            print(f"\n🔗 RELATIONSHIP TYPES ({len(rel_types)} total):")
            print("-" * 40)
            for rel_type in rel_types:
                count_result = session.run(f'MATCH ()-[r:{rel_type}]->() RETURN count(r) as count')
                count = count_result.single()['count']
                print("4d")

            # Show sample nodes for key neuroscience labels
            print(f"\n📊 SAMPLE NODES (by neuroscience category):")
            print("-" * 40)

            neuroscience_labels = [
                label for label in labels
                if any(keyword in label.lower() for keyword in [
                    'attention', 'memory', 'emotion', 'motivation', 'critical',
                    'executive', 'mindset', 'stress', 'learning', 'cognitive',
                    'metacognition', 'creativity', 'social', 'working'
                ])
            ]

            for label in neuroscience_labels[:10]:  # Show first 10 neuroscience labels
                result = session.run(f'MATCH (n:{label}) RETURN n.name, n.category LIMIT 3')
                nodes = [(record['n.name'], record['n.category']) for record in result]
                if nodes:
                    print(f"\n🔬 {label}:")
                    for name, category in nodes:
                        print(f"   • {name} (cat: {category})")

            # Total graph stats
            result = session.run('MATCH (n) RETURN count(n) as nodes')
            total_nodes = result.single()['nodes']

            result = session.run('MATCH ()-[r]-() RETURN count(r) as relationships')
            total_rels = result.single()['relationships']

            print(f"\n📈 GRAPH STATISTICS:")
            print("-" * 20)
            print(f"Total Nodes:      {total_nodes}")
            print(f"Total Relationships: {total_rels}")
            print(f"Node Labels:      {len(labels)}")
            print(f"Relationship Types: {len(rel_types)}")

            # Show Text2Cypher mismatch
            print(f"\n🎯 Text2Cypher SCHEMA MISMATCH:")
            print("-" * 30)
            print("Text2Cypher assumes labels like:")
            print("   • PositiveEmotions, NegativeEmotions")
            print("   • Mindset")
            print("   • REINFORCE, INFLUENCES relationships")
            print("\nBut your graph actually has labels like:")
            print(f"   • {', '.join(labels[:3])}...")
            print(f"   • Relationships: {', '.join(rel_types[:3])}...")

    except Exception as e:
        print(f"❌ Error analyzing graph schema: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    analyze_neuroscience_graph()
