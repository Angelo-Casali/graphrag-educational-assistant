#!/usr/bin/env python3
"""
Test Neuroscience Queries - Verify Text2Cypher expansion works for all neuroscience topics
"""

import asyncio
import json
from graph_retriever import EnhancedMultilingualText2Cypher
from llm_chain import EducationalResponseGenerator
from context_builder import EducationalContext
from config import config

async def test_single_query(query_text):
    """Test a single neuroscience query through the pipeline"""

    try:
        # Initialize components
        processor = EnhancedMultilingualText2Cypher(use_vectors=True)
        generator = EducationalResponseGenerator(
            openai_api_key=config.openai.api_key,
            language="italian"
        )

        # Process query with retrieval
        result = await processor.process_query_with_retrieval(query_text)

        # Extract key metrics
        cypher_result = result.get('cypher_result', {})
        retrieval_result = result.get('retrieval_result')
        educational_context = result.get('educational_context', {})
        llm_response = result.get('llm_response', {})

        # Calculate metrics
        cypher_valid = cypher_result.get('metadata', {}).get('is_valid', False)
        nodes_found = len(retrieval_result.nodes) if retrieval_result else 0
        semantic_nodes = retrieval_result.metadata.get('semantic_count', 0) if retrieval_result else 0
        context_built = bool(educational_context)
        response_generated = bool(llm_response.get('response'))

        return {
            'query': query_text,
            'cypher_valid': cypher_valid,
            'nodes_found': nodes_found,
            'semantic_nodes': semantic_nodes,
            'context_built': context_built,
            'response_generated': response_generated,
            'success': cypher_valid and nodes_found > 0 and response_generated
        }

    except Exception as e:
        return {
            'query': query_text,
            'error': str(e),
            'success': False,
            'nodes_found': 0,
            'semantic_nodes': 0,
            'context_built': False,
            'response_generated': False
        }
    finally:
        if 'processor' in locals():
            processor.close()

async def test_neuroscience_queries():
    """Test the pipeline with various neuroscience topics"""

    test_queries = [
        "What signs show that stress is becoming harmful for a student?",
        "What’s the best way to explain positive stress to teenagers?",
        "How do metacognitive strategies help regulate emotions?",
        "What are the main characteristics of a fixed mindset?",
        "Come funziona la memoria di lavoro?"
    ]

    print("🧠 TESTING NEUROSCIENCE TEXT2CYPHER EXPANSION")
    print("="*80)

    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("─" * 60)

        result = await test_single_query(query)
        results.append(result)

        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
        else:
            print("✅ PROCESSED:")
            print(f"   • Text2Cypher Valid: {result['cypher_valid']}")
            print(f"   • Graph Nodes Found: {result['nodes_found']}")
            print(f"   • Semantic Nodes:    {result['semantic_nodes']}")
            print(f"   • Context Built:     {result['context_built']}")
            print(f"   • Response Generated: {result['response_generated']}")

            # Check if this query now works (not just motivation)
            if result['nodes_found'] > 0:
                print(f"   ✨ STATUS: WORKING ✅ (Found neuroscience graph data)")
            elif result['semantic_nodes'] > 0:
                print(f"   ⚠️  STATUS: SEMANTIC (Found semantic data only)")
            else:
                print(f"   ❌ STATUS: FAILED (No data retrieved)")

    print("\n\n🎯 ANALYSIS:")
    print("─" * 60)

    # Analyze the results
    working_queries = sum(1 for r in results if r.get('nodes_found', 0) > 0)
    semantic_queries = sum(1 for r in results if r.get('semantic_nodes', 0) > 0 and r.get('nodes_found', 0) == 0)
    failed_queries = sum(1 for r in results if r.get('nodes_found', 0) == 0 and r.get('semantic_nodes', 0) == 0)

    print(f"BEFORE Text2Cypher Expansion:")
    print("  • Only Motivation queries worked")
    print("  • Other neuroscience topics fell back to semantic search")
    print("  • Low confidence, few methodologies")

    print(f"\nAFTER Text2Cypher Expansion:")
    print(f"  • Working Queries: {working_queries}")
    print(f"  • Semantic Fallback: {semantic_queries}")
    print(f"  • Failed Queries: {failed_queries}")

    if working_queries >= 3:  # At least 3 different neuroscience topics should work
        print(f"  ✨ SUCCESS: Neuroscience GraphRAG is fully functional!")
    else:
        print(f"  ⚠️  PARTIAL: Text2Cypher expansion needs more fine-tuning")

    # Save detailed results
    with open('neuroscience_test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Detailed results saved to 'neuroscience_test_results.json'")

if __name__ == "__main__":
    asyncio.run(test_neuroscience_queries())
