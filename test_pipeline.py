#!/usr/bin/env python3
"""
Test script for GraphRAG pipeline - Neuroscience Knowledge Graph
Tests each module interaction and complete pipeline flow
"""

import asyncio
import json
from typing import Dict, Any
from graph_retriever import EnhancedMultilingualText2Cypher
from llm_chain import EducationalResponseGenerator
from context_builder import EducationalContext
from config import config

async def test_complete_pipeline():
    """Test the complete GraphRAG pipeline for Italian neuroscience queries"""

    print("🧠 Testing Complete GraphRAG Pipeline - Neuroscience Knowledge Graph")
    print("="*80)

    # Test query in Italian
    test_query = "Qual è la differenza tra motivazione intrinseca ed estrinseca?"

    try:
        # Initialize components
        processor = EnhancedMultilingualText2Cypher(use_vectors=True)
        generator = EducationalResponseGenerator(
            openai_api_key=config.openai.api_key,
            language="italian"
        )

        print(f"📝 Test Query: {test_query}")
        print()

        # Step 1: Process query with retrieval
        print("🔄 Step 1: Processing query with hybrid retrieval...")
        result = await processor.process_query_with_retrieval(test_query)

        # Analyze results step by step

        # Step 2: Check Cypher generation
        cypher_result = result.get('cypher_result', {})
        print(f"⚡ Stage 2: Text2Cypher")
        print(f"   ✅ Generated Cypher: {cypher_result.get('cypher_query', 'N/A')[:120]}...")
        print(f"   ✅ Valid: {cypher_result.get('metadata', {}).get('is_valid', False)}")
        print()

        # Step 3: Check retrieval
        retrieval_result = result.get('retrieval_result')
        if retrieval_result:
            metadata = retrieval_result.metadata
            print(f"🔍 Stage 3: Graph Retrieval")
            print(f"   📊 Graph Nodes: {metadata.get('graph_count', 0)}")
            print(f"   📊 Semantic Nodes: {metadata.get('semantic_count', 0)}")
            print(f"   📊 Total Nodes: {metadata.get('total_nodes', 0)}")
            print(f"   🔗 Total Relationships: {metadata.get('total_triples', 0)}")
            print(f"   ⏱️ Execution Time: {metadata.get('timings', {}).get('total', 0):.3f}s")
            print()

            # Show sample nodes
            if retrieval_result.nodes:
                print(f"📋 Found Neuroscience Concepts:")
                for i, node in enumerate(retrieval_result.nodes[:5], 1):
                    labels = node.get('labels', [])
                    name = node.get('name', 'Unknown')
                    category = node.get('category', 'Unknown')
                    source = node.get('source', 'Unknown')
                    print(f"   {i}. '{name}' ({', '.join(labels)}) - {source}")
                print()

            # Show sample relationships
            if retrieval_result.triples:
                print("🔗 Found Key Relationships:")
                for i, (source, rel_type, target) in enumerate(retrieval_result.triples[:5], 1):
                    print(f"   {i}. '{source}' ⟷ '{target}' [{rel_type}]")
                print()

        # Step 4: Check context building
        educational_context = result.get('educational_context', {})
        if educational_context:
            print(f"🎯 Stage 4: Educational Context Building")
            confidence = educational_context.get('confidence_assessment', 'Unknown')
            print(f"   📊 Confidence: {confidence}")

            methodologies = educational_context.get('primary_methodologies', [])
            print(f"   📚 Primary Methodologies: {len(methodologies)}")

            if methodologies:
                methodology = methodologies[0]
                print(f"   📝 Sample: {methodology.get('name', 'N/A')}")
                print(f"   💡 Confidence: {methodology.get('confidence', 'N/A')}")
            print()

        # Step 5: Check LLM response generation
        llm_response = result.get('llm_response', {})
        if llm_response:
            print(f"🧠 Stage 5: LLM Response Generation (Italian)")
            response_text = llm_response.get('response', 'No response')
            confidence = llm_response.get('confidence', 'Unknown')
            print(f"   🎯 Confidence: {confidence}")
            print(f"   📝 Response: {response_text[:200]}..." if len(response_text) > 200 else f"   📝 Response: {response_text}")
            print()

        # Final Summary
        print("🎉 PIPELINE SUMMARY:")
        print("-" * 50)

        # Overall assessments
        cypher_valid = cypher_result.get('metadata', {}).get('is_valid', False)
        nodes_found = len(retrieval_result.nodes) if retrieval_result else 0
        relationships_found = len(retrieval_result.triples) if retrieval_result else 0
        context_built = bool(educational_context)
        response_generated = bool(llm_response.get('response'))

        assessments = [
            ("Text2Cypher", "✅ PASSED" if cypher_valid else "❌ FAILED"),
            ("Graph Retrieval", f"✅ PASSED ({nodes_found} nodes, {relationships_found} relationships)"),
            ("Context Building", "✅ PASSED" if context_built else "❌ FAILED"),
            ("LLM Response", "✅ PASSED" if response_generated else "❌ FAILED")
        ]

        for component, status in assessments:
            print(f"   {component}: {status}")

        print()
        print("🔬 FINAL RESULT:")
        if cypher_valid and nodes_found > 0 and response_generated:
            print("   ✨ SUCCESS: Complete pipeline working! Neuroscience GraphRAG is functional.")
        else:
            print("   ⚠️ ISSUES DETECTED: Some components need debugging.")

        # Export detailed results for analysis
        summary = {
            'query': test_query,
            'cypher_valid': cypher_valid,
            'nodes_found': nodes_found,
            'relationships_found': relationships_found,
            'context_built': context_built,
            'response_generated': response_generated,
            'complete_success': all([cypher_valid, nodes_found > 0, relationships_found >= 0, response_generated])
        }

        with open('pipeline_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("   📄 Detailed results saved to 'pipeline_test_results.json'")

    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        if 'processor' in locals():
            processor.close()
        # Note: EducationalResponseGenerator doesn't have a close method

if __name__ == "__main__":
    asyncio.run(test_complete_pipeline())
