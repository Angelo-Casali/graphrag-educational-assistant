#!/usr/bin/env python3
"""
Streamlit Demo Interface for GraphRAG Educational Assistant
Same interface as Gradio version with all pipeline stages, evidence, and comparison
"""

import streamlit as st
import asyncio
import logging
import json
import time
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import asdict

# Import your GraphRAG components
from graph_retriever import EnhancedMultilingualText2Cypher
from llm_chain import EducationalResponseGenerator
from context_builder import EducationalContext
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🧠 Neuroscience Knowledge Graph - Assistente per Insegnanti",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        font-size: 16px;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the GraphRAG system components (cached)"""
    logger.info("Initializing GraphRAG system...")
    processor = EnhancedMultilingualText2Cypher(use_vectors=True)
    generator = EducationalResponseGenerator(
        openai_api_key=config.openai.api_key,
        language="italian"
    )
    logger.info("System initialized successfully!")
    return processor, generator

async def process_query_async(query: str, processor, generator):
    """Process query through the full pipeline"""
    try:
        # Get retrieval result
        result = await processor.process_query_with_retrieval(query)
        
        # Generate response
        educational_context_obj = result.get('educational_context_obj')
        if educational_context_obj:
            response = await generator.generate_response(educational_context_obj, query)
            result['llm_response'] = response
        else:
            result['llm_response'] = {
                'response': "Non sono riuscito a generare una risposta. Riprova con una domanda diversa.",
                'confidence': 'LOW',
                'evidence_used': []
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            'error': str(e),
            'llm_response': {
                'response': f"Errore: {str(e)}",
                'confidence': 'VERY_LOW',
                'evidence_used': []
            }
        }

def format_pipeline_stages(result: Dict) -> str:
    """Format pipeline execution stages for display"""
    stages = []
    
    # Stage 1: Text2Cypher
    cypher_result = result.get('cypher_result', {})
    is_valid = cypher_result.get('metadata', {}).get('is_valid', False)
    stages.append(f"### ✅ Stage 1: Text2Cypher")
    stages.append(f"**Status:** {'Valid ✓' if is_valid else 'Invalid ✗'}")
    stages.append(f"**Generated Cypher:**")
    stages.append(f"```cypher\n{cypher_result.get('cypher_query', 'N/A')}\n```")
    
    # Stage 2: Graph Retrieval
    retrieval_result = result.get('retrieval_result')
    if retrieval_result:
        metadata = retrieval_result.metadata
        stages.append(f"\n### ✅ Stage 2: Graph Retrieval")
        stages.append(f"- **Graph Nodes:** {metadata.get('graph_count', 0)}")
        stages.append(f"- **Semantic Nodes:** {metadata.get('semantic_count', 0)}")
        stages.append(f"- **Total Nodes:** {metadata.get('total_nodes', 0)}")
        stages.append(f"- **Relationships:** {metadata.get('total_triples', 0)}")
        
        # Timings
        timings = metadata.get('timings', {})
        stages.append(f"\n**⏱️ Execution Times:**")
        stages.append(f"- Graph Traversal: {timings.get('graph_traversal', 0):.3f}s")
        stages.append(f"- Semantic Search: {timings.get('semantic_search', 0):.3f}s")
        stages.append(f"- Fusion: {timings.get('fusion', 0):.3f}s")
        stages.append(f"- **Total: {timings.get('total', 0):.3f}s**")
    
    # Stage 3: Context Building
    educational_context_dict = result.get('educational_context', {})
    if educational_context_dict:
        stages.append(f"\n### ✅ Stage 3: Context Builder")
        stages.append(f"**Confidence:** {educational_context_dict.get('confidence_assessment', 'N/A')}")
        
        primary = educational_context_dict.get('primary_methodologies', [])
        stages.append(f"\n**Primary Methodologies:** {len(primary)}")
        for i, method in enumerate(primary, 1):
            stages.append(f"  {i}. **{method.get('name', 'N/A')}** (confidence: {method.get('confidence', 'N/A')})")
        
        supporting = educational_context_dict.get('supporting_methodologies', [])
        if supporting:
            stages.append(f"\n**Supporting Methodologies:** {len(supporting)}")
            for i, method in enumerate(supporting[:3], 1):
                stages.append(f"  {i}. {method.get('name', 'N/A')}")
    
    # Stage 4: LLM Generation
    llm_response = result.get('llm_response', {})
    stages.append(f"\n### ✅ Stage 4: Response Generation")
    stages.append(f"**Status:** Complete ✓")
    stages.append(f"**Language:** Italian 🇮🇹")
    stages.append(f"**Confidence:** {llm_response.get('confidence', 'N/A')}")
    
    return "\n".join(stages)

def format_evidence(result: Dict):
    """Format evidence data for display"""
    retrieval_result = result.get('retrieval_result')
    
    if not retrieval_result:
        return "No evidence available", pd.DataFrame()
    
    # Create nodes dataframe
    nodes_data = []
    for node in retrieval_result.nodes[:15]:
        nodes_data.append({
            'Name': node.get('name', 'N/A'),
            'Category': node.get('category', 'N/A'),
            'Labels': ', '.join(node.get('labels', [])),
            'Source': node.get('source', 'graph')
        })
    
    nodes_df = pd.DataFrame(nodes_data) if nodes_data else pd.DataFrame()
    
    # Create relationships text
    relationships_text = []
    relationships_text.append("### 🔗 Key Relationships\n")
    for i, (source, rel_type, target) in enumerate(retrieval_result.triples[:10], 1):
        relationships_text.append(f"{i}. **{source}** → `{rel_type}` → **{target}**")
    
    # Facets
    facets_text = []
    facets_text.append("\n### 📊 Knowledge Graph Statistics\n")
    label_counts = retrieval_result.facets.get('label_counts', {})
    if label_counts:
        facets_text.append("**Node Types:**")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            facets_text.append(f"- {label}: {count}")
    
    rel_counts = retrieval_result.facets.get('rel_counts', {})
    if rel_counts:
        facets_text.append("\n**Relationship Types:**")
        for rel, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            facets_text.append(f"- {rel}: {count}")
    
    evidence_text = "\n".join(relationships_text) + "\n" + "\n".join(facets_text)
    
    return evidence_text, nodes_df

def main():
    """Main Streamlit app"""
    
    # Title and description
    st.title("🧠 Neuroscience Knowledge Graph - Assistente per Insegnanti")
    st.markdown("""
    **Sistema di supporto basato su neuroscienze per insegnanti e formatori italiani**
    
    Fai domande su motivazione, stress, mindset, emozioni e metacognizione per ricevere risposte scientificamente fondate con strategie pratiche.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Informazioni")
        st.markdown("""
        ### 🔍 Come funziona
        
        1. **Text2Cypher**: Traduce la tua domanda in query Cypher
        2. **Graph Retrieval**: Recupera dati dal knowledge graph (con Node2Vec)
        3. **Context Builder**: Costruisce contesto educativo strutturato
        4. **Response Generator**: Genera risposta in italiano
        
        ### 💡 Esempi di domande
        
        **Livello 1 - Base:**
        - "Qual è la differenza tra motivazione intrinseca ed estrinseca?"
        - "Lo stress può essere positivo per l'apprendimento?"
        - "Cosa significa 'growth mindset'?"
        
        **Livello 2 - Strategie:**
        - "Come posso incoraggiare la motivazione intrinseca?"
        - "Quali attività riducono lo stress prima degli esami?"
        - "Come promuovere un growth mindset?"
        
        **Livello 3 - Interventi:**
        - "Come spostare gli studenti da motivazione estrinseca a intrinseca?"
        - "Come aiutare studenti a reinterpretare il fallimento?"
        
        **Livello 4 - Integrazione:**
        - "Come lo stress influenza la motivazione?"
        - "Qual è il legame tra emozioni e mindset?"
        
        ### ℹ️ Aree Tematiche
        
        - 🎯 **Motivazione** (intrinseca vs estrinseca)
        - 💪 **Stress** (positivo vs negativo)
        - 🧩 **Mindset** (fisso vs crescita)
        - 😊 **Emozioni** (positive vs negative)
        - 🤔 **Metacognizione** (consapevolezza, riflessione)
        
        ### ℹ️ Tecnologie
        
        - **Neo4j** + **Node2Vec** (Knowledge Graph)
        - **OpenAI GPT** (LLM)
        - **LangChain** (Orchestration)
        - **Streamlit** (Interface)
        """)
        
        st.divider()
        
        # Settings
        st.header("⚙️ Impostazioni")
        show_pipeline = st.checkbox("Mostra Pipeline Stages", value=True)
        show_evidence = st.checkbox("Mostra Evidence", value=True)
        show_context = st.checkbox("Mostra Context", value=False)
    
    # Initialize system
    with st.spinner("🔄 Inizializzazione sistema..."):
        processor, generator = initialize_system()
    
    # Main query interface
    st.header("💬 Fai una Domanda")
    
    query = st.text_area(
        "Inserisci la tua domanda in italiano:",
        placeholder="Es: Ci sono strategie per studenti con ADHD?",
        height=100,
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button("🚀 Genera Risposta", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and query.strip():
        with st.spinner("🔄 Elaborazione in corso..."):
            # Run async function
            result = asyncio.run(process_query_async(query, processor, generator))
            
            # Store in session state
            st.session_state['last_result'] = result
            st.session_state['last_query'] = query
    
    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        
        st.divider()
        
        # Response
        st.header("📝 Risposta")
        llm_response = result.get('llm_response', {})
        response_text = llm_response.get('response', 'No response generated')
        
        # Display response in a nice box
        st.markdown(f"""
        <div class="success-box">
        {response_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence indicator
        confidence = llm_response.get('confidence', 'MEDIUM')
        confidence_colors = {
            'VERY_HIGH': '🟢',
            'HIGH': '🟢',
            'MEDIUM': '🟡',
            'LOW': '🟠',
            'VERY_LOW': '🔴'
        }
        st.caption(f"Confidence: {confidence_colors.get(confidence, '⚪')} {confidence}")
        
        # Tabs for detailed information
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Pipeline Stages",
            "📊 Evidence & Data",
            "🧩 Educational Context",
            "📈 Comparison"
        ])
        
        with tab1:
            if show_pipeline:
                st.markdown("### 🔬 Pipeline Execution Stages")
                pipeline_info = format_pipeline_stages(result)
                st.markdown(pipeline_info)
            else:
                st.info("Enable 'Mostra Pipeline Stages' in sidebar to see details")
        
        with tab2:
            if show_evidence:
                st.markdown("### 📊 Evidence from Knowledge Graph")
                evidence_text, nodes_df = format_evidence(result)
                
                st.markdown(evidence_text)
                
                if not nodes_df.empty:
                    st.markdown("\n### 📋 Retrieved Nodes")
                    st.dataframe(nodes_df, use_container_width=True)
            else:
                st.info("Enable 'Mostra Evidence' in sidebar to see details")
        
        with tab3:
            if show_context:
                st.markdown("### 🧩 Educational Context (Structured)")
                educational_context = result.get('educational_context', {})
                
                if educational_context:
                    # Student Profile
                    st.markdown("#### 👤 Student Profile")
                    profile = educational_context.get('student_profile', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Primary Needs:**", ', '.join(profile.get('primary_needs', [])))
                    with col2:
                        st.write("**Educational Context:**", profile.get('educational_context', 'general'))
                    
                    # Methodologies
                    st.markdown("#### 📚 Recommended Methodologies")
                    primary = educational_context.get('primary_methodologies', [])
                    for i, method in enumerate(primary, 1):
                        with st.expander(f"{i}. {method.get('name', 'N/A')} - {method.get('confidence', 'N/A')}"):
                            st.write("**Category:**", method.get('category', 'N/A'))
                            st.write("**Implementation:**", method.get('implementation_guidance', 'N/A'))
                            st.write("**Applications:**")
                            for app in method.get('classroom_applications', []):
                                st.write(f"- {app}")
                    
                    # Evidence
                    st.markdown("#### 🔬 Evidence Summary")
                    st.write(educational_context.get('evidence_summary', 'N/A'))
                    
                    # Implementation Priority
                    st.markdown("#### 🎯 Implementation Priority")
                    for i, priority in enumerate(educational_context.get('implementation_priority', []), 1):
                        st.write(f"{i}. {priority}")
                
                else:
                    st.warning("No educational context available")
            else:
                st.info("Enable 'Mostra Context' in sidebar to see details")
        
        with tab4:
            st.markdown("### 📈 Comparison: Hybrid vs Graph-Only")
            
            retrieval_result = result.get('retrieval_result')
            if retrieval_result:
                metadata = retrieval_result.metadata
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Graph Nodes (Direct)",
                        metadata.get('graph_count', 0),
                        help="Nodes from direct graph traversal"
                    )
                    st.metric(
                        "Total Nodes",
                        metadata.get('total_nodes', 0),
                        help="Total nodes after hybrid retrieval"
                    )
                
                with col2:
                    st.metric(
                        "Semantic Nodes (Node2Vec)",
                        metadata.get('semantic_count', 0),
                        help="Additional nodes from semantic search"
                    )
                    st.metric(
                        "Relationships",
                        metadata.get('total_triples', 0),
                        help="Total relationships retrieved"
                    )
                
                st.markdown("""
                **💡 Hybrid Retrieval Benefits:**
                - 🎯 **Precision**: Direct graph relationships (exact matches)
                - 🔍 **Breadth**: Node2Vec semantic similarity (related concepts)
                - 🚀 **Coverage**: Neighbor expansion (contextual information)
                """)
            else:
                st.warning("No comparison data available")

if __name__ == "__main__":
    main()
