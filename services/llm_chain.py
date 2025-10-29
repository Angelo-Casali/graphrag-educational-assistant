
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

cognitive_analysis_template = PromptTemplate.from_template("""
You are an expert in neuroscience and cognitive processes, tasked with generating a comprehensive analysis.

Context:
{context}

Research Question:
{query}

Instructions:
Generate a detailed cognitive analysis based on the provided context and research question. The analysis should include:
- Title: A clear, descriptive title for the cognitive analysis.
- Key Concepts: Identify and explain the main neuroscience and cognitive concepts involved.
- Relationships and Interactions: Describe how these concepts are related, supported, hindered, or caused by each other, drawing from the graph data.
- Implications/Insights: Provide insights into the functional implications of these relationships for cognitive performance, learning, or behavior.
- Further Research Questions: Suggest potential avenues for further investigation based on the analysis.

Ensure the analysis is:
- Scientifically accurate
- Well-structured and coherent
- Directly relevant to the provided context and query
""")

def call_llm(query: str, context: str) -> str:
    # Set environment variables for ChatOpenAI to work with OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        os.environ["OPENAI_API_KEY"] = openrouter_key
    else:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        temperature=0.7
    )
    chain = cognitive_analysis_template | llm
    try:
        result = chain.invoke({"query": query, "context": context})
        # Result is an AIMessage, so access .content (should be string)
        if hasattr(result, 'content') and isinstance(result.content, str):
            return result.content.strip()
        else:
            return "⚠️ Unexpected response format from LLM."
    except Exception as e:
        print("❌ LLM generation error:", e)
        return "⚠️ Unable to generate cognitive analysis."
