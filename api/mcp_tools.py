from mcp.server.fastmcp import FastMCP
from utils.extract import extract_text_for_embedding, extract_graph_data
from core.embedding import labse_model, reranker_model
from services.milvus_client import connect_to_milvus, COLLECTION_NAME
from services.neo4j_client import get_neo4j_graph
from services.groq_client import get_llm
from langchain.prompts import PromptTemplate
import json
from typing import List, Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastMCP
mcp = FastMCP("RAG Graph Intelligence MCP")
client = connect_to_milvus()

# Load collection
from services.milvus_client import initialize_collection
initialize_collection(client)

# MCP Tools
@mcp.tool()
async def ingest_documents(documents: List[Dict[str, Any]]):
    """
    Ingest documents into the vector database.
    
    Args:
        documents: List of documents to ingest
        
    Returns:
        Confirmation of ingestion with document IDs
    """
    try:
        documents_to_insert = []
        
        for doc in documents:
            if not isinstance(doc, dict):
                return json.dumps({"error": "Each document must be a dictionary"})
            
            # Handle graph-structured data
            if "nodes" in doc and "relationships" in doc:
                if "id" not in doc:
                    if len(doc["nodes"]) > 0 and "id" in doc["nodes"][0]:
                        doc["id"] = f"graph_{doc['nodes'][0]['id']}"
                    else:
                        doc["id"] = f"graph_{len(documents_to_insert)}"
            elif "id" not in doc:
                return json.dumps({"error": "Missing required 'id' field"})
            
            # Generate embedding
            text_for_embedding = extract_text_for_embedding(doc)
            vector = labse_model.encode(text_for_embedding).tolist()
            
            # Prepare document with vector
            doc_with_vector = doc.copy()
            doc_with_vector["vector"] = vector
            documents_to_insert.append(doc_with_vector)
        
        # Insert into Milvus
        result = client.insert(
            collection_name=COLLECTION_NAME,
            data=documents_to_insert
        )
        
        if not result or "insert_count" not in result:
            return json.dumps({"error": "Failed to get insertion confirmation"})
        
        return json.dumps({
            "status": "success",
            "inserted_count": result["insert_count"],
            "ids": [doc["id"] for doc in documents_to_insert]
        })
    
    except Exception as e:
        logger.error(f"Error in ingest_documents: {str(e)}")
        return json.dumps({"error": f"Ingestion failed: {str(e)}"})

@mcp.tool()
async def search_documents(query: str, limit: int = 10, filter: Optional[Dict[str, Any]] = None):
    """
    Search for documents using vector similarity.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        filter: Optional filter criteria
        
    Returns:
        List of matching documents with scores
    """
    try:
        # Generate query embedding
        query_vector = labse_model.encode(query).tolist()
        
        # Configure search parameters
        search_params = {
            "params": {"nprobe": 16}
        }

        # Perform vector search
        result = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            search_params=search_params,
            limit=limit,
            output_fields=["*"],
            anns_field="vector"
        )

        # Process results
        candidates = []
        for hits in result:
            for hit in hits:
                metadata = {k: v for k, v in hit.entity.items() 
                           if k not in ["id", "vector"]}
                
                candidates.append({
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "metadata": metadata
                })

        # Rerank results
        if candidates:
            pairs = [(query, extract_text_for_embedding(c["metadata"])) 
                    for c in candidates]
            reranker_scores = reranker_model.predict(pairs)
            
            for i, score in enumerate(reranker_scores):
                candidates[i]["score"] = float(score)
            
            candidates.sort(key=lambda x: x["score"], reverse=True)

        return json.dumps({
            "results": candidates[:limit]
        })

    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        return json.dumps({"error": f"Search error: {str(e)}"})

@mcp.tool()
async def graph_search(query: str, limit: int = 1):
    """
    Perform graph-based search using vector similarity and Neo4j.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        
    Returns:
        Graph search results with visualization data
    """
    try:
        # Generate query embedding
        query_vector = labse_model.encode(query).tolist()
        
        search_params = {
            "params": {"nprobe": 16}
        }

        result = client.search(
            collection_name=COLLECTION_NAME,
            data=[query_vector],
            search_params=search_params,
            limit=limit,
            output_fields=["*"],
            anns_field="vector"
        )
        
        # Process search results
        context_texts = []
        for hits in result:
            for hit in hits:
                clean_entity = {}
                for k, v in hit.entity.items():
                    if k != "vector":
                        clean_entity[k] = v
                 
                context_text = extract_text_for_embedding(clean_entity)
                context_texts.append(context_text)
        
        # Join context texts
        context = " ".join(context_texts)
        
        # Get Neo4j schema
        graph = get_neo4j_graph()
        schema = graph.get_schema
        
        # Generate Cypher query using Groq LLM
        llm = get_llm()
        
        # Prepare the prompt for Cypher generation
        cypher_prompt = PromptTemplate(
            input_variables=["schema", "context", "question"],
            template="""
            You are an expert in building Cypher queries for Neo4j graph databases. Using the schema details and the contextual information provided, your task is to generate a precise, syntactically correct, and visualization-friendly Cypher query that directly answers the user's natural language question.

            Schema Overview:
            {schema}

            Relevant Context from Knowledge Base:
            {context}

            User Question:
            {question}

            Instructions:
            - Use only the node labels, relationship types, and properties defined in the schema.
            - Leverage pattern matching to express meaningful connections between relevant nodes.
            - Identify and use correct node types based on the topic.
            - Optimize the query for graph visualization by returning connected paths.
            - Prefer using MATCH, OPTIONAL MATCH, and WHERE clauses as needed.
            - If applicable, alias paths or nodes for clarity and visualization grouping.
            - Return both answer data and the relationships between entities.
            - Do not fabricate labels, properties, or relationships that are not explicitly defined.
            - Ensure the query is ready to be run directly in Neo4j.
            - Cypher query should be accurate to the schema and context provided.

            Generate only the Cypher query below without additional explanation.

            Cypher query:
            """
        )
        
        # Format prompt
        formatted_prompt = cypher_prompt.format(
            schema=schema,
            context=context[:500] if context else "No relevant context found.",
            question=query
        )
        
        # Generate Cypher query
        llm_response = llm.invoke(formatted_prompt)
        cypher_query = llm_response.content.strip()
        
        # Extract just the Cypher query if it contains additional text
        if "```cypher" in cypher_query:
            cypher_query = cypher_query.split("```cypher")[1].split("```")[0].strip()
        elif "```" in cypher_query:
            cypher_query = cypher_query.split("```")[1].split("```")[0].strip()
        
        # Execute Cypher query and get visualization data
        visualization_data = extract_graph_data(graph, cypher_query)
        
        return json.dumps({
            "answer": "Query executed successfully",
            "cypher_query": cypher_query,
            "visualization_data": visualization_data
        })
        
    except Exception as e:
        logger.error(f"Error in graph_search: {str(e)}")
        return json.dumps({"error": f"Graph search error: {str(e)}"})

