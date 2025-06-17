from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
import os
import json
import uvicorn
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from pymilvus import connections, MilvusClient, DataType
from langchain.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastMCP
mcp = FastMCP("RAG Graph Intelligence MCP")

# Configuration
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_DIM = 768  # LaBSE embedding dimension
SERVER_ADDR = f"http://{MILVUS_HOST}:{MILVUS_PORT}"

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Groq LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

# Initialize embedders with CPU fallback
labse_model = SentenceTransformer('sentence-transformers/LaBSE', device='cpu')
reranker_model = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')

# Connect to Milvus
def connect_to_milvus():
    try:
        connections.connect(
            alias="default",
            uri=SERVER_ADDR,
            token=MILVUS_TOKEN
        )
        logger.info("Successfully connected to Milvus")
        return MilvusClient(uri=SERVER_ADDR, token=MILVUS_TOKEN)
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {str(e)}")
        raise

# Initialize Neo4j connection
def get_neo4j_graph():
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE
        )
        logger.info("Successfully connected to Neo4j")
        return graph
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}")
        raise

# Initialize Groq LLM client
def get_llm():
    try:
        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL
        )
        logger.info("Successfully initialized Groq LLM")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize Groq LLM: {str(e)}")
        raise

# Initialize Milvus client
client = connect_to_milvus()

# Create collection if it doesn't exist
def initialize_collection():
    try:
        if not client.has_collection(COLLECTION_NAME):
            schema = MilvusClient.create_schema(
                auto_id=False,
                enable_dynamic_field=True,
            )
            schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
            
            index_params = MilvusClient.prepare_index_params()
            index_params.add_index(
                field_name="vector", 
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 1024}
            )
            
            client.create_collection(
                collection_name=COLLECTION_NAME,
                schema=schema,
                index_params=index_params
            )
            
            client.load_collection(COLLECTION_NAME)
            logger.info(f"Collection {COLLECTION_NAME} initialized successfully")
        else:
            client.load_collection(COLLECTION_NAME)
            logger.info(f"Collection {COLLECTION_NAME} already exists")
            
    except Exception as e:
        logger.error(f"Error initializing collection: {str(e)}")
        raise

# Initialize collection on startup
initialize_collection()

# Helper function to extract text for embedding
def extract_text_for_embedding(document):
    """Extract text from document for embedding generation, ensuring vector fields are excluded"""
    text_parts = []
    
    if not isinstance(document, dict):
        return ""
    
    # Special handling for graph-structured data
    if "nodes" in document and "relationships" in document:
        # Process nodes
        if isinstance(document["nodes"], list):
            for node in document["nodes"]:
                if isinstance(node, dict):
                    # Extract node ID and labels
                    node_id = node.get("id", "")
                    node_labels = node.get("labels", "")
                    text_parts.append(f"Node ID: {node_id}, Labels: {node_labels}")
                    
                    # Extract node properties
                    if "properties" in node and isinstance(node["properties"], dict):
                        for prop_key, prop_value in node["properties"].items():
                            if isinstance(prop_value, list):
                                prop_text = ", ".join(str(item) for item in prop_value)
                                text_parts.append(f"{prop_key}: {prop_text}")
                            else:
                                text_parts.append(f"{prop_key}: {str(prop_value)}")
        
        # Process relationships
        if isinstance(document["relationships"], list):
            for rel in document["relationships"]:
                if isinstance(rel, dict):
                    # Extract relationship information
                    source = rel.get("source", "")
                    target = rel.get("target", "")
                    rel_type = rel.get("type", "")
                    text_parts.append(f"Relationship: {source} -{rel_type}-> {target}")
                    
                    # Extract relationship properties
                    if "properties" in rel and isinstance(rel["properties"], dict):
                        for prop_key, prop_value in rel["properties"].items():
                            text_parts.append(f"{prop_key}: {str(prop_value)}")
        
        return " ".join(text_parts)
    
    # Process all document fields for embedding
    for key, value in document.items():
        # Skip vector field explicitly
        if key == "vector":
            continue
            
        # Handle nested JSON structures
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                # Skip if this is a vector field at any level
                if nested_key == "vector":
                    continue
                    
                # Handle "data" array if it exists
                if nested_key == "data" and isinstance(nested_value, list):
                    for item in nested_value:
                        if isinstance(item, dict) and "key" in item and "value" in item:
                            try:
                                if isinstance(item["value"], str) and (item["value"].startswith("[") or item["value"].startswith("{")):
                                    try:
                                        parsed_value = json.loads(item["value"])
                                        if isinstance(parsed_value, list):
                                            values = []
                                            for pv in parsed_value:
                                                if isinstance(pv, dict) and "fileName" in pv:
                                                    values.append(pv["fileName"])
                                                else:
                                                    values.append(str(pv))
                                            clean_value = ", ".join(values)
                                        elif isinstance(parsed_value, dict):
                                            clean_value = ", ".join([f"{k}: {v}" for k, v in parsed_value.items()])
                                        else:
                                            clean_value = str(parsed_value)
                                    except json.JSONDecodeError:
                                        clean_value = item["value"].strip('"')
                                elif isinstance(item["value"], str):
                                    clean_value = item["value"].strip('"')
                                else:
                                    clean_value = str(item["value"])
                            except Exception:
                                if isinstance(item["value"], str):
                                    clean_value = item["value"].strip('"')
                                else:
                                    clean_value = str(item["value"])
                            
                            text_parts.append(f"{item['key']}: {clean_value}")
                else:
                    text_parts.append(f"{nested_key}: {str(nested_value)}")
        elif isinstance(value, list) and all(isinstance(item, dict) for item in value) and \
             all("key" in item and "value" in item for item in value):
            for item in value:
                text_parts.append(f"{item['key']}: {str(item['value'])}")
        else:
            text_parts.append(f"{key}: {str(value)}")
    
    return " ".join(text_parts)

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

def extract_graph_data(graph, cypher_query):
    """Execute Cypher query and convert results to visualization-friendly format"""
    try:
        result = graph.query(cypher_query)
        
        # Process result into a format suitable for visualization
        nodes = []
        relationships = []
        node_ids = set()
        
        for record in result:
            for key, value in record.items():
                if hasattr(value, 'id') and hasattr(value, 'labels'):
                    # This is a node
                    if value.id not in node_ids:
                        node_data = {
                            "id": value.id,
                            "labels": list(value.labels),
                            "properties": dict(value)
                        }
                        nodes.append(node_data)
                        node_ids.add(value.id)
                elif hasattr(value, 'start') and hasattr(value, 'end') and hasattr(value, 'type'):
                    # This is a relationship
                    rel_data = {
                        "source": value.start,
                        "target": value.end,
                        "type": value.type,
                        "properties": dict(value)
                    }
                    relationships.append(rel_data)
        
        return {
            "nodes": nodes,
            "relationships": relationships
        }
    except Exception as e:
        logger.error(f"Error executing Cypher query: {str(e)}")
        return {"nodes": [], "relationships": []}

if __name__ == "__main__":
    mcp.run(transport="stdio")
