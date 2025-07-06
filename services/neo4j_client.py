import os
from langchain.graphs import Neo4jGraph
from utils.logger import get_logger

logger = get_logger(__name__)

def get_neo4j_graph():
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j")
        )
        logger.info("Connected to Neo4j")
        return graph
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        raise
