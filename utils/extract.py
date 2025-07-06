import json
from utils.logger import get_logger

logger = get_logger(__name__)

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
