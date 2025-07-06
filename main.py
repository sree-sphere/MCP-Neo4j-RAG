from dotenv import load_dotenv
from api.mcp_tools import mcp

if __name__ == "__main__":
    load_dotenv()
    mcp.run(transport="stdio")
