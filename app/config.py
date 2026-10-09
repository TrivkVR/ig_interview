import os
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_COLLECTION = "ManufacturingDocs"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5.6-luna")
GUARDRAIL_MODEL = os.getenv("GUARDRAIL_MODEL", "gpt-5.6-luna")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoint_db")
CHECKPOINT_DB_URL = os.getenv(
    "CHECKPOINT_DB_URL",
    f"postgresql://{quote(DB_USER, safe='')}:{quote(DB_PASS, safe='')}@{DB_HOST}:{DB_PORT}/{CHECKPOINT_DB}",
)
API_URL = os.getenv("API_URL", "http://localhost:8000")
PHOENIX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT = os.getenv("PHOENIX_PROJECT_NAME", "manufacturing-agent")
