from dotenv import load_dotenv
import os
import nest_asyncio
from llama_extract import LlamaExtract

# Load environment variables
load_dotenv()
print("API Key:", os.getenv("LLAMA_CLOUD_API_KEY"))  # Debug API key loading

# Set up the extractor
nest_asyncio.apply()
extractor = LlamaExtract()  # Adding timeout

# Infer schema from the file
file_path = "3-Food Science   Nutrition - 2023 - Nosworthy - The in vivo and in vitro protein quality of three hemp protein sources.pdf"
assert os.path.exists(file_path), "File does not exist!"
extraction_schema = extractor.infer_schema("Our Schema", [file_path])

# Extract data
results = extractor.extract(
    extraction_schema.id,
    ["data/file3.pdf"],
)

print(results)
