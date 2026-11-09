
from dotenv import load_dotenv

load_dotenv("api.env")
import os
import pandas as pd
import numpy as np
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

load_dotenv("api.env")

from pathlib import Path
from supabase import create_client

# Project paths
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = (
    BASE_DIR
    / "notebooks"
    / "data"
    / "processed"
    / "customer_support_chunks.csv"
)

EMBEDDINGS_PATH = (
    BASE_DIR
    / "notebooks"
    / "data"
    / "processed"
    / "customer_support_embeddings.npy"
)

# Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
print("Project URL loaded:", bool(SUPABASE_URL))
print("Secret key loaded:", bool(SUPABASE_KEY))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError(
        "Please configure SUPABASE_URL and "
        "SUPABASE_SERVICE_ROLE_KEY environment variables."
    )

# Connect to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

print("Connected to Supabase.")

# Load data
df = pd.read_csv(CSV_PATH)
embeddings = np.load(EMBEDDINGS_PATH)

print("CSV shape:", df.shape)
print("Embeddings shape:", embeddings.shape)

# Validate data
if len(df) != len(embeddings):
    raise ValueError("CSV and embedding counts do not match.")

if embeddings.shape[1] != 384:
    raise ValueError("Expected embedding dimension: 384.")

if df["text_chunk"].isna().any():
    raise ValueError("Some text chunks are missing.")

# Prepare records
records = []

for i, row in df.iterrows():
    records.append({
        "id": i + 1,
        "question": str(row["instruction_raw"]),
        "answer": str(row["response"]),
        "content": str(row["text_chunk"]),
        "embedding": embeddings[i].astype(float).tolist()
    })

print("Prepared records:", len(records))

# Upload in batches
BATCH_SIZE = 25
total = len(records)

for start in range(0, total, BATCH_SIZE):
    batch = records[start:start + BATCH_SIZE]

    supabase.table("customer_support_knowledge").upsert(
    batch,
    on_conflict="id"
).execute()

    uploaded = min(start + BATCH_SIZE, total)

    print(f"Uploaded {uploaded}/{total} records")

print("Knowledge base upload completed successfully!")
