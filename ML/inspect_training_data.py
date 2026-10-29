
from pathlib import Path
import pyarrow.parquet as pq

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "Data" / "Processed"
file_path = DATA_DIR / "train_CA_1.parquet"

parquet_file = pq.ParquetFile(file_path)
columns = parquet_file.schema.names

print("File:", file_path.name)
print("Total rows:", parquet_file.metadata.num_rows)
print("Total columns:", len(columns))

print("\n--- COLUMN NAMES ---")
for col in columns:
    print(col)

print("\n--- IMPORTANT COLUMN CHECK ---")
for col in ["sales", "d", "date", "item_id", "store_id"]:
    print(f"{col}: {'FOUND' if col in columns else 'NOT FOUND'}")

print("\n--- SAMPLE DATA ---")
batch = next(parquet_file.iter_batches(batch_size=3))
df = batch.to_pandas()

# Display only the first 8 columns to avoid wrapping
print(df.iloc[:, :8].to_string(index=False))
