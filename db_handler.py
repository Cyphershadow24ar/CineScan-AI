import os
import lancedb
import pyarrow as pa
from typing import Iterator, Dict, Any, List

class VectorDB:
    def __init__(self, db_path: str = "data/lancedb"):
        """
        Initializes the LanceDB connection.
        """
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.table_name = "video_frames"
        
        # Define the schema strictly as requested. 768 is the SigLIP dimension.
        self.schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("video_name", pa.string()),
            pa.field("timestamp", pa.float32()),
            pa.field("timestamp_str", pa.string()),
            pa.field("frame_path", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 768))
        ])
        
        # Create table if it doesn't exist
        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=self.schema)
        else:
            self.table = self.db.open_table(self.table_name)

    def add_embeddings(self, data_generator: Iterator[Dict[str, Any]], batch_size: int = 100):
        """
        Consumes a generator and inserts records into LanceDB in batches.
        This provides high memory efficiency for long videos.
        """
        batch = []
        for record in data_generator:
            # We assume the generator yields dicts with:
            # video_name, timestamp, timestamp_str, frame_path, and vector
            
            # Generate a unique string id
            doc_id = f"{record['video_name']}_{record['timestamp']:.2f}"
            
            batch.append({
                "id": doc_id,
                "video_name": record["video_name"],
                "timestamp": float(record["timestamp"]),
                "timestamp_str": record["timestamp_str"],
                "frame_path": record["frame_path"],
                "vector": record["vector"]
            })
            
            if len(batch) >= batch_size:
                self.table.add(batch)
                batch = []
                
        # Insert any remaining records
        if batch:
            self.table.add(batch)

    def search(self, query_vector: List[float], top_k: int = 5, start_time: float = None, end_time: float = None) -> List[Dict]:
        """
        Searches the ANN index for the most relevant frames.
        Supports temporal pre-filtering using LanceDB's SQL WHERE clause.
        """
        search_query = self.table.search(query_vector).limit(top_k)
        
        # Apply temporal filters if requested
        if start_time is not None and end_time is not None:
            # LanceDB supports SQL-like where clauses for filtering before/during search
            search_query = search_query.where(f"timestamp >= {start_time} AND timestamp <= {end_time}")
        elif start_time is not None:
            search_query = search_query.where(f"timestamp >= {start_time}")
        elif end_time is not None:
            search_query = search_query.where(f"timestamp <= {end_time}")
            
        results = search_query.to_list()
        
        return results
