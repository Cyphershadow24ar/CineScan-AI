import lancedb
db = lancedb.connect("data/lancedb")
table = db.open_table("video_frames")
print(f"Total records in DB: {len(table)}")
print("Sample record metadata:", table.head(1).to_dict())