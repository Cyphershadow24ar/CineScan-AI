import streamlit as st
import os
from PIL import Image
import cv2

# Import our custom modules
from model_utils import ModelProcessor
from db_handler import VectorDB
from video_utils import VideoProcessor

# 1. Page config
st.set_page_config(page_title="CineScan AI", page_icon="🎬", layout="wide")

# 2. Custom CSS Styling (Refined for Readability)
st.markdown("""
    <style>
    .stApp {
        background-color: #0B0E11;
        color: #E0E0E0;
    }
    section[data-testid="stSidebar"] {
        background-color: #111418;
    }
    h1 {
        color: #00D1FF;
        font-weight: 800;
        margin-bottom: 0px;
    }
    .description-box {
        background-color: #1A202C;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #00D1FF;
        margin-bottom: 25px;
    }
    div.stButton > button:first-child {
        background-color: #00D1FF;
        color: #0B0E11;
        border-radius: 8px;
        font-weight: 700;
        width: 100%;
    }
    .stImage > img {
        border-radius: 12px;
        border: 1px solid #2D3748;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return ModelProcessor()

@st.cache_resource
def get_db():
    return VectorDB()

# --- Header Section ---
st.title("🎬 CineScan")
st.subheader("Intelligent Neural Video Discovery")

# New Instructional Description
st.markdown("""
<div class="description-box">
    <strong>How CineScan Works:</strong><br>
    Unlike traditional search that looks for filenames, CineScan uses <strong>Artificial Intelligence</strong> to 'watch' your videos. 
    It converts every frame into a mathematical fingerprint and matches it against the meaning of your words. 
    <br><br>
    <em>Try searching for actions like "a goal being scored," objects like "a red car," or specific people!</em>
</div>
""", unsafe_allow_html=True)

st.caption("Powered by SigLIP Vision Transformers & LanceDB Vector Engine")

# --- Sidebar ---
with st.sidebar:
    st.header("🎞️ Add New Video")
    uploaded_file = st.file_uploader("Upload a video to start scanning", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        video_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button("Start AI Analysis"):
            with st.spinner("AI is learning your video content..."):
                model = load_models()
                db = get_db()
                vp = VideoProcessor()
                
                output_dir = os.path.join("data", "extracted_frames", os.path.splitext(uploaded_file.name)[0])
                os.makedirs(output_dir, exist_ok=True)
                
                cap = cv2.VideoCapture(video_path)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.release()
                expected_count = max(1, int((total_frames / fps) * vp.target_fps))

                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_paths_cache = {}
                stats = {"processed": 0}

                def caching_frame_generator():
                    for timestamp, image in vp.extract_frames(video_path):
                        frame_filename = f"frame_{timestamp:.2f}.jpg"
                        frame_path = os.path.join(output_dir, frame_filename)
                        image.save(frame_path)
                        frame_paths_cache[timestamp] = frame_path
                        yield (timestamp, image)

                embeddings_gen = model.generate_embeddings(caching_frame_generator(), batch_size=32)

                def db_generator():
                    for timestamp, embedding in embeddings_gen:
                        frame_path = frame_paths_cache.pop(timestamp, "")
                        
                        hours, rem = divmod(int(timestamp), 3600)
                        minutes, seconds = divmod(rem, 60)
                        timestamp_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                        
                        stats["processed"] += 1
                        progress = min(stats["processed"] / expected_count, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"AI is 'watching' frame {stats['processed']}...")

                        yield {
                            "video_name": uploaded_file.name,
                            "timestamp": float(timestamp),
                            "timestamp_str": timestamp_str,
                            "frame_path": frame_path,
                            "vector": embedding
                        }

                db.add_embeddings(db_generator(), batch_size=50)
                st.toast(f"Success! '{uploaded_file.name}' is now searchable.", icon="✅")
    
    st.divider()
    st.header("⚙️ Refine Search")
    use_time_filter = st.checkbox("Search specific time range")
    start_time, end_time = None, None
    if use_time_filter:
        col1, col2 = st.columns(2)
        with col1: start_time = st.number_input("From (sec)", min_value=0.0, value=0.0)
        with col2: end_time = st.number_input("To (sec)", min_value=0.0, value=60.0)
            
    top_k = st.slider("How many results to show?", 1, 20, 6)

# --- Main Search UI ---
st.header("🔍 What are you looking for?")
query = st.text_input("", placeholder="Describe a scene (e.g. 'a player celebrating a goal' or 'crowd cheering')")

if query:
    with st.spinner(f"Scanning video for: '{query}'..."):
        model = load_models()
        db = get_db()
        query_vector = model.generate_text_embedding(query)
        
        results = db.search(query_vector, top_k=top_k, 
                           start_time=start_time if use_time_filter else None, 
                           end_time=end_time if use_time_filter else None)
        
        if not results:
            st.error("The AI couldn't find a close match for that description.")
        else:
            st.success(f"CineScan found {len(results)} moments that match your description.")
            
            # Display results
            cols_per_row = 3
            for i in range(0, len(results), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, result in enumerate(results[i:i+cols_per_row]):
                    with cols[j]:
                        try:
                            img = Image.open(result["frame_path"])
                            st.image(img, use_container_width=True)
                            
                            st.markdown(f"**🕒 Time: {result['timestamp_str']}**")
                            distance = result.get("_distance", 0.0)
                            # This uses a 'power' function to boost the scores of the top results
                            # It makes a 'good' match feel like a 'great' match in the UI
                            raw_score = max(0, 1 - (distance / 2.0))
                            match_score = int((raw_score ** 0.5) * 100) # Square root scaling
                            st.caption(f"AI Match Score: {match_score}%")
                        except Exception:
                            st.error("Image loading failed.")