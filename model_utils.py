import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import typing
import logging
import itertools

logger = logging.getLogger(__name__)

class ModelProcessor:
    """
    Handles SigLIP model loading and embedding generation.
    
    This class supports batch processing of frames (yielded from the VideoProcessor)
    to guarantee stable memory usage when analyzing 30-minute long videos.
    """

    def __init__(self, model_id: str = "google/siglip-base-patch16-224"):
        """
        Initializes the ModelProcessor and preloads the SigLIP model.
        
        Args:
            model_id (str): The Hugging Face model identifier for the SigLIP model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading model '{model_id}' onto {self.device}...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        logger.info("Model loaded successfully.")

    def generate_embeddings(
        self, 
        frames_iterator: typing.Iterator[typing.Tuple[float, Image.Image]], 
        batch_size: int = 32
    ) -> typing.Iterator[typing.Tuple[float, typing.List[float]]]:
        """
        Generates visual embeddings for a continuous stream of frames in batches.
        
        Args:
            frames_iterator (Iterator): The generator from VideoProcessor yielding (timestamp, Image).
            batch_size (int): Number of frames to process in a single fast forward pass.
            
        Yields:
            tuple: A tuple containing:
                   - timestamp_sec (float): Timestamp of the corresponding frame.
                   - embedding (list[float]): The 1D normalized feature vector for semantic search.
        """
        while True:
            # Consume up to `batch_size` items from the iterator
            batch = list(itertools.islice(frames_iterator, batch_size))
            if not batch:
                break
                
            timestamps, images = zip(*batch)
            
            try:
                # Preprocess the images and move tensors to the correct device
                inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
                
                # Perform a forward pass with no gradient calculation for speed and memory efficiency
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    # SigLIP returns an object; we need the actual tensor data
                    image_features = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs
                    # Now we can normalize the math tensor
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Yield embeddings sequentially for the downstream DB handler to consume
                embeddings_list = image_features.cpu().numpy().tolist()
                
                for timestamp, emb in zip(timestamps, embeddings_list):
                    yield (timestamp, emb)
                    
            except Exception as e:
                logger.error(f"Failed during embedding generation for batch starting at {timestamps[0]}s: {e}")
                raise

    def generate_text_embedding(self, query: str) -> typing.List[float]:
        try:
            # 1. Prepare the text query
            inputs = self.processor(text=[query], return_tensors="pt", padding="max_length", truncation=True).to(self.device)
            
            # 2. Run the model without tracking gradients (saves memory/time)
            with torch.no_grad():
                model_output = self.model.get_text_features(**inputs)
            
            # 3. Extract the raw numerical tensor from the output object
            if isinstance(model_output, torch.Tensor):
                text_features = model_output
            elif hasattr(model_output, 'pooler_output') and model_output.pooler_output is not None:
                text_features = model_output.pooler_output
            else:
                # Fallback for different transformers library versions
                text_features = model_output[0] 

            # 4. L2 Normalization (Essential for accurate Cosine Similarity search)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 5. Convert to a standard list for LanceDB
            return text_features.squeeze(0).cpu().numpy().tolist()
        except Exception as e:
            logger.error(f"Failed to generate text embedding: {e}")
            raise

# Example usage pattern
if __name__ == "__main__":
    pass
