import cv2
from PIL import Image
import typing
import logging

# Configure logger
logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Handles efficient frame extraction from video files.
    
    This class is designed for memory efficiency, explicitly yielding frames
    to prevent RAM spikes when processing long videos (e.g., 30+ minutes).
    """

    def __init__(self, target_fps: float = 0.2):
        """
        Initializes the VideoProcessor.
        
        Args:
            target_fps (int): The number of frames to extract per second of the video.
                              Default is 1 FPS, which captures significant semantic changes
                              while reducing index size by 30x compared to raw 30 FPS video.
        """
        self.target_fps = target_fps

    def extract_frames(self, video_path: str) -> typing.Iterator[typing.Tuple[float, Image.Image]]:
        """
        Extracts frames from the video at the specified target FPS using a generator.
        
        Args:
            video_path (str): The absolute or relative path to the video file.
            
        Yields:
            tuple: A tuple containing:
                   - timestamp_sec (float): The timestamp of the frame in seconds.
                   - image (Image.Image): The extracted frame as a PIL Image.
        """
        logger.info(f"Starting frame extraction for: {video_path} at {self.target_fps} FPS.")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Failed to open video file: {video_path}")

        # Get the original video frame rate
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        # Handle cases where FPS might be reported as 0 or missing
        if original_fps == 0 or not original_fps:
            original_fps = 30.0
            logger.warning(f"Could not read intrinsic FPS from video, defaulting to {original_fps}")

        # Calculate the step size (how many frames to skip)
        frame_step = int(round(original_fps / self.target_fps))
        if frame_step < 1:
            frame_step = 1

        frame_idx = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video

                # Process only the frames that align with our target sampling rate
                if frame_idx % frame_step == 0:
                    # Calculate timestamp in seconds based on actual frame index
                    timestamp_sec = frame_idx / original_fps
                    
                    # OpenCV uses BGR natively, but SigLIP processing requires RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    yield (timestamp_sec, pil_image)
                    
                frame_idx += 1
        except Exception as e:
            logger.error(f"Error occurred during video processing: {e}")
            raise
        finally:
            cap.release()
            logger.info("Video streaming and extraction finished and resources released.")

# Example usage pattern (do not run side-effects on import)
if __name__ == "__main__":
    pass
