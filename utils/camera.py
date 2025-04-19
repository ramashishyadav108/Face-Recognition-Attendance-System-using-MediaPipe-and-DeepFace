import cv2
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

video_capture = None

def init_camera(max_retries=3, retry_delay=1):
    """Initialize the camera capture with optimized settings and retries"""
    global video_capture
    if video_capture is None or not video_capture.isOpened():
        for attempt in range(max_retries):
            try:
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    raise RuntimeError("Could not open video source")
                
                # Set optimal camera properties
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                video_capture.set(cv2.CAP_PROP_FPS, 15)
                video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # Increased buffer size
                time.sleep(0.1)  # Brief delay to ensure camera is ready
                logger.info(f"Camera initialized successfully: {video_capture.isOpened()}")
                return True
            except RuntimeError as e:
                logger.error(f"Camera initialization attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                video_capture = None
        logger.error("Failed to initialize camera after all retries")
        return False
    return True

def get_video_capture():
    """Return video_capture, initializing if necessary"""
    if video_capture is None or not video_capture.isOpened():
        success = init_camera()
        if not success:
            return None
    logger.debug(f"Video capture state: {video_capture.isOpened()}")
    return video_capture

def release_camera():
    """Release the camera resource"""
    global video_capture
    if video_capture is not None:
        video_capture.release()
        video_capture = None
        logger.info("Camera released")