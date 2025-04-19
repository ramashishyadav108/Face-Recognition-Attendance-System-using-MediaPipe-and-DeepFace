import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cosine
from detection.face_matching import detect_faces, align_face, extract_features, match_face
from .camera import get_video_capture, init_camera
from .database import load_student_database
import logging
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mp_face_detection = mp.solutions.face_detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# In-memory store for recognized students
_recognition_counts = defaultdict(int)
_marked_students = []

def match_with_database(img, database):
    faces = detect_faces(img)
    cv2.imwrite("static/recognized/recognized.png", img)

    if len(faces) == 0:
        try:
            embedding = extract_features(img)
            if embedding and len(embedding) > 0:
                match = match_face(embedding[0]["embedding"], database)
                if match is not None:
                    return "Match found (low confidence): " + match
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
        return "No clear face detected"

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
        try:
            aligned_face = align_face(img, (x, y, w, h))
            embedding = extract_features(aligned_face)
            if embedding and len(embedding) > 0:
                match = match_face(embedding[0]["embedding"], database)
                if match is not None:
                    return f"Match found: {match}"
        except Exception as e:
            logger.error(f"Face processing error: {str(e)}")
            continue
    
    return "No match found"

def gen_frames():
    init_camera()
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    database = load_student_database()
    frame_skip = 2
    frame_count = 0
    read_retries = 3
    retry_delay = 0.1
    
    global _recognition_counts, _marked_students
    _recognition_counts.clear()
    _marked_students.clear()
    
    try:
        while True:
            video_capture = get_video_capture()
            if video_capture is None:
                logger.error("No video capture available")
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Camera not available", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode(".jpg", blank_frame)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(1)
                continue
            
            success = False
            frame = None
            for attempt in range(read_retries):
                success, frame = video_capture.read()
                if success:
                    break
                logger.warning(f"Frame read attempt {attempt + 1} failed")
                time.sleep(retry_delay)
            
            if not success:
                logger.error("Failed to read frame from video capture after retries")
                blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Failed to read frame", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode(".jpg", blank_frame)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(1)
                continue
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                ret, buffer = cv2.imencode(".jpg", frame)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                continue
            
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = small_frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    faces.append((x, y, w, h))
            
            if not faces:
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(20, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
            
            faces = [(int(x*2), int(y*2), int(w*2), int(h*2)) for (x, y, w, h) in faces]
            recognized_students = []
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                try:
                    face_region = frame[y:y+h, x:x+w]
                    if w > 100 and h > 100:
                        aligned_face = align_face(frame, (x, y, w, h))
                        embedding = extract_features(aligned_face)
                        if embedding and len(embedding) > 0:
                            min_distance = float('inf')
                            best_match = None
                            for name, db_embedding in database.items():
                                distance = cosine(embedding[0]["embedding"], db_embedding["embeddings"])
                                if distance < min_distance:
                                    min_distance = distance
                                    best_match = name
                            confidence = 1 - min_distance
                            if confidence > 0.5:
                                recognized_students.append(best_match)
                                match_text = f"{best_match} ({confidence:.2f})"
                                cv2.putText(frame, match_text, (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                           (0, 255, 0), 2)
                                if best_match in _marked_students:
                                    cv2.putText(frame, "Attendance Marked", (x, y+h+20), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except Exception as e:
                    logger.error(f"Recognition error: {str(e)}")
            
            for student in recognized_students:
                _recognition_counts[student] += 1
                if _recognition_counts[student] >= 5 and student not in _marked_students:
                    _marked_students.append(student)
            
            ret, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        face_detection.close()

def get_recognition_data():
    return {
        'recognition_counts': dict(_recognition_counts),
        'marked_students': _marked_students.copy()
    }

def clear_recognition_data():
    global _recognition_counts, _marked_students
    _recognition_counts.clear()
    _marked_students.clear()