import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Create face detector instances for reuse
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for closer faces, 1 for faces further away
    min_detection_confidence=0.5
)

# Load the cascade as backup
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(img):
    '''
    Face detection using MediaPipe first, then falling back to Haar Cascade
    '''
    # Convert to RGB for MediaPipe
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Try MediaPipe face detection first (more accurate)
    detection_result = face_detection.process(rgb_img)
    faces = []
    
    if detection_result.detections:
        for detection in detection_result.detections:
            # Get bounding box from detection
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            faces.append((x, y, w, h))
        return faces
    
    # If MediaPipe fails, try Haar cascade (faster but less accurate)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar_faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    if len(haar_faces) > 0:
        return haar_faces
    
    return []

def align_face(img, face):
    '''
    Align face using MediaPipe face mesh landmarks instead of dlib
    '''
    x, y, w, h = face
    face_roi = img[y:y+h, x:x+w]
    
    # Check if ROI is valid
    if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
        return cv2.resize(img, (256, 256))
    
    # Convert to RGB for MediaPipe
    rgb_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    
    # Get face mesh landmarks
    result = face_mesh.process(rgb_roi)
    
    if not result.multi_face_landmarks:
        # If no landmarks found, just resize the face ROI
        return cv2.resize(face_roi, (256, 256))
    
    landmarks = result.multi_face_landmarks[0].landmark
    
    # Get coordinates for eyes (using MediaPipe face mesh indices)
    # Left eye indices: 33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7
    # Right eye indices: 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382
    
    # Using just a subset of points for center calculation
    left_eye_indices = [33, 246, 161, 160, 159, 158]
    right_eye_indices = [362, 398, 384, 385, 386, 387]
    
    # Calculate eye centers
    left_eye_points = []
    right_eye_points = []
    
    h_roi, w_roi, _ = face_roi.shape
    
    for idx in left_eye_indices:
        point = landmarks[idx]
        x_coord, y_coord = int(point.x * w_roi), int(point.y * h_roi)
        left_eye_points.append((x_coord, y_coord))
    
    for idx in right_eye_indices:
        point = landmarks[idx]
        x_coord, y_coord = int(point.x * w_roi), int(point.y * h_roi)
        right_eye_points.append((x_coord, y_coord))
    
    left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
    right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
    
    # Calculate angle between eyes
    dY = right_eye_center[1] - left_eye_center[1]
    dX = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    # Desired face dimensions
    desired_face_width = 256
    desired_face_height = 256
    
    # Calculate scale
    dist = np.sqrt((dX**2) + (dY**2))
    desired_dist = desired_face_width * 0.27
    scale = desired_dist / max(1, dist)  # Avoid division by zero
    
    # Center point of eyes
    eyes_center = (
        int((left_eye_center[0] + right_eye_center[0]) // 2),
        int((left_eye_center[1] + right_eye_center[1]) // 2),
    )
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    
    # Update translation component
    tX = desired_face_width * 0.5
    tY = desired_face_height * 0.3
    M[0, 2] += tX - eyes_center[0]
    M[1, 2] += tY - eyes_center[1]
    
    # Apply the affine transformation
    output = cv2.warpAffine(face_roi, M, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)
    
    return output

def extract_features(face):
    '''
    Extract facial embeddings using DeepFace
    '''
    try:
        # Check if face is valid
        if face is None or face.size == 0:
            return None
            
        # Convert to RGB (DeepFace expects RGB)
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Ensure minimum size
        if face_rgb.shape[0] < 64 or face_rgb.shape[1] < 64:
            face_rgb = cv2.resize(face_rgb, (160, 160))
        
        # Use DeepFace for feature extraction
        embedding = DeepFace.represent(
            face_rgb, 
            model_name="Facenet", 
            enforce_detection=False,
            detector_backend="opencv"
        )
        
        return embedding
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return None

def match_face(embedding, database):
    '''
    Improved face matching with dynamic thresholding
    '''
    if isinstance(database, dict) and not all(isinstance(v, list) for v in database.values()):
        # Handle nested structure for database with additional info
        distances = {}
        for name, data in database.items():
            if isinstance(data, dict) and "embeddings" in data:
                distance = cosine(embedding, data["embeddings"])
                distances[name] = distance
            elif isinstance(data, list):
                distance = cosine(embedding, data)
                distances[name] = distance
    else:
        # Original behavior for simple name:embedding dictionary
        distances = {}
        for name, db_embedding in database.items():
            distance = cosine(embedding, db_embedding)
            distances[name] = distance
    
    # Find the best match
    if not distances:
        return None
        
    best_match = min(distances.items(), key=lambda x: x[1])
    name, min_distance = best_match
    
    # Calculate the difference between the best match and the second best
    other_distances = [d for n, d in distances.items() if n != name]
    second_best = min(other_distances) if other_distances else 1.0
    
    # Dynamic threshold based on match quality
    confidence = 1 - min_distance
    threshold = 0.45  # Base threshold
    
    # If the best match is significantly better than the second best, be more lenient
    if (second_best - min_distance) > 0.2:
        threshold = 0.4
        
    # Return the match if confidence is high enough
    if confidence > threshold:
        return name
    else:
        return None