
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app
from werkzeug.utils import secure_filename
import os
import cv2
from firebase_admin import db
from werkzeug.security import generate_password_hash
from utils.camera import init_camera, release_camera, get_video_capture
from utils.database import upload_database
from detection.face_matching import detect_faces, align_face, extract_features
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

register_bp = Blueprint('register', __name__)

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@register_bp.route("/register_upload", methods=["GET", "POST"])
def register_upload():
    release_camera()
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded")
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            ref = db.reference("Students")
            try:
                studentId = len(ref.get())
            except TypeError:
                studentId = 1

            filename = f"{studentId}.png"
            file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
            
            val, err = upload_database(filename)
            if val:
                session['filename'] = filename
                return redirect(url_for("register.add_info"))
            elif err:
                flash(err)
                return redirect(request.url)
    
    return render_template("register_upload.html")

@register_bp.route("/register_camera", methods=["GET", "POST"])
def register_camera():
    if request.method == "POST":
        init_camera()
        video_capture = get_video_capture()
        if video_capture is None:
            logger.error("No video capture available in register_camera")
            flash("Camera not available. Please try again or use file upload.")
            release_camera()
            return redirect(request.url)
        
        try:
            ret, frame = video_capture.read()
            if not ret:
                logger.error("Failed to read frame in register_camera")
                flash("Failed to capture image from camera. Please try again.")
                return redirect(request.url)
            
            ref = db.reference("Students")
            try:
                studentId = len(ref.get())
            except TypeError:
                studentId = 1

            filename = f"{studentId}.png"
            cv2.imwrite(os.path.join(current_app.config["UPLOAD_FOLDER"], filename), frame)
            
            val, err = upload_database(filename)
            if not val and err:
                flash(err)
                return redirect(request.url)
                
            session['filename'] = filename
            return redirect(url_for("register.add_info"))
        finally:
            release_camera()
    
    init_camera()
    return render_template("register_camera.html")

@register_bp.route("/add_info", methods=["GET", "POST"])
def add_info():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        userType = request.form.get("userType")
        classes = request.form.getlist("classes")
        password = request.form.get("password")

        filename = session.get('filename')
        if not filename:
            flash("No image found. Please capture or upload an image first.")
            return redirect(url_for("home.register_options"))

        fileName = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        data = cv2.imread(fileName)

        try:
            faces = detect_faces(data)
            if len(faces) == 0:
                flash("Warning: No face clearly detected, but proceeding with registration.")
                
            aligned_face = data
            if len(faces) > 0:
                aligned_face = align_face(data, faces[0])
            
            embedding = extract_features(aligned_face)
            
            if not embedding:
                flash("Warning: Could not extract facial features, but proceeding with registration.")
                embedding = [{"embedding": [0]*128}]
            
            ref = db.reference("Students")
            studentId, _ = os.path.splitext(filename)
            data = {
                str(studentId): {
                    "name": name,
                    "email": email,
                    "userType": userType,
                    "classes": {class_: int("0") for class_ in classes},
                    "password": generate_password_hash(password),
                    "embeddings": embedding[0]["embedding"],
                    "has_face": len(faces) > 0
                }
            }

            for key, value in data.items():
                ref.child(key).set(value)

            return render_template("registration_success.html", 
                                filename=filename,
                                name=name,
                                email=email,
                                userType=userType,
                                classes=classes)
            
        except Exception as e:
            flash(f"Error processing registration: {str(e)}")
            return redirect(url_for("register.add_info"))
    
    return render_template("add_info.html")

@register_bp.route("/submit_info", methods=["POST"])
def submit_info():
    release_camera()
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        userType = request.form.get("userType")
        classes = request.form.getlist("classes")
        password = request.form.get("password")

        filename = session.get('filename')
        if not filename:
            flash("No image found. Please capture or upload an image first.")
            return redirect(url_for("home.register_options"))

        fileName = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        
        try:
            data = cv2.imread(fileName)
            if data is None:
                flash("Could not read the image file. Please try again.")
                return redirect(url_for("register.add_info"))

            faces = detect_faces(data)
            if len(faces) == 0:
                flash("No face detected in the image. Please try again.")
                return redirect(url_for("register.add_info"))

            aligned_face = align_face(data, faces[0])
            embedding = extract_features(aligned_face)
            
            ref = db.reference("Students")
            studentId, _ = os.path.splitext(filename)
            user_data = {
                str(studentId): {
                    "name": name,
                    "email": email,
                    "userType": userType,
                    "classes": {class_: int("0") for class_ in classes},
                    "password": generate_password_hash(password),
                    "embeddings": embedding[0]["embedding"],
                }
            }

            ref.update(user_data)

            return render_template("registration_success.html", 
                                filename=filename,
                                name=name,
                                email=email,
                                userType=userType,
                                classes=classes)
            
        except Exception as e:
            flash(f"Error processing face: {str(e)}")
            return redirect(url_for("register.add_info"))
    
    return redirect(url_for("register.add_info"))
