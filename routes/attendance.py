from flask import Blueprint, render_template, request, redirect, url_for, Response, session, current_app, flash, jsonify
import cv2
import mediapipe as mp
from scipy.spatial.distance import cosine
import time
import datetime
from firebase_admin import db
from utils.camera import init_camera, release_camera, get_video_capture
from utils.database import load_student_database
from utils.face_recognition import gen_frames, get_recognition_data, clear_recognition_data
from detection.face_matching import detect_faces, align_face, extract_features
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

attendance_bp = Blueprint('attendance', __name__)

mp_face_detection = mp.solutions.face_detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@attendance_bp.route("/mark_attendance")
def mark_attendance():
    release_camera()
    if not session.get('teacher_logged_in'):
        return redirect(url_for("teacher.teacher_login"))
    return render_template("select_class.html")

@attendance_bp.route("/attendance_camera")
def attendance_camera():
    if not session.get('teacher_logged_in'):
        return redirect(url_for("teacher.teacher_login"))
    if 'selected_class' not in session:
        return redirect(url_for("attendance.mark_attendance"))
    return render_template("attendance_camera.html",
                           start_time=session.get('attendance_start_time', time.time()),
                           duration=session.get('attendance_duration', 300))

@attendance_bp.route("/start_attendance", methods=["POST"])
def start_attendance():
    if not session.get('teacher_logged_in'):
        return redirect(url_for("teacher.teacher_login"))
    
    selected_class = request.form.get("classes")
    duration = request.form.get("duration")
    if not selected_class or not duration:
        flash("Please select a class and duration")
        return redirect(url_for("attendance.mark_attendance"))
    
    session['selected_class'] = selected_class
    session['attendance_duration'] = int(duration) * 60  # Convert minutes to seconds
    session['attendance_start_time'] = time.time()
    session['recognition_counts'] = {}
    session['marked_students'] = []
    clear_recognition_data()  # Reset in-memory store
    return redirect(url_for("attendance.attendance_camera"))

@attendance_bp.route("/get_marked_students")
def get_marked_students():
    if not session.get('teacher_logged_in'):
        return jsonify([])
    recognition_data = get_recognition_data()
    return jsonify(recognition_data['marked_students'])

@attendance_bp.route("/sync_recognition_data", methods=["GET"])
def sync_recognition_data():
    if not session.get('teacher_logged_in'):
        return jsonify({'recognition_counts': {}, 'marked_students': []})
    recognition_data = get_recognition_data()
    session['recognition_counts'] = recognition_data['recognition_counts']
    session['marked_students'] = recognition_data['marked_students']
    session.modified = True
    return jsonify(recognition_data)

@attendance_bp.route("/finish_attendance", methods=["POST"])
def finish_attendance():
    if not session.get('teacher_logged_in'):
        return redirect(url_for("teacher.teacher_login"))
    if 'selected_class' not in session:
        return redirect(url_for("attendance.mark_attendance"))
    
    # Sync in-memory data to session
    recognition_data = get_recognition_data()
    session['recognition_counts'] = recognition_data['recognition_counts']
    session['marked_students'] = recognition_data['marked_students']
    session.modified = True
    
    selected_class = session['selected_class']
    marked_students = session.get('marked_students', [])
    
    ref = db.reference("Students")
    students_data = ref.get() or {}
    
    # Fix: Check if students_data is a list and convert it appropriately
    if isinstance(students_data, list):
        students_dict = {}
        for i, student in enumerate(students_data):
            if student:  # Make sure the student data exists
                students_dict[str(i)] = student
        students_data = students_dict
    
    # Initialize attendance_records and enrollment_messages
    session_attendance_records = []
    enrollment_messages = []
    
    # Debug: Print the number of marked students
    logger.info(f"Processing {len(marked_students)} marked students")
    
    for student_name in marked_students:
        student_found = False
        for student_key, studentInfo in students_data.items():
            if student_key.isdigit() and isinstance(studentInfo, dict) and "name" in studentInfo and studentInfo["name"] == student_name:
                student_found = True
                if "classes" in studentInfo and selected_class in studentInfo["classes"]:
                    current_attendance = int(studentInfo["classes"].get(selected_class, 0))
                    ref.child(f"{student_key}/classes/{selected_class}").set(current_attendance + 1)
                    
                    # Create a unique key for each attendance record
                    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                    record_key = f"{timestamp}_{student_key}"
                    readable_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    attendance_record = {
                        "student_name": student_name,
                        "student_id": student_key,
                        "class": selected_class,
                        "date": readable_date,
                        "timestamp": timestamp,
                        "image_name": f"{student_key}.png"
                    }
                    # Store with unique key
                    db.reference(f"attendance/{record_key}").set(attendance_record)
                    # Add to our collection for display
                    session_attendance_records.append(attendance_record)
                    logger.info(f"Added attendance record for {student_name}")
                else:
                    message = f"Student {student_name} is not enrolled in {selected_class}"
                    enrollment_messages.append(message)
                    logger.info(message)
                break
        if not student_found:
            message = f"Student {student_name} not found in database"
            enrollment_messages.append(message)
            logger.info(message)
    
    # Debug: Print the number of records collected in this session
    logger.info(f"Collected {len(session_attendance_records)} attendance records in this session")
    logger.info(f"Found {len(enrollment_messages)} enrollment issues")
    
    # Clean up
    session.pop('recognition_counts', None)
    session.pop('marked_students', None)
    session.pop('attendance_start_time', None)
    session.pop('attendance_duration', None)
    session.pop('selected_class', None)
    clear_recognition_data()
    
    release_camera()
    # Pass both attendance records and enrollment messages to the template
    return render_template("attendance_summary.html", 
                          attendance_records=session_attendance_records,
                          enrollment_messages=enrollment_messages,
                          selected_class=selected_class)

@attendance_bp.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")