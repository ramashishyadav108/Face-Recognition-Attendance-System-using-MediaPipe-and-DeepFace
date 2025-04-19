from flask import Blueprint, render_template, request, redirect, url_for, session, flash,current_app
from werkzeug.security import check_password_hash
from firebase_admin import db
from utils.camera import release_camera

teacher_bp = Blueprint('teacher', __name__)

@teacher_bp.route("/teacher_login", methods=["GET", "POST"])
def teacher_login():
    release_camera()
    if request.method == "POST":
        password = request.form.get("password")
        next_page = request.form.get("next", "")
        if check_password_hash(current_app.config["TEACHER_PASSWORD_HASH"], password):
            session['teacher_logged_in'] = True
            if next_page == "mark_attendance":
                return redirect(url_for("attendance.mark_attendance"))
            elif next_page == "records":
                return redirect(url_for("teacher.records"))
            return redirect(url_for("teacher.attendance"))
        else:
            flash("Incorrect password")
    return render_template("teacher_login.html", next=request.args.get("next", ""))

@teacher_bp.route("/records")
def records():
    release_camera()
    if not session.get('teacher_logged_in'):
        return redirect(url_for("teacher.teacher_login"))
    return redirect(url_for("teacher.attendance"))

@teacher_bp.route("/attendance", methods=["GET", "POST"])
def attendance():
    release_camera()
    if not session.get('teacher_logged_in'):
        return redirect(url_for("teacher.teacher_login"))
        
    if request.method == "POST":
        selected_class = request.form.get("class")
        selected_date = request.form.get("date")
        
        attendance_ref = db.reference("attendance")
        all_attendance = attendance_ref.get() or {}
        
        filtered_records = []
        for timestamp, record in all_attendance.items():
            if isinstance(record, dict):
                record_date = record.get("date", timestamp)[:10]
                if (not selected_class or record.get("class") == selected_class) and \
                   (not selected_date or record_date == selected_date):
                    filtered_records.append({
                        "timestamp": record.get("date", timestamp),
                        "student_name": record.get("student_name", "Unknown"),
                        "class": record.get("class", "N/A"),
                        "student_id": record.get("student_id", ""),
                        "image_name": record.get("image_name", "")
                    })
        
        student_ref = db.reference("Students")
        students_data = student_ref.get() or {}
        
        classes = set()
        for record in all_attendance.values():
            if isinstance(record, dict):
                classes.add(record.get("class", ""))
        classes.discard("")
        
        dates = set()
        for timestamp, record in all_attendance.items():
            if isinstance(record, dict):
                dates.add(record.get("date", timestamp)[:10])
        
        return render_template("attendance.html", 
                            students=students_data,
                            attendance=filtered_records,
                            classes=sorted(classes),
                            dates=sorted(dates, reverse=True),
                            selected_class=selected_class,
                            selected_date=selected_date)
    
    attendance_ref = db.reference("attendance")
    all_attendance = attendance_ref.get() or {}
    
    classes = set()
    for record in all_attendance.values():
        if isinstance(record, dict):
            classes.add(record.get("class", ""))
    classes.discard("")
    
    dates = set()
    for timestamp, record in all_attendance.items():
        if isinstance(record, dict):
            dates.add(record.get("date", timestamp)[:10])
    
    return render_template("attendance.html", 
                         students={},
                         attendance=[],
                         classes=sorted(classes),
                         dates=sorted(dates, reverse=True))

@teacher_bp.route("/student_details/<student_id>")
def student_details(student_id):
    release_camera()
    if not session.get('teacher_logged_in'):
        return redirect(url_for("teacher.teacher_login"))
    
    student_ref = db.reference(f"Students/{student_id}")
    student_info = student_ref.get() or {}
    
    attendance_ref = db.reference("attendance")
    all_attendance = attendance_ref.get() or {}
    
    attendance_by_class = {}
    for timestamp, record in all_attendance.items():
        if isinstance(record, dict) and record.get("student_id") == student_id:
            class_name = record.get("class", "Unknown")
            if class_name not in attendance_by_class:
                attendance_by_class[class_name] = []
            attendance_by_class[class_name].append({
                "date": record.get("date", timestamp),
                "class": class_name,
                "image_name": record.get("image_name", "")
            })
    
    return render_template("student_details.html",
                         student=student_info,
                         attendance_by_class=attendance_by_class,
                         student_id=student_id)