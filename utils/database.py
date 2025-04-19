from firebase_admin import db
import datetime

def upload_database(filename):
    """Upload attendance data to Firebase"""
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        ref = db.reference(f'/attendance/{timestamp}')
        ref.set({
            'image_name': filename,
            'timestamp': timestamp
        })
        return True, None
    except Exception as e:
        return False, str(e)

def load_student_database():
    """Load student data from Firebase"""
    ref = db.reference("Students")
    students_data = ref.get()
    database = {}
    
    if students_data:
        if isinstance(students_data, list):
            for i in range(1, len(students_data)):
                if students_data[i] and isinstance(students_data[i], dict):
                    student = students_data[i]
                    if "name" in student and "embeddings" in student:
                        database[student["name"]] = {
                            "embeddings": student["embeddings"],
                            "id": i
                        }
        else:
            for k, v in students_data.items():
                if k.isdigit() and isinstance(v, dict) and "name" in v and "embeddings" in v:
                    database[v["name"]] = {
                        "embeddings": v["embeddings"],
                        "id": k
                    }
    return database