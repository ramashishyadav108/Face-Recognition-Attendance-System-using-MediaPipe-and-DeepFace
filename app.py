from flask import Flask, session
from firebase_admin import credentials, initialize_app
from utils.configuration import load_yaml
from utils.camera import release_camera
from routes.home import home_bp
from routes.register import register_bp
from routes.attendance import attendance_bp
from routes.teacher import teacher_bp
import atexit

# Load configuration
config_file_path = load_yaml("/home/ram/Rama/attendance/vb1/ifrs/configs/database.yaml")

# Initialize Firebase
cred = credentials.Certificate(config_file_path["firebase"]["pathToServiceAccount"])
initialize_app(
    cred,
    {
        "databaseURL": config_file_path["firebase"]["databaseURL"],
    },
)

# Initialize Flask app
app = Flask(__name__, template_folder="templates")
app.secret_key = "123456"
app.config["UPLOAD_FOLDER"] = "static/images"
app.config["TEACHER_PASSWORD_HASH"] = config_file_path["teacher"]["password_hash"]

# Register blueprints
app.register_blueprint(home_bp)
app.register_blueprint(register_bp)
app.register_blueprint(attendance_bp)
app.register_blueprint(teacher_bp)

# Ensure camera is released on shutdown
atexit.register(release_camera)

if __name__ == "__main__":
    app.run(debug=True)