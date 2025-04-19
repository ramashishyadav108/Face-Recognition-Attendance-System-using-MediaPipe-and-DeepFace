from flask import Blueprint, render_template
from utils.camera import release_camera

home_bp = Blueprint('home', __name__)

@home_bp.route("/")
def home():
    release_camera()
    return render_template("home.html")

@home_bp.route("/register_options")
def register_options():
    release_camera()
    return render_template("register_options.html")