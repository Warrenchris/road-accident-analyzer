import os

# Flask settings
SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'

# File upload settings
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)