"""
Flask web server for the Facial Recognition System.
Provides REST API endpoints for the frontend.
"""
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from matching_module import FaceMatcher
import config
import json

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize face matcher
face_matcher = None

def init_face_matcher():
    """Initialize the face matcher."""
    global face_matcher
    if face_matcher is None:
        face_matcher = FaceMatcher(device='auto')
    return face_matcher

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_uploaded_file(file):
    """Save uploaded file and return the path."""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add unique prefix to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        return filepath
    return None

@app.route('/')
def index():
    """Serve the main frontend page."""
    return render_template('index.html')

@app.route('/api/enroll', methods=['POST'])
def enroll_user():
    """Enroll a new user."""
    try:
        matcher = init_face_matcher()
        
        # Check if files are present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'}), 400
        
        # Get user ID
        user_id = request.form.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID is required'}), 400
        
        # Save uploaded file
        image_path = save_uploaded_file(file)
        if not image_path:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Get metadata
        metadata = {}
        if request.form.get('metadata'):
            try:
                metadata = json.loads(request.form.get('metadata'))
            except json.JSONDecodeError:
                pass
        
        # Enroll user
        result = matcher.enroll_user(
            user_id=user_id,
            image_path=image_path,
            metadata=metadata
        )
        
        # Clean up uploaded file
        try:
            os.remove(image_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/authenticate', methods=['POST'])
def authenticate_user():
    """Authenticate a user."""
    try:
        matcher = init_face_matcher()
        
        # Check if files are present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'}), 400
        
        # Get threshold
        threshold = request.form.get('threshold', config.SIMILARITY_THRESHOLD)
        try:
            threshold = float(threshold)
        except ValueError:
            threshold = config.SIMILARITY_THRESHOLD
        
        # Save uploaded file
        image_path = save_uploaded_file(file)
        if not image_path:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Authenticate user
        result = matcher.authenticate_user(
            image_path=image_path,
            threshold=threshold
        )
        
        # Clean up uploaded file
        try:
            os.remove(image_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/identify', methods=['POST'])
def identify_user():
    """Identify a user from their face."""
    try:
        matcher = init_face_matcher()
        
        # Check if files are present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'}), 400
        
        # Get threshold
        threshold = request.form.get('threshold', config.SIMILARITY_THRESHOLD)
        try:
            threshold = float(threshold)
        except ValueError:
            threshold = config.SIMILARITY_THRESHOLD
        
        # Save uploaded file
        image_path = save_uploaded_file(file)
        if not image_path:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Identify user
        result = matcher.identify_user(
            image_path=image_path,
            threshold=threshold
        )
        
        # Clean up uploaded file
        try:
            os.remove(image_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_faces():
    """Compare two face images."""
    try:
        matcher = init_face_matcher()
        
        # Check if files are present
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'success': False, 'error': 'Two image files are required'}), 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        if file1.filename == '' or file2.filename == '':
            return jsonify({'success': False, 'error': 'Both image files must be selected'}), 400
        
        # Save uploaded files
        image1_path = save_uploaded_file(file1)
        image2_path = save_uploaded_file(file2)
        
        if not image1_path or not image2_path:
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Compare faces
        result = matcher.compare_faces(image1_path, image2_path)
        
        # Clean up uploaded files
        try:
            os.remove(image1_path)
            os.remove(image2_path)
        except:
            pass
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    try:
        matcher = init_face_matcher()
        result = matcher.get_system_stats()
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/<user_id>', methods=['GET'])
def get_user_info(user_id):
    """Get user information."""
    try:
        matcher = init_face_matcher()
        result = matcher.get_user_info(user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    """Delete a user."""
    try:
        matcher = init_face_matcher()
        result = matcher.delete_user(user_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Facial Recognition API is running'})

if __name__ == '__main__':
    print("Starting Facial Recognition Web Server...")
    print("Initializing face matcher...")
    init_face_matcher()
    print("Server ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)
