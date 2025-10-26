"""
Flask web server for the Facial Recognition System.
Provides REST API endpoints for the frontend.
"""
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS
import os
import uuid
import time
from werkzeug.utils import secure_filename
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.matching_module import FaceMatcher
from src.core import config
import json

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize face matcher globally
face_matcher = None

def get_face_matcher():
    """Get or initialize the face matcher."""
    global face_matcher
    if face_matcher is None:
        print("Initializing Face Recognition System...")
        face_matcher = FaceMatcher(device='auto')
        print("Face Recognition System ready!")
    return face_matcher

def init_face_matcher():
    """Initialize the face matcher."""
    return get_face_matcher()

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
    """Enroll a new user with real face processing."""
    start_time = time.time()
    
    try:
        matcher = init_face_matcher()
        
        # Get user ID
        user_id = request.form.get('user_id', '').strip()
        if not user_id:
            return jsonify({'success': False, 'error': 'User ID is required'}), 400
        
        # Validate user ID format
        if not user_id.replace('_', '').replace('-', '').isalnum():
            return jsonify({'success': False, 'error': 'User ID must contain only letters, numbers, hyphens, and underscores'}), 400
        
        # Get metadata
        metadata = {'enrolled_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        if request.form.get('metadata'):
            try:
                user_metadata = json.loads(request.form.get('metadata'))
                metadata.update(user_metadata)
            except json.JSONDecodeError as e:
                return jsonify({'success': False, 'error': f'Invalid JSON in metadata: {str(e)}'}), 400
        
        # Process image data
        image_data = None
        image_path = None
        
        # Check for camera data first
        camera_data = request.form.get('camera_data')
        if camera_data:
            image_data = camera_data
            print(f"Processing camera capture for user: {user_id}")
        
        # Check for uploaded file
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({'success': False, 'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
                
                # Save uploaded file temporarily
                image_path = save_uploaded_file(file)
                if not image_path:
                    return jsonify({'success': False, 'error': 'Failed to save uploaded file'}), 400
                print(f"Processing uploaded file for user: {user_id}")
        
        if not image_data and not image_path:
            return jsonify({'success': False, 'error': 'No image provided. Please capture a photo or upload an image file.'}), 400
        
        # Enroll user with real processing
        print(f"Starting enrollment process for user: {user_id}")
        result = matcher.enroll_user(
            user_id=user_id,
            image_path=image_path,
            image_data=image_data,
            metadata=metadata,
            save_image=True
        )
        
        # Add processing time to result
        processing_time = time.time() - start_time
        result['processing_time_ms'] = round(processing_time * 1000, 2)
        
        # Clean up temporary uploaded file
        if image_path:
            try:
                os.remove(image_path)
            except:
                pass
        
        # Add success details
        if result['success']:
            result['message'] = f"User '{user_id}' enrolled successfully with {result['faces_detected']} face(s) detected"
            print(f"✅ User {user_id} enrolled successfully in {processing_time:.2f}s")
        else:
            print(f"❌ Enrollment failed for user {user_id}: {result.get('error', 'Unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Server error during enrollment: {str(e)}"
        print(f"❌ Enrollment error: {error_msg}")
        return jsonify({
            'success': False, 
            'error': error_msg,
            'processing_time_ms': round(processing_time * 1000, 2)
        }), 500

@app.route('/api/authenticate', methods=['POST'])
def authenticate_user():
    """Authenticate a user with real face processing."""
    start_time = time.time()
    
    try:
        matcher = init_face_matcher()
        
        # Get threshold
        threshold = request.form.get('threshold', config.SIMILARITY_THRESHOLD)
        try:
            threshold = float(threshold)
            if threshold < 0 or threshold > 1:
                return jsonify({'success': False, 'error': 'Threshold must be between 0.0 and 1.0'}), 400
        except ValueError:
            threshold = config.SIMILARITY_THRESHOLD
        
        # Process image data
        image_data = None
        image_path = None
        
        # Check for camera data first
        camera_data = request.form.get('camera_data')
        if camera_data:
            image_data = camera_data
            print("Processing camera capture for authentication")
        
        # Check for uploaded file
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename != '':
                if not allowed_file(file.filename):
                    return jsonify({'success': False, 'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400
                
                # Save uploaded file temporarily
                image_path = save_uploaded_file(file)
                if not image_path:
                    return jsonify({'success': False, 'error': 'Failed to save uploaded file'}), 400
                print("Processing uploaded file for authentication")
        
        if not image_data and not image_path:
            return jsonify({'success': False, 'error': 'No image provided. Please capture a photo or upload an image file.'}), 400
        
        # Authenticate user with real processing
        print(f"Starting authentication with threshold: {threshold}")
        result = matcher.authenticate_user(
            image_path=image_path,
            image_data=image_data,
            threshold=threshold
        )
        
        # Add processing time to result
        processing_time = time.time() - start_time
        result['processing_time_ms'] = round(processing_time * 1000, 2)
        result['threshold_used'] = threshold
        
        # Clean up temporary uploaded file
        if image_path:
            try:
                os.remove(image_path)
            except:
                pass
        
        # Add detailed response messages
        if result['success']:
            if result.get('authenticated', False):
                user_id = result.get('user_id', 'Unknown')
                similarity = result.get('similarity', 0)
                result['message'] = f"Authentication successful! User: {user_id} (Similarity: {similarity:.3f})"
                print(f"✅ Authentication successful for user: {user_id} in {processing_time:.2f}s")
            else:
                result['message'] = "Authentication failed: No matching user found"
                print(f"❌ Authentication failed: No match found in {processing_time:.2f}s")
        else:
            print(f"❌ Authentication error: {result.get('error', 'Unknown error')}")
        
        return jsonify(result)
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Server error during authentication: {str(e)}"
        print(f"❌ Authentication error: {error_msg}")
        return jsonify({
            'success': False, 
            'error': error_msg,
            'processing_time_ms': round(processing_time * 1000, 2)
        }), 500

@app.route('/api/identify', methods=['POST'])
def identify_user():
    """Identify a user from their face."""
    try:
        matcher = init_face_matcher()
        
        # Get threshold
        threshold = request.form.get('threshold', config.SIMILARITY_THRESHOLD)
        try:
            threshold = float(threshold)
        except ValueError:
            threshold = config.SIMILARITY_THRESHOLD
        
        # Check for camera data first
        camera_data = request.form.get('camera_data')
        if camera_data:
            # Process camera capture
            result = matcher.identify_user(
                image_data=camera_data,
                threshold=threshold
            )
            return jsonify(result)
        
        # Check if files are present
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file or camera data provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'}), 400
        
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

@app.route('/api/stats', methods=['GET'])
def get_system_stats():
    """Get detailed system statistics."""
    try:
        matcher = init_face_matcher()
        stats = matcher.get_system_stats()
        
        # Add additional system info
        if stats['success']:
            stats.update({
                'system_info': {
                    'version': '1.0.0',
                    'encryption': 'AES-256',
                    'face_detection': 'OpenCV Haar Cascades',
                    'face_embedding': 'Custom Feature Extraction',
                    'storage': 'ChromaDB Vector Database'
                }
            })
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return send_file('static/favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Facial Recognition Web Server...")
    print("Initializing face matcher...")
    init_face_matcher()
    print("Server ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)
