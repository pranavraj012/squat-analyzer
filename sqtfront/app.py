import os
import time
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from process_frame_front import ProcessFrameFront
from thresholds import get_thresholds_beginner, get_thresholds_pro
from thresholds_front import get_thresholds_beginner_front, get_thresholds_pro_front


def create_processor(mode: str, view_type: str, exercise_type: str, flip_frame: bool = True):
    """
    Create appropriate processor based on mode, view_type, and exercise_type parameters.
    
    Args:
        mode (str): Processing mode - 'Beginner' or 'Pro'
        view_type (str): Camera view type - 'side' or 'front'
        exercise_type (str): Exercise type - currently supports 'squat'
        flip_frame (bool): Whether to flip the frame horizontally
        
    Returns:
        ProcessFrame or ProcessFrameFront: Appropriate processor instance
        
    Raises:
        ValueError: If invalid parameter combinations are provided
    """
    # Validate input parameters
    valid_modes = ['Beginner', 'Pro']
    valid_view_types = ['side', 'front']
    valid_exercise_types = ['squat']  # Currently only squat is supported
    
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
    
    if view_type not in valid_view_types:
        raise ValueError(f"Invalid view_type '{view_type}'. Must be one of: {valid_view_types}")
    
    if exercise_type not in valid_exercise_types:
        raise ValueError(f"Invalid exercise_type '{exercise_type}'. Must be one of: {valid_exercise_types}")
    
    try:
        if view_type == 'side':
            # Side view processing - use ProcessFrame with appropriate thresholds
            if mode == 'Beginner':
                thresholds = get_thresholds_beginner()
            else:  # mode == 'Pro'
                thresholds = get_thresholds_pro()
            
            return ProcessFrame(thresholds=thresholds, flip_frame=flip_frame)
        
        else:  # view_type == 'front'
            # Front view processing - use ProcessFrameFront with appropriate thresholds
            if mode == 'Beginner':
                thresholds = get_thresholds_beginner_front()
            else:  # mode == 'Pro'
                thresholds = get_thresholds_pro_front()
            
            return ProcessFrameFront(thresholds=thresholds, flip_frame=flip_frame)
    
    except Exception as e:
        raise ValueError(f"Failed to create processor: {str(e)}")


# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'outputs')
app.config['IMAGE_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'img')

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['IMAGE_FOLDER'], exist_ok=True)

# Global variables
camera = None
live_process_frame = None
pose = get_mediapipe_pose()
current_exercise_type = 'squat'  # Default exercise type
current_view_type = 'side'  # Default view type (side or front)

# Routes for traditional Flask templates (legacy)
@app.route('/legacy')
def legacy_index():
    return render_template('index.html')

@app.route('/legacy/live_stream')
def legacy_live_stream():
    return render_template('live_stream.html')

@app.route('/legacy/upload_video')
def legacy_upload_video():
    return render_template('upload_video.html')

# Main route for SPA frontend
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    # This will handle all requests for React routes
    # in a production environment you'd serve the React build
    # For development, we'll just redirect to the React dev server
    if os.environ.get('FLASK_ENV') == 'production':
        return send_from_directory('frontend/dist', 'index.html')
    return jsonify({'message': 'API is running, frontend should be at http://localhost:3000'})

# API Routes
@app.route('/api/modes', methods=['GET'])
def get_modes():
    return jsonify({
        'modes': [
            {'id': 'Beginner', 'name': 'Beginner Mode'},
            {'id': 'Pro', 'name': 'Professional Mode'}
        ]
    })

@app.route('/api/exercises', methods=['GET'])
def get_exercises():
    return jsonify({
        'exercises': [
            {
                'id': 'squat',
                'name': 'Squat Analysis',
                'description': 'Perfect your squat form with AI feedback.',
                'image': f"{request.host_url}static/img/squat.jpg"
            },
            {
                'id': 'plank',
                'name': 'Plank Form',
                'description': 'Maintain proper plank position with real-time posture correction.',
                'image': f"{request.host_url}static/img/plank.jpg"
            }
        ]
    })

@app.route('/api/views', methods=['GET'])
def get_views():
    return jsonify({
        'views': [
            {'id': 'side', 'name': 'Side View'},
            {'id': 'front', 'name': 'Front View'}
        ]
    })

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global live_process_frame, current_exercise_type, current_view_type
    
    try:
        mode = request.json.get('mode', 'Beginner')
        exercise_type = request.json.get('exerciseType', 'squat')
        view_type = request.json.get('viewType', 'side')
        
        # Use the processor factory to create the appropriate processor
        processor = create_processor(mode, view_type, exercise_type, flip_frame=True)
        
        # Validate that processor was created successfully
        if processor is None:
            return jsonify({
                'status': 'error',
                'message': 'Failed to create processor'
            }), 500
        
        # Update global state only after successful processor creation
        current_exercise_type = exercise_type
        current_view_type = view_type
        live_process_frame = processor
        
        return jsonify({
            'status': 'success', 
            'mode': mode, 
            'exerciseType': exercise_type,
            'viewType': view_type
        })
        
    except ValueError as e:
        # Handle validation errors from create_processor
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
        
    except Exception as e:
        # Handle any other unexpected errors
        return jsonify({
            'status': 'error',
            'message': f'Failed to set mode: {str(e)}'
        }), 500

@app.route('/api/video_feed_url')
def get_video_feed_url():
    exercise_type = request.args.get('exercise', 'squat')
    view_type = request.args.get('view', 'side')
    return jsonify({
        'url': f"http://{request.host}/video_feed?exercise={exercise_type}&view={view_type}"
    })

def generate_frames():
    global camera, live_process_frame, current_view_type
    
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        try:
            success, frame = camera.read()
            if not success:
                break
            
            # Check if we have a valid processor
            if live_process_frame is None:
                # Skip frame if no processor is set
                continue
            
            # Convert BGR to RGB for consistent processing (matching upload logic)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame - both ProcessFrame and ProcessFrameFront now expect RGB input
            out_frame, _ = live_process_frame.process(frame_rgb, pose)
            
            # Convert back to BGR for video output
            out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            
            # Encode as jpeg
            ret, buffer = cv2.imencode('.jpg', out_frame_bgr)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
        except Exception as e:
            # Log error and continue with next frame
            print(f"Frame processing error: {e}")
            continue
        
        # Small delay to reduce CPU usage
        time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    global current_exercise_type, current_view_type, live_process_frame
    
    # Validate that processor is initialized before starting stream
    if live_process_frame is None:
        return jsonify({
            'status': 'error',
            'message': 'Processor not initialized. Please set mode first using /set_mode endpoint.'
        }), 400
    
    exercise_type = request.args.get('exercise', 'squat')
    view_type = request.args.get('view', 'side')
    
    # Ensure current_exercise_type and current_view_type are synchronized with the processor
    # Only update if they differ from current values to maintain consistency
    if exercise_type != current_exercise_type or view_type != current_view_type:
        # Parameters have changed, need to validate they match the current processor
        # For now, we'll update the global variables but in a production system
        # we might want to recreate the processor or return an error
        current_exercise_type = exercise_type
        current_view_type = view_type
    
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera, current_exercise_type, current_view_type, live_process_frame
    
    try:
        # Get parameters from request
        exercise_type = request.args.get('exercise', 'squat')
        view_type = request.args.get('view', 'side')
        
        # Validate that the processor is properly configured before starting camera
        if live_process_frame is None:
            return jsonify({
                'status': 'error',
                'message': 'Processor not initialized. Please set mode first using /set_mode endpoint.'
            }), 400
        
        # Validate that current processor configuration matches requested parameters
        if exercise_type != current_exercise_type or view_type != current_view_type:
            return jsonify({
                'status': 'error',
                'message': f'Processor configuration mismatch. Current: {current_exercise_type}/{current_view_type}, Requested: {exercise_type}/{view_type}. Please update mode first.'
            }), 400
        
        # Initialize camera if not already initialized
        if camera is None:
            try:
                camera = cv2.VideoCapture(0)
                
                # Check if camera was successfully opened
                if not camera.isOpened():
                    camera = None
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to open camera. Please check if camera is connected and not being used by another application.'
                    }), 500
                
                # Set camera settings to match upload processing requirements
                # These settings ensure consistent resolution and frame rate
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_FPS, 30)  # Set consistent frame rate
                
                # Verify that camera settings were applied successfully
                actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                actual_fps = camera.get(cv2.CAP_PROP_FPS)
                
                # Test camera by reading a frame to ensure it's working
                ret, test_frame = camera.read()
                if not ret or test_frame is None:
                    camera.release()
                    camera = None
                    return jsonify({
                        'status': 'error',
                        'message': 'Camera opened but failed to capture frames. Please check camera functionality.'
                    }), 500
                
                print(f"Camera initialized successfully - Resolution: {int(actual_width)}x{int(actual_height)}, FPS: {int(actual_fps)}")
                
            except Exception as e:
                # Clean up camera resource if initialization failed
                if camera is not None:
                    try:
                        camera.release()
                    except:
                        pass
                    camera = None
                
                return jsonify({
                    'status': 'error',
                    'message': f'Camera initialization failed: {str(e)}'
                }), 500
        
        # Update global state variables
        current_exercise_type = exercise_type
        current_view_type = view_type
        
        return jsonify({
            'status': 'started', 
            'exerciseType': current_exercise_type,
            'viewType': current_view_type,
            'message': 'Camera started successfully'
        })
        
    except Exception as e:
        # Handle any unexpected errors
        return jsonify({
            'status': 'error',
            'message': f'Unexpected error starting camera: {str(e)}'
        }), 500

@app.route('/stop_camera')
def stop_camera():
    global camera, live_process_frame, current_exercise_type, current_view_type
    
    try:
        # Check if camera is already stopped or doesn't exist
        if camera is None:
            return jsonify({
                'status': 'already_stopped',
                'message': 'Camera is already stopped or was never started'
            })
        
        # Attempt to release camera resources
        try:
            camera.release()
            print("Camera resources released successfully")
        except Exception as e:
            # Log the error but continue with cleanup
            print(f"Warning: Error releasing camera resources: {e}")
        
        # Reset camera to None regardless of release success/failure
        camera = None
        
        # Reset processor state when camera is stopped
        # This ensures clean state for next camera session
        live_process_frame = None
        
        # Optionally reset other state variables to defaults
        # This can help prevent state inconsistencies
        current_exercise_type = 'squat'  # Reset to default
        current_view_type = 'side'  # Reset to default
        
        return jsonify({
            'status': 'stopped',
            'message': 'Camera stopped successfully and processor state reset'
        })
        
    except Exception as e:
        # Handle any unexpected errors during stop operation
        # Ensure camera is set to None even if errors occur
        camera = None
        live_process_frame = None
        
        return jsonify({
            'status': 'error',
            'message': f'Error stopping camera: {str(e)}. Camera state has been reset.'
        }), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    mode = request.form.get('mode', 'Beginner')
    exercise_type = request.form.get('exerciseType', 'squat')
    
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process video
        output_filename = f"processed_{exercise_type}_{filename}"
        # Keep .mp4 extension for better browser compatibility
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Get thresholds based on mode and view type
        view_type = request.form.get('viewType', 'side')
        
        # Use the same processor factory as live processing for consistency
        process_frame = create_processor(mode, view_type, exercise_type, flip_frame=False)
        
        # Process video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return jsonify({'status': 'error', 'message': 'Failed to open uploaded video'}), 400

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        frame_size = (width, height)

        # Downscale very large frames to improve processing speed & avoid stall
        MAX_WIDTH = 960
        if width > MAX_WIDTH:
            scale = MAX_WIDTH / width
            target_size = (int(width * scale), int(height * scale))
            print(f"[upload] Downscaling video from {width}x{height} to {target_size[0]}x{target_size[1]} for processing speed")
            frame_size = target_size
        else:
            target_size = frame_size

        # Robust codec fallback sequence (AV1 removed due to unreliable support in current OpenCV build)
        codec_candidates = [
            ('avc1', 'H.264 avc1'),           # Preferred (browser friendly) – may fail if OpenH264 not present
            ('H264', 'H.264 H264'),           # Alternate tag
            ('X264', 'H.264 X264'),           # Another possible fourcc
            ('mp4v', 'MP4V (MPEG-4 Part 2)'), # Fallback if H.264 not available
            ('MJPG', 'MJPG fallback')         # Last resort
        ]

        video_output = None
        chosen_codec = None
        for fourcc_tag, label in codec_candidates:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
            vo = cv2.VideoWriter(output_path, fourcc, fps, target_size)
            if vo.isOpened():
                video_output = vo
                chosen_codec = label
                print(f"[upload] Using codec: {label} ({fourcc_tag}) -> {output_filename}")
                break
            else:
                vo.release()
                print(f"[upload] Codec failed: {label} ({fourcc_tag})")

        if video_output is None:
            return jsonify({'status': 'error', 'message': 'Failed to initialize any video writer codec'}), 500

        # Initialize counters and state for the video processing
        frame_count = 0
        processed_frames = 0
        last_log_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Resize if we downscaled
                if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

                # Convert frame from BGR to RGB before processing it
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process the frame based on the view type
                try:
                    out_frame, _ = process_frame.process(frame_rgb, pose)
                except Exception as e:
                    # Log per-frame error but continue
                    if frame_count <= 5:
                        print(f"[upload] Frame {frame_count} processing error (continuing): {e}")
                    out_frame = frame_rgb  # fallback to original frame

                processed_frames += 1

                # Convert back to BGR for writing to video
                out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
                video_output.write(out_frame_bgr)

                # Periodic progress log (every ~1s)
                if time.time() - last_log_time > 1.0:
                    last_log_time = time.time()
                    print(f"[upload] Progress: {processed_frames} frames written (codec={chosen_codec})")
        finally:
            cap.release()
            video_output.release()
            print(f"[upload] Completed processing {processed_frames} frames (total read: {frame_count}). Output: {output_filename} codec={chosen_codec}")
        
        return jsonify({
            'status': 'success',
            'original': f"/uploads/{filename}",
            'processed': f"/outputs/{output_filename}",
            'exerciseType': exercise_type,
            'viewType': view_type
        })
    
    return jsonify({'status': 'error', 'message': 'Failed to process file'})

@app.after_request
def add_header(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    # Ensure proper MIME type for video files
    if filename.lower().endswith('.mp4'):
        response.headers['Content-Type'] = 'video/mp4'
    return response

@app.route('/outputs/<filename>')
def output_file(filename):
    response = send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    # Ensure proper MIME type for video files
    if filename.lower().endswith('.mp4'):
        response.headers['Content-Type'] = 'video/mp4'
    return response

# Add static routes for direct file access
@app.route('/static/outputs/<filename>')
def static_output_file(filename):
    response = send_from_directory(app.config['OUTPUT_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    # Ensure proper MIME type for video files
    if filename.lower().endswith('.mp4'):
        response.headers['Content-Type'] = 'video/mp4'
    return response

@app.route('/static/uploads/<filename>')
def static_upload_file(filename):
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    # Ensure proper MIME type for video files
    if filename.lower().endswith('.mp4'):
        response.headers['Content-Type'] = 'video/mp4'
    return response

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
