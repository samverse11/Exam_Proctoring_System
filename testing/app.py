import cv2
import base64
import numpy as np
import logging
from flask import Flask, render_template, request, jsonify, Response, session
from proctoring_logic import detect_face_and_behavior, detect_objects, verify_identity, camera_matrix, dist_coeffs
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up logging for standard logs (terminal)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Set up a separate logger for detection outputs
detection_logger = logging.getLogger("detection_logger")
detection_handler = logging.FileHandler("detection_output_log.txt")
detection_handler.setLevel(logging.INFO)
detection_formatter = logging.Formatter('%(asctime)s: %(message)s')
detection_handler.setFormatter(detection_formatter)
detection_logger.addHandler(detection_handler)

# Proctoring status
proctoring_active = True
camera = cv2.VideoCapture(0)
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame to JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format as part of the MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    # Return the response with the generated frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to render the HTML template
@app.route('/')
def home():
    return render_template('index.html')


# New route for starting proctoring
@app.route('/start_proctoring', methods=['POST'])
def start_proctoring():
    global proctoring_active
    try:
        # Get data from request
        data = request.get_json()  # Make sure to use get_json for JSON data
        user_id = data.get('user_id')
        course_id = data.get('course_id')
        
        if not user_id or not course_id:
            return jsonify({'error': 'Missing user or course ID'}), 400
        
        logger.info(f"Proctoring started for user {user_id} on course {course_id}")
        
        proctoring_active = True  # Activate the proctoring system
        return jsonify({'message': 'Proctoring started successfully'}), 200

    except Exception as e:
        logger.error(f"Error starting proctoring: {str(e)}")
        return jsonify({'error': 'Failed to start proctoring'}), 500


@app.route('/process_frame', methods=['POST'])
def process_frame():
    global proctoring_active  # Make sure to use the global variable
    alert_messages = []
    error_messages = []  # New list to collect error messages
    alert_count = session.get('alert_count', 0)  # Track the number of alerts using Flask session
    try:
        # Fetch incoming data
        data = request.json
        print(f"Received data")

        if 'frame' not in data:
            error_messages.append("No frame data received.")
            return jsonify({"alerts": alert_messages, "errors": error_messages}), 400

        frame_data = data['frame']
        if frame_data.startswith('data:image/jpeg;base64,'):
            frame_data = frame_data.split(',')[1]

        img_data = base64.b64decode(frame_data)
        np_arr = np.frombuffer(img_data, np.uint8)

        if np_arr.size == 0:
            error_messages.append("Empty buffer after decoding base64.")
            return jsonify({"alerts": alert_messages, "errors": error_messages}), 400

        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            error_messages.append("Image decoding failed.")
            return jsonify({"alerts": alert_messages, "errors": error_messages}), 400

        # Initialize frame count dictionary
        frame_count = {"away": 0}

        # Process frame for identity verification and detection
        try:
            verify_identity(frame, alert_messages)
            detect_face_and_behavior(frame, frame_count, camera_matrix, dist_coeffs, alert_messages)
            detect_objects(frame, alert_messages)

        except Exception as e:
            error_message = f"Error during frame processing: {str(e)}"
            error_messages.append(error_message)
            return jsonify({"alerts": alert_messages, "errors": error_messages}), 200


 # Add logic to track alert count
        if len(alert_messages) > 0:
            alert_count += 1
            session['alert_count'] = alert_count  # Store in session

        # If more than 3 alerts, signal the form to be submitted
        if alert_count > 3:
            return jsonify({"alerts": alert_messages, "errors": error_messages, "submit_form": True}), 200

        return jsonify({"alerts": alert_messages, "errors": error_messages, "submit_form": False}), 200

        # Check for suspicious behavior
        if frame_count["away"] > 30:  # Example threshold for gaze detection
            alert_message = "Suspicious Gaze Detected!"
            alert_messages.append(alert_message)

        # Send the response back to the frontend
        return jsonify({"alerts": alert_messages, "errors": error_messages}), 200
    

    except Exception as e:
        error_messages.append(f"Error: {str(e)}")
        return jsonify({"alerts": alert_messages, "errors": error_messages}), 500



if __name__ == '__main__':
    app.run(debug=True)
