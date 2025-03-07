from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time  # To wait for file generation
from celery.result import AsyncResult
from celery_worker import celery, worker  # Import Celery app

# Enable expandable segments for CUDA memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Initialize Flask app
app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',  # Redis broker URL
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',  # Redis result backend
)

# Enable CORS for specific origins
#CORS(app, origins=['https://aleksanderekman.github.io', "https://bakkadiffusion.vercel.app"])

# Route to generate an image based on the prompt
@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate prompt input
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']
        num_steps = data.get('num_inference_steps', 4)
        guidance_scale = data.get('guidance_scale', 0.0)
        max_seq_length = data.get('max_sequence_length', 512)
        userHeight = data.get('height', 1024)
        userWidth = data.get('width', 1024)

        # Start Celery task
        task = worker.delay(prompt, num_steps, guidance_scale,max_seq_length, userHeight, userWidth)

        print('message: ', task.id, task.status)

        return jsonify({"message": "Task started", "task_id": task.id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    try:
        task = AsyncResult(task_id, app=celery)

        print(f"Task state: {task.state}")

        if task.state == 'PENDING':
            return jsonify({"message": "Processing, please check later"}), 202

        elif task.state == 'SUCCESS':
            result = task.result
            print(f"Task result: {result}")

            # Check if the task was successful and completed
            if result["status"] == "completed":
                file_path = result["file_path"]
                # Ensure the file exists before returning it (retry for a few seconds)
                for _ in range(5):
                    if os.path.exists(file_path):
                        return send_file(file_path, mimetype='image/png')

                    # Retry after 1 second if file doesn't exist yet
                    time.sleep(1)

                # If file is not found after retries, return an error
                return jsonify({"error": "File not found, try again later"}), 404

            else:
                # If there was an error during image generation
                return jsonify({"error": result["error"]}), 500

        else:
            # If the task is in any other state
            return jsonify({"status": task.state}), 200

    except Exception as e:
        print(f"Error in get_result: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
