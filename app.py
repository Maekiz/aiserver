from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from celery.result import AsyncResult
from celery_worker import worker

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379/0',  # Redis broker URL
    CELERY_RESULT_BACKEND='redis://localhost:6379/0',  # Redis result backend
)
CORS(app, origins=['https://aleksanderekman.github.io', "https://bakkadiffusion.vercel.app"])

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Get JSON data from request
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
        task = worker.apply_async(args=[prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth])

        return jsonify({"message": "Task started", "task_id": task.id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    try:
        # Use AsyncResult to check the status of the task
        task = AsyncResult(task_id, app=worker)

        if task.state == 'PENDING':
            return jsonify({"message": "Processing, please check later"}), 202
        elif task.state == 'SUCCESS':
            result = task.result
            if result["status"] == "completed":
                # Ensure the file exists before sending it
                file_path = result["file_path"]
                if os.path.exists(file_path):
                    return send_file(file_path, mimetype='image/png')
                else:
                    return jsonify({"error": "File not found"}), 404
            else:
                return jsonify({"error": result["error"]}), 500
        else:
            return jsonify({"status": task.state}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
