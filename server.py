import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))

import torch.multiprocessing as mp
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from gunicorn.app.base import BaseApplication
from worker import worker_process

mp.set_start_method('spawn', force=True)

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

num_workers = 2
task_queue = mp.Queue()
result_queue = mp.Queue()
workers = []

def start_workers():
    global workers
    for _ in range(num_workers):
        p = mp.Process(target=worker_process, args=(task_queue, result_queue))
        p.start()
        workers.append(p)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()

        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

        prompt = data['prompt']
        num_steps = data.get('num_inference_steps', 4)
        guidance_scale = data.get('guidance_scale', 0.0)
        max_seq_length = data.get('max_sequence_length', 512)
        userHeight = data.get('height', 1024)
        userWidth = data.get('width', 1024)

        app.logger.info(f"Generating image for prompt: {prompt}")
        app.logger.info(f"{userWidth}x{userHeight}")

        task_queue.put((prompt, num_steps, guidance_scale, max_seq_length, userHeight, userWidth))
        output_path = result_queue.get()

        return send_file(output_path, mimetype='image/png')

    except Exception as e:
        app.logger.error(f"Error generating image: {str(e)}")
        return jsonify({"error": str(e)}), 500

class StandaloneApplication(BaseApplication):
    def __init__(self, app, options=None):
        self.options = options or {}
        self.application = app
        super().__init__()

    def load_config(self):
        config = {key: value for key, value in self.options.items()
                  if key in self.cfg.settings and value is not None}
        for key, value in config.items():
            self.cfg.set(key.lower(), value)

    def load(self):
        return self.application

if __name__ == '__main__':
    start_workers()
    
    options = {
        'bind': '0.0.0.0:5000',
        'workers': 1,
        'timeout': 600,
        'worker_class': 'gevent'
    }
    StandaloneApplication(app, options).run()

    # Cleanup
    for _ in range(num_workers):
        task_queue.put(None)
    for p in workers:
        p.join()
