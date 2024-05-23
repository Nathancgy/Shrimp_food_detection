from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from process_video import process_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['video']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(input_path)

            return jsonify({"uploaded_video": file.filename}), 200
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)