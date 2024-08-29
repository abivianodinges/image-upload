from flask import Flask, render_template, request
from omit import predict_image, predict_violence

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monkey', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    if file:
        return predict_image(file)


@app.route('/violence', methods=['POST'])
def upload_file_test():
    file = request.files['file']
    
    if file:
        return predict_violence(file)

@app.route('/upload', methods=['POST'])
def upload_file_violence():
    file = request.files['file']
    
    if file:
        tag_list = [predict_image(file), predict_violence(file)]
        return ''.join(f'<a>{tag}</a>' for tag in tag_list)

if __name__ == '__main__':
    app.run(debug=True)