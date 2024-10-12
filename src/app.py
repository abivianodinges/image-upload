from flask import Flask, render_template, request
from omit import predict_image, predict_dog, predict_dog_but_be_sure_about_it, classes_dog, classes
import os
import cv2
import glob

VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def extract_images_from_video(video_path, output_folder):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = 0

    while success:
        if frame_count % 5 == 0:
            # Save frame as image
            image_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
            cv2.imwrite(image_filename, image)

        success, image = vidcap.read()
        frame_count += 1

    vidcap.release()
    cv2.destroyAllWindows()  # Ensure all resources are released
    cv2.waitKey(1)  # To clear the buffer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monkey', methods=['POST'])
def upload_file():
    file = request.files['file']
    
    if file:
        return predict_image(file)


@app.route('/dog', methods=['POST'])
def upload_file_test():
    file = request.files['file']
    
    if file:
        return predict_dog(file) 

@app.route('/upload', methods=['POST'])
def upload_file_dog():
    file = request.files['file']
    
    if file:
        if allowed_file(file.filename, VIDEO_EXTENSIONS):
            os.makedirs('uploads',exist_ok=True)
            video_path = os.path.join('uploads', file.filename)
            # create a temp folder
            file.save(video_path)

            output_folder = os.path.join('uploads', 'frames')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Extract images from video
            extract_images_from_video(video_path, output_folder)

            images = glob.glob(os.path.join('uploads', 'frames', '*'))

            tag_list = []
            for image in images:
                tag_list +=[predict_image(image), predict_dog_but_be_sure_about_it(image)]
           
            tag_list = list(set(tag_list))
            classes_dog = ["dog", "notDog"]

            tag_list = [item for item in tag_list if item in classes + classes_dog ]
            # delete the temp folder
            os.remove(os.path.join('uploads', 'frames'))
            return ''.join(f'<a>{tag}</a>' for tag in tag_list)

        else:
            tag_list = [predict_image(file), predict_dog(file)]
            return ''.join(f'<a>{tag}</a>' for tag in tag_list)

if __name__ == '__main__':
    app.run(debug=True)