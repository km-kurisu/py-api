from flask import Flask, request, jsonify
import requests
import pytesseract
import cv2
from deepface import DeepFace
import os
import tempfile

app = Flask(__name__)

# Set tesseract path if needed (comment out if not required on server)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def download_file(url):
    local_fd, local_path = tempfile.mkstemp()
    os.close(local_fd)
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)
    return local_path

def extract_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return image[y:y+h, x:x+w]
    return None

def verify_document(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        doc_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(doc_contour)
        doc_img = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(doc_img)
        doc_face = extract_face(doc_img)
        doc_face_path = None
        if doc_face is not None:
            doc_face_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(doc_face_path, doc_face)
        return text, doc_face_path
    return '', None

def extract_face_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_face_path = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_face_path = tempfile.mktemp(suffix='.jpg')
        cv2.imwrite(video_face_path, frame)
        break
    cap.release()
    return video_face_path

def verify_full_name(extracted_text, provided_name):
    extracted_name = ' '.join(extracted_text.lower().split())
    provided_name = ' '.join(provided_name.lower().split())
    return provided_name in extracted_name

@app.route('/api/verify', methods=['POST'])
def api_verify():
    data = request.json
    document_url = data.get('document_url')
    video_url = data.get('video_url')
    full_name = data.get('full_name')
    if not document_url or not video_url or not full_name:
        return jsonify({'error': 'Missing required fields'}), 400
    # Download files
    doc_path = download_file(document_url)
    vid_path = download_file(video_url)
    # Document verification
    extracted_text, doc_face_path = verify_document(doc_path)
    # Video face extraction
    video_face_path = extract_face_from_video(vid_path)
    # Name match
    name_match = verify_full_name(extracted_text, full_name)
    # Face match
    face_match = False
    similarity = 0
    if doc_face_path and video_face_path:
        try:
            result = DeepFace.verify(img1_path=doc_face_path, img2_path=video_face_path, model_name="Facenet")
            distance = result.get("distance", 1)
            similarity = 100 - distance * 100
            face_match = similarity >= 10 and name_match
        except Exception as e:
            return jsonify({'error': f'Face matching error: {str(e)}'}), 500
    # Clean up temp files
    for p in [doc_path, vid_path, doc_face_path, video_face_path]:
        if p and os.path.exists(p):
            os.remove(p)
    return jsonify({
        'name_match': name_match,
        'similarity': similarity,
        'face_match': face_match
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
