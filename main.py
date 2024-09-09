import os
import shutil  # For copying files
import sys
import argparse
import numpy as np
import cv2 as cv
from tqdm import tqdm  # For progress bar
from sface import SFace
from yunet import YuNet

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(
    description="Face recognition on a folder of images"
)
parser.add_argument('--target', '-t', type=str, required=True,
                    help='Path to the target face image.')
parser.add_argument('--query_dir', '-q', type=str, required=True,
                    help='Directory path for the query images.')
parser.add_argument('--model', '-m', type=str, default='face_recognition_sface_2021dec.onnx',
                    help='Path to the face recognition model.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--dis_type', type=int, choices=[0, 1], default=0,
                    help='Distance type: 0 for cosine (default), 1 for norm_l1.')
parser.add_argument('--save', '-s', type=str, default=None,
                    help='Directory path to save images with recognized faces. If not set, no images will be copied.')
parser.add_argument('--threshold', '-th', type=float, default=0.6,
                    help='Accuracy threshold for saving recognized faces (default 0.6).')
parser.add_argument('--logfile', '-l', type=str, default=None,
                    help='Path to the log file to record recognized faces. Defaults to no log file if not set.')
args = parser.parse_args()

SUPPORTED_EXTENSIONS = ['bmp', 'pbm', 'pgm', 'ppm', 'sr', 'ras', 'jpg', 'jpeg', 'png', 'tiff', 'tif', 'webp', 'jp2', 'exr']

def process_image(query_img_path, img1, faces1, recognizer, detector, logfile=None, save_dir=None):
    img2 = cv.imread(query_img_path)

    # Check if image is valid
    if img2 is None:
        return None  # Skip this image

    # Detect faces in query image
    detector.setInputSize([img2.shape[1], img2.shape[0]])
    faces2 = detector.infer(img2)
    if faces2.shape[0] == 0:
        return None  # No faces found

    # Match faces between target and query image
    scores = []
    matches = []
    for face in faces2:
        result = recognizer.match(img1, faces1[0][:-1], img2, face[:-1])
        scores.append(result[0])
        matches.append(result[1])

    # Check if any match exceeds the threshold
    if any(score > args.threshold for score in scores):
        message = f"Face recognized in {os.path.basename(query_img_path)} with score: {max(scores)}"
        if logfile:
            logfile.write(f"{message}\n")
        
        # Save the image if save_dir is specified
        if save_dir:
            save_path = os.path.join(save_dir, os.path.basename(query_img_path))
            shutil.copy(query_img_path, save_path)
        
        return os.path.basename(query_img_path)
    
    return None  # No recognized face

def recognize_faces_in_folder():
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Instantiate SFace for face recognition
    recognizer = SFace(modelPath=args.model,
                       disType=args.dis_type,
                       backendId=backend_id,
                       targetId=target_id)
    
    # Instantiate YuNet for face detection
    detector = YuNet(modelPath='../face_detection_yunet/face_detection_yunet_2023mar.onnx',
                     inputSize=[320, 320],
                     confThreshold=0.9,
                     nmsThreshold=0.3,
                     topK=5000,
                     backendId=backend_id,
                     targetId=target_id)

    # Load the target image
    img1 = cv.imread(args.target)
    detector.setInputSize([img1.shape[1], img1.shape[0]])
    faces1 = detector.infer(img1)
    assert faces1.shape[0] > 0, 'Cannot find a face in the target image'

    logfile = None
    if args.logfile:
        logfile = open(args.logfile, "a")  # Append mode

    save_dir = args.save
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Create the directory if it does not exist

    matching_images = []

    # Get the list of query images, filtering by supported formats
    query_images = [os.path.join(args.query_dir, img_name) for img_name in os.listdir(args.query_dir) 
                    if img_name.split('.')[-1].lower() in SUPPORTED_EXTENSIONS]
    
    # Sequential processing with tqdm for progress bar
    for img_path in tqdm(query_images, desc="Processing images"):
        result = process_image(img_path, img1, faces1, recognizer, detector, logfile, save_dir)
        if result:
            matching_images.append(result)  # Collect matching images

    if logfile:
        logfile.close()

    # Print matching images to terminal
    if matching_images:
        print("Recognized faces in the following images:")
        for img_name in matching_images:
            print(img_name)

if __name__ == '__main__':
    recognize_faces_in_folder()
