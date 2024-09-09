# Face Recognition in a Directory

## Overview

This project provides a Python script to perform face recognition on all images within a specified directory. Using OpenCV, the YuNet face detection model, and pre-trained face recognition models, the script detects and recognizes faces in each image. Recognized faces are logged, and matching images can be copied to a specified directory.

For those moments when you have an entire folder of vacation photos and need to find out if your friend’s face appears in every single one of them—this tool’s got your back 😉.

## Features

- Detect and recognize faces in a directory of images using the YuNet face detection model.
- Logs recognized faces with accuracy scores to a text file.
- Copy images with recognized faces to a specified directory.
- Progress tracking with a visual progress bar using `tqdm`.

## Requirements
```bash
opencv-python
numpy
tqdm
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Andrei-Iosifescu123/batch-face-recognition.git
    cd batch-face-recognition
    ```

2. Install the required packages:
    ```bash
    pip3 install -r requirements.txt
    ```

## Usage
### Arguments
- `--target` or `-t`: Path to the target image containing the face to match against.
- `--query_dir` or `-q`: Directory containing images to search for faces.
- `--model` or `-m`: Path to the face recognition model (default: face_recognition_sface_2021dec.onnx).
- `--backend_target` or `-bt`: Choose the backend-target combination to run the demo. Options include:  
            0: OpenCV + CPU (default)  
            1: CUDA + GPU  
            2: CUDA + GPU (FP16)  
            3: TIM-VX + NPU  
            4: CANN + NPU  
- `--dis_type`: Distance type for face matching (0: cosine, 1: norm_l1; default: 0).
- `--save` or `-s`: Directory to copy images with recognized faces. If not provided, images won’t be copied.
- `--threshold` or `-th`: Accuracy threshold for recognizing faces (default: 0.6).
- `--logfile` or `-l`: Path to the log file to record recognized faces. If not provided, no log is created.

### Example
The following command will recognize all faces in the `/media/hdd/family_photos` directory matching the face in the image `Emily.jpg`, without saving matching images or outputting to a logfile:
```bash
python3 main.py --target Emily.jpg --query_dir /media/hdd/family_photos
```

## Contributing
Feel free to fork the repository and submit pull requests. If you find any issues or have suggestions for improvements, please open an issue.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/Andrei-Iosifescu123/batch-face-recognition/blob/main/LICENSE) file for details.
