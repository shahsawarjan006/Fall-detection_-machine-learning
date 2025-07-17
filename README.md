# AI-RDP: Fall Detection System![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.1-green.svg)](https://mediapipe.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
-Usage](#usage)
- [Project Structure](#project-structure)
- Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

AI-RDP (AI-based Real-time Detection and Prediction) is an advanced fall detection system that uses computer vision and machine learning to detect falls from video footage in real-time. The system employs multiple pose estimation models and three different machine learning algorithms to achieve high accuracy in fall detection.

### Key Capabilities

- **Real-time fall detection** from video streams
- **Multi-model approach** using SVM, MLP, and LSTM
- **Dual pose estimation** with MediaPipe and MoveNet Lightning
- **Comprehensive analysis** with detailed CSV reports
- **Annotated video output** with pose landmarks and fall predictions

## ✨ Features

### 🤖 Machine Learning Models
- **SVM (Support Vector Machine)**: Fast binary classification
- **MLP (Multi-Layer Perceptron)**: Neural network for complex patterns
- **LSTM (Long Short-Term Memory)**: Sequential data analysis

### 📹 Pose Estimation
- **MediaPipe**: Googles pose estimation framework
- **MoveNet Lightning**: TensorFlow's efficient pose detection

### 📊 Analysis & Output
- **Frame-by-frame analysis** with timestamps
- **Processing time metrics** for performance evaluation
- **Annotated videos** with pose landmarks
- **Detailed CSV reports** with predictions and metadata
- **Fall detection timing** with first detection timestamps

## 🏗️ Architecture

```
Input Video → Pose Estimation → Feature Extraction → ML Models → Fall Detection
     ↓              ↓                    ↓              ↓           ↓
  Video File   MediaPipe/MoveNet   Keypoint Data   SVM/MLP/LSTM   Results
```

### Processing Pipeline1. **Video Input**: Accepts video files for processing
2. **Pose Detection**: Extracts 17 body points using MediaPipe or MoveNet
3. **Feature Engineering**: Processes keypoint coordinates for ML models
4. **Model Prediction**: Three models analyze the pose data
5. **Result Generation**: Creates annotated videos and detailed reports

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Google Colab (recommended) or Jupyter Notebook
- GPU support (optional, for faster processing)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-RDP.git
   cd AI-RDP
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **For Google Colab users**
   - Upload the project to Google Drive
   - Open `DEMO/PrototypesDemo.ipynb` in Google Colab
   - Mount your Google Drive

### Required Packages

```bash
pip install tensorflow mediapipe opencv-python pandas numpy matplotlib
pip install scikit-learn joblib tensorflow-hub
```

## 📖 Usage

### Quick Start

1Open the main demo file**
   ```python
   # Navigate to DEMO folder
   # Open PrototypesDemo.ipynb in Google Colab
   ```

2. **Set your video path**
   ```python
   VIDPATH = '/path/to/your/video.mp4
   vid_num = 1  # Video identifier
   TEST = True   # Enable processing
   ```

3. **Run fall detection**
   ```python
   # For SVM model
   createDemoVideo(vid_num, VIDPATH, TEST, mv_svm_model, mp_svm_model, 'svm)
   
   # For MLP model
   createDemoVideo(vid_num, VIDPATH, TEST, mv_mlp_model, mp_mlp_model, 'mlp')
   
   # For LSTM model
   createDemoVideo(vid_num, VIDPATH, TEST, mv_lstm_model, mp_lstm_model, 'lstm')
   ```

### Advanced Usage

#### Processing Multiple Videos
```python
# Process validation set
for i in range(1, 31):
    vid_num = i
    VIDPATH = f'/path/to/validation/fall{vid_num}.mp4'
    createDemoVideo(vid_num, VIDPATH, TEST, mv_svm_model, mp_svm_model, 'svm')
```

#### Custom Model Selection
```python
# Run specific models only
createDemoVideo(vid_num, VIDPATH, TEST, mv_svm_model, mp_svm_model, 'svm')    # SVM only
createDemoVideo(vid_num, VIDPATH, TEST, mv_mlp_model, mp_mlp_model, 'mlp')    # MLP only
createDemoVideo(vid_num, VIDPATH, TEST, mv_lstm_model, mp_lstm_model, 'lstm') # LSTM only
```

## 📁 Project Structure

```
AI-RDP/
├── DEMO/                          # Main demo files
│   ├── PrototypesDemo.ipynb      # Main execution file
│   ├── Models_Training.ipynb     # Model training scripts
│   ├── Dataset_of_FEATURES_MoveNet&MediaPipe.ipynb
│   ├── Charts&Graphs.ipynb       # Visualization scripts
│   ├── READ ME.txt               # Project documentation
│   ├── demo_video.mp4            # Sample video
│   └── *.h5, *.keras, *.joblib  # Trained model files
├── PREDICTED_VIDEOS/             # Output videos
├── VALIDATION_SET/               # Test dataset
├── PRESENTATION & DOCs/          # Documentation
└── TestDatasetOut/               # Test results
```

### Key Files

- **`PrototypesDemo.ipynb`**: Main execution file with all functionality
- **`Models_Training.ipynb`**: Training scripts for ML models
- **`Dataset_of_FEATURES_MoveNet&MediaPipe.ipynb`**: Data preprocessing
- **`Charts&Graphs.ipynb`**: Results visualization

## 🤖 Models

### Trained Models Included

| Model Type | MediaPipe Model | MoveNet Model | File Format |
|------------|----------------|---------------|-------------|
| SVM        | `svm_MediaPipe_model.joblib` | `svm_lightning_model.joblib` | .joblib |
| MLP        | `MediaPipe_NN_model50pochsv2.keras` | `MoveNet_NN_model50pochsv2.keras` | .keras |
| LSTM       | `lstm_fall_detection_mpv2h5` | `lstm_fall_detection_mvv2| .h5 |

### Model Performance

- **SVM**: Fast inference, binary classification (0/1)
- **MLP**: Neural network with probability scores (0-1)
- **LSTM**: Sequential analysis with temporal patterns

## 📊 Results

### Output Files

For each processed video, the system generates:

#### Videos (6 files)
- `MediaPipe_SVM_detected_video_X.mp4`
- `MediaPipe_MLP_detected_video_X.mp4`
- `MediaPipe_LSTM_detected_video_X.mp4 `MoveNetLightning_SVM_detected_video_X.mp4 `MoveNetLightning_MLP_detected_video_X.mp4 `MoveNetLightning_LSTM_detected_video_X.mp4`

#### CSV Reports (6 files)
- Detailed frame-by-frame analysis
- Columns: `video_num`, `frame_idx`, `predictions`, `frame_processing_duration`, `fall_detection_duration_perVideo`, `frame_width`, `frame_height`

### Sample Output Structure

```
Frame Analysis:
├── Video Processing Time: X ms
├── Fall Detection Time: Y ms
├── Frame Dimensions: WxH
└── Prediction Confidence: 01`

## 🎯 Use Cases

### Healthcare
- **Elderly care facilities**: Monitor residents for falls
- **Hospitals**: Patient safety monitoring
- **Home care**: Remote fall detection

### Security & Safety
- **Public spaces**: Accident detection
- **Workplace safety**: Industrial accident prevention
- **Sports facilities**: Athletic injury detection

### Research
- **Computer vision research**: Pose estimation studies
- **ML research**: Multi-model comparison
- **Healthcare research**: Fall prevention studies

## 🔧 Configuration

### Model Parameters

```python
# SVM Configuration
svm_model = SVC(kernel='linear,random_state=42)

# MLP Configuration
mlp_model = Sequential([
    Dense(64, activation=relu'),
    Dense(32, activation=relu'),
    Dense(1, activation=sigmoid')
])

# LSTM Configuration
lstm_model = Sequential([
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1, activation='sigmoid')
])
```

### Pose Estimation Settings

```python
# MediaPipe Settings
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    running_mode=vision.RunningMode.VIDEO
)

# MoveNet Settings
input_size = 192  # Lightning model
input_size = 256  # Thunder model
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**4ests if applicable**
5. **Commit your changes**
   ```bash
   git commit -m Add: your feature description  ```
6. **Push to the branch**
   ```bash
   git push origin feature/your-feature-name
   ```
7ate a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Update documentation for new features
- Test with sample videos before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google MediaPipe** for pose estimation framework
- **TensorFlow** for MoveNet implementation
- **Scikit-learn** for SVM implementation
- **OpenCV** for video processing
- **Pandas & NumPy** for data manipulation

## 📞 Contact

For questions, issues, or contributions:

 
- **Project**: [AI-RDP Repository](https://github.com/yourusername/AI-RDP)

---

**Note**: This project is designed for research and educational purposes. For production use in healthcare or safety applications, additional validation and testing is recommended.

**Made with ❤️ by the Fantastic Four Team**
