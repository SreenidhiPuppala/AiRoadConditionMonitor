# 🚧 AI Road Condition Monitor

An AI-powered road condition monitoring system that analyzes public vehicle dashcam footage to automatically detect road damage and assist in efficient road maintenance.

---

## 📌 Overview

Road infrastructure plays a crucial role in transportation safety. Manual road inspections are time-consuming, expensive, and often fail to detect defects promptly.

This project leverages **Computer Vision** and **Deep Learning (YOLOv8)** to automatically detect various types of road damages from dashcam images and videos, enabling faster and more efficient road maintenance.

---

## ✨ Features

- Detects multiple types of road damage
- Supports both image and video inputs
- Real-time object detection using YOLOv8
- User-friendly web interface
- Displays detected road damages with bounding boxes
- Can be extended for smart city and infrastructure monitoring

---

## 🛠️ Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Flask / Streamlit
- NumPy
- Pillow

---

## 📂 Project Structure

```
AIRoadConditionMonitor/
│── datasets/
│   ├── test
│   └── train
|   └── val
│
│── app.py
│── data.yaml
│── requirements.txt
│── .gitignore
│── README.md
```

---

## 🚀 Installation

### Clone the repository

```bash
git clone https://github.com/SreenidhiPuppala/AIRoadConditionMonitor.git
```

```bash
cd AIRoadConditionMonitor
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the virtual environment

**Windows**

```bash
venv\Scripts\activate
```

**macOS/Linux**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Flask

```bash
python app.py
```

### Streamlit (if applicable)

```bash
streamlit run app.py
```

---

## 📊 Model

- Model: **YOLOv8**
- Framework: **Ultralytics**
- Task: Road Damage Detection

The model is trained to identify different categories of road defects from dashcam images.

---

## 📁 Dataset

This project uses the **RDD2022 (Road Damage Detection)** dataset.

> **Note:**  
> The dataset is **not included** in this repository due to GitHub's file size limitations.

Download the dataset from the official Road Damage Detection (RDD2022) source and place it inside the `datasets/` directory before training.

---

## 📸 Sample Output

Outputs of this project are uploaded in outputImages folder


---

## 🔮 Future Enhancements

- Live dashcam video processing
- GPS-based road damage mapping
- Severity estimation
- Road maintenance prioritization
- Cloud deployment
- Mobile application support

---

## 👩‍💻 Author

**Sreenidhi Puppala**

GitHub: https://github.com/SreenidhiPuppala

---

## 📄 License

This project is developed for educational and research purposes.
