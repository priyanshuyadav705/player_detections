# ⚽ Football Player & Ball Detection & Tracking

This project performs real-time detection and tracking of football players, the ball, referees, and goalkeepers using a YOLOv8 model and Deep SORT tracker. It maintains consistent IDs even when players leave and re-enter the frame (re-identification).

---

## 📌 Features

- Detects and tracks:
  - 🧍‍♂️ Player
  - 🧤 Goalkeeper
  - ⚽ Ball
  - 🧑‍⚖️ Referee
- Assigns **consistent IDs** across frames
- Re-identifies players who leave and re-enter the frame
- Exports a **video with bounding boxes** and ID labels

---

## 🧰 Requirements

- Python ≥ 3.8 (recommended: 3.10 or 3.11)

---

## 🐍 Setup Guide

### ✅ 1. Install Python

Get Python from the official site:  
➡️ https://www.python.org/downloads/

Make sure it's accessible from terminal:

```bash
python --version
pip --version

✅ 2. Create and Activate Virtual Environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# OR
.venv\Scripts\activate  # Windows

✅ 3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

Or manually:
pip install ultralytics deep_sort_realtime opencv-python numpy scikit-learn
⚠️ Don't use pip install sklearn — use scikit-learn.

🚀 Run 
python run.py
✅ Output

Bounding boxes for all detected classes
ID displayed like:
player ID:3
ball ID:7
referee ID:5
Output saved in /outputs/result.mp4
📄 requirements.txt
ultralytics
deep_sort_realtime
opencv-python
numpy
scikit-learn

⚙️ Optional Setup Script (setup.sh)
#!/bin/bash

echo "Creating virtual environment..."
python -m venv .venv
source .venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "All done. Run with: python run.py"
Make executable and run:

chmod +x setup.sh
./setup.sh
🧠 Model Class Index Reference

Class ID	Class Name
0	player
1	goalkeeper
2	ball
3	referee
Make sure these labels match your model's training configuration.

👤 Author
Priyanshu Yadav

📜 License

This project is licensed under the MIT License.