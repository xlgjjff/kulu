# kulu
my flask web site

## Native desktop assistant

The desktop UI uses Python's built-in Tkinter. It offers:

- Local command execution with the full command and working directory shown in a separate confirmation dialog before **each** run. Commands use the current user's normal permissions; they are not elevated or exposed through a web server. If the UI itself was started elevated, command execution is refused.
- Offline image OCR (Chinese and English) and COCO object detection. Images stay on this computer; no image or recognition result is uploaded by the app.

Install the requirements and start the desktop UI:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-desktop.txt
python desktop_app.py
```

Tkinter is included with most standard Windows Python installations. RapidOCR's OCR models (about 13 MB) ship with the Python package. On first object detection, the app downloads the official OpenCV Zoo YOLOX ONNX model (35,858,002 bytes, about 34.2 MiB) and verifies its SHA-256 before use. It is cached at `%LOCALAPPDATA%\Kulu\models\object_detection_yolox_2022nov.onnx`; after installation, both OCR and object detection run offline. If the model host is unavailable during setup, the UI shows an explicit error and OCR can still run independently. The app does not load model files selected by the user.
