import cv2
import os
import numpy as np
import streamlit as st
from PIL import Image
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
    page_title="Smart Absensi",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #3a506b;
            color: #333;
        }
        .sidebar .sidebar-content {
            background-color: #262730;
            color: white;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            height: 40px;
            width: 100%;
            font-size: 18px;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .stTextInput > div > input {
            background-color: #f0f0f0;
            border-radius: 5px;
            height: 40px;
            font-size: 18px;
        }
        .stSidebar .stTextInput > div > input {
            background-color: #333;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Function to record face data
def rekamDataWajah(nama_lengkap, nidn, jabatan):
    wajahDir = 'datawajah'
    if not os.path.exists(wajahDir):
        os.makedirs(wajahDir)

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    face_cascade_path = "histogram_frontalface_default.xml"
    eye_cascade_path = "histogram_eye.xml"
    faceDetector = cv2.CascadeClassifier(face_cascade_path)
    eyeDetector = cv2.CascadeClassifier(eye_cascade_path)
    ambilData = 1

    stframe = st.empty()

    while True:
        retV, frame = cam.read()
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.3, 5)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            namaFile = f'{nidn}_{nama_lengkap}_{jabatan}_{ambilData}.jpg'
            cv2.imwrite(os.path.join(wajahDir, namaFile), frame)
            ambilData += 1
            roiabuabu = abuabu[y:y + h, x:x + w]
            roiwarna = frame[y:y + h, x:x + w]
            eyes = eyeDetector.detectMultiScale(roiabuabu)
            for (xe, ye, we, he) in eyes:
                cv2.rectangle(roiwarna, (xe, ye), (xe + we, ye + he), (0, 255, 255), 1)

        stframe.image(frame, channels="BGR", caption="Rekaman Data Wajah")
        if ambilData > 30:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    st.success("Rekam Data Telah Selesai!")

# Function to train face data
def trainingWajah():
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'
    if not os.path.exists(latihDir):
        os.makedirs(latihDir)

    def getImageLabel(path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        faceSamples = []
        faceIDs = []
        for imagePath in imagePaths:
            try:
                PILimg = Image.open(imagePath).convert('L')
                imgNum = np.array(PILimg, 'uint8')
                filename = os.path.split(imagePath)[-1]
                faceID = int(filename.split('_')[0])
                faces = faceDetector.detectMultiScale(imgNum)
                for (x, y, w, h) in faces:
                    faceSamples.append(imgNum[y:y + h, x:x + w])
                    faceIDs.append(faceID)
            except Exception as e:
                st.error(f"Error processing file {imagePath}: {e}")
        return faceSamples, faceIDs

    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceDetector = cv2.CascadeClassifier('histogram_frontalface_default.xml')
    faces, IDs = getImageLabel(wajahDir)

    if len(faces) > 1 and len(IDs) > 1:
        faceRecognizer.train(faces, np.array(IDs))
        faceRecognizer.write(os.path.join(latihDir, 'training.xml'))
        st.success("Training Wajah Telah Selesai!")
    else:
        st.error("Not enough data to train the model.")

# Function to mark attendance
def markAttendance(nama_lengkap, nidn, jabatan):
    with open("Kehadiran.csv", 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        namelist = [line.split(',')[0] for line in lines]

        if nama_lengkap not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{nama_lengkap},{jabatan},{nidn},{dtString}')

# Function to recognize faces and mark attendance
def absensiWajah(nama_lengkap, nidn, jabatan):
    latihDir = 'latihwajah'
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('histogram_frontalface_default.xml')
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(os.path.join(latihDir, 'training.xml'))
    font = cv2.FONT_HERSHEY_SIMPLEX

    stframe = st.empty()

    while True:
        retV, frame = cam.read()
        frame = cv2.flip(frame, 1)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.2, 5,
                                              minSize=(round(0.1 * cam.get(3)), round(0.1 * cam.get(4))))
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = faceRecognizer.predict(abuabu[y:y + h, x:x + w])
            if confidence < 100:
                id = nama_lengkap
                confidence = f"  {round(100 - confidence)}%"
                markAttendance(nama_lengkap, nidn, jabatan)  # Ensure attendance is marked for recognized faces
            else:
                id = "Tidak Diketahui"
                confidence = f"  {round(100 - confidence)}%"

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h + 25), font, 1, (255, 255, 0), 2)

        stframe.image(frame, channels="BGR", caption="Absensi Wajah")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    st.success("Absensi Telah Dilakukan!")
    cam.release()
    cv2.destroyAllWindows()

# Streamlit app layout
st.title("📸 Smart Absensi - Face Attendance")
st.write("### Aplikasi absensi berbasis pengenalan wajah")

st.sidebar.title("User Information")
nama_lengkap = st.sidebar.text_input("Nama Lengkap")
nidn = st.sidebar.text_input("NIDN")
jabatan = st.sidebar.text_input("Jabatan")

st.sidebar.write("---")

if st.sidebar.button('Take Images'):
    if nama_lengkap and nidn and jabatan:
        rekamDataWajah(nama_lengkap, nidn, jabatan)
    else:
        st.sidebar.error("Please fill out all fields!")

if st.sidebar.button('Training'):
    trainingWajah()

if st.sidebar.button('Automatic Attendance'):
    if nama_lengkap and nidn and jabatan:
        st.sidebar.info("Mengambil gambar, tekan 'q' untuk berhenti...")
        st.sidebar.info("Melakukan absensi otomatis...")
        absensiWajah(nama_lengkap, nidn, jabatan)
    else:
        st.sidebar.error("Please fill out all fields!")
