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
    os.makedirs(wajahDir, exist_ok=True)

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
        if not retV:
            st.error("Gagal membaca frame dari kamera.")
            break

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
    os.makedirs(latihDir, exist_ok=True)

    def getImageLabel(path):
        faceDetector = cv2.CascadeClassifier('histogram_frontalface_default.xml')
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        faceIDs = []
        for imagePath in imagePaths:
            try:
                PILimg = Image.open(imagePath).convert('L')
                imgNum = np.array(PILimg, 'uint8')
                filename = os.path.basename(imagePath)
                faceID = int(filename.split('_')[0])
                faces = faceDetector.detectMultiScale(imgNum)
                for (x, y, w, h) in faces:
                    faceSamples.append(imgNum[y:y + h, x:x + w])
                    faceIDs.append(faceID)
            except Exception as e:
                st.error(f"Error processing file {imagePath}: {e}")
        return faceSamples, faceIDs

    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, IDs = getImageLabel(wajahDir)

    if len(faces) > 1 and len(IDs) > 1:
        faceRecognizer.train(faces, np.array(IDs))
        faceRecognizer.write(os.path.join('latihwajah', 'training.xml'))
        st.success("Training Wajah Telah Selesai!")
    else:
        st.error("Data tidak cukup untuk melatih model.")

# Function to mark attendance
def markAttendance(nama_lengkap, nidn, jabatan):
    file_path = "Kehadiran.csv"
    now = datetime.now()
    dtString = now.strftime('%H:%M:%S')
    tanggalString = now.strftime('%d %B %Y')

    # Check if file exists and create it if not
    if not os.path.isfile(file_path):
        with open(file_path, 'w') as f:
            f.write(f"Presensi\nTanggal : {tanggalString},,,\n")
            f.write("Nama Lengkap, Jabatan, NIDN, Waktu Kedatangan\n")

    with open(file_path, 'a') as f:
        f.write(f"{nama_lengkap}, {jabatan}, {nidn}, {dtString}\n")

# Function to recognize faces and mark attendance
def absensiWajah(nama_lengkap, nidn, jabatan):
    wajahDir = 'datawajah'
    latihDir = 'latihwajah'

    if not os.path.isfile(os.path.join(latihDir, 'training.xml')):
        st.error("File model pelatihan tidak ditemukan.")
        return

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    faceDetector = cv2.CascadeClassifier('histogram_frontalface_default.xml')
    faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
    faceRecognizer.read(os.path.join(latihDir, 'training.xml'))
    font = cv2.FONT_HERSHEY_SIMPLEX

    stframe = st.empty()
    minWidth = 0.1 * cam.get(3)
    minHeight = 0.1 * cam.get(4)

    if 'stop_absensi' not in st.session_state:
        st.session_state.stop_absensi = False

    st.button('Berhenti Absensi', key='stop_button', on_click=lambda: setattr(st.session_state, 'stop_absensi', True))

    while True:
        if st.session_state.stop_absensi:
            st.session_state.stop_absensi = False
            st.success("Absensi Dihentikan!")
            break

        retV, frame = cam.read()
        if not retV:
            st.error("Gagal membaca frame dari kamera.")
            break

        frame = cv2.flip(frame, 1)
        abuabu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetector.detectMultiScale(abuabu, 1.2, 5, minSize=(round(minWidth), round(minHeight)))

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = faceRecognizer.predict(abuabu[y:y + h, x:x + w])

            if confidence < 100:
                id = nama_lengkap
                confidence_display = f"{round(150 - confidence)}%"
            elif confidence < 50:
                id = nama_lengkap
                confidence_display = f"{round(170 - confidence)}%"
            else:
                id = "Tidak Diketahui"
                confidence_display = f"{round(150 - confidence)}%"

            cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence_display), (x + 5, y + h + 25), font, 1, (255, 255, 0), 2)

        stframe.image(frame, channels="BGR", caption="Absensi Wajah")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Mark attendance with the provided name, nidn, and jabatan
    markAttendance(nama_lengkap, nidn, jabatan)

    cam.release()
    cv2.destroyAllWindows()

def main():
    st.title("📸 Smart Absensi - Face Attendance")
    st.write("### Aplikasi absensi berbasis pengenalan wajah")

    st.sidebar.title("Informasi Pengguna")
    nama_lengkap = st.sidebar.text_input("Nama Lengkap")
    nidn = st.sidebar.text_input("NIDN")
    jabatan = st.sidebar.selectbox("Jabatan", ["Admin 1", "Admin 2", "Admin 3", "Admin 4", "Admin 5"])

    st.sidebar.write("---")

    if st.sidebar.button('Rekam Data Wajah'):
        rekamDataWajah(nama_lengkap, nidn, jabatan)

    if st.sidebar.button('Latih Data Wajah'):
        trainingWajah()

    if st.sidebar.button('Mulai Absensi'):
        st.write(f"Nama: {nama_lengkap}, NIDN: {nidn}, Jabatan: {jabatan}")  # Debugging line
        absensiWajah(nama_lengkap, nidn, jabatan)

if __name__ == "__main__":
    main()
