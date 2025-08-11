import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import json
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import math


class FaceLandmarkExtractor:
    """Extract facial landmarks using MediaPipe"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize MediaPipe Face Mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_landmarks(self, image):
        """Extract facial landmarks from image"""
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            # Get the first face landmarks
            landmarks = results.multi_face_landmarks[0]

            # Convert landmarks to numpy array
            landmark_points = []
            for landmark in landmarks.landmark:
                landmark_points.append([landmark.x, landmark.y, landmark.z])

            return np.array(landmark_points)

        return None

    def extract_landmarks_from_file(self, image_path):
        """Extract landmarks from image file"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        return self.extract_landmarks(image)

    def draw_landmarks(self, image, landmarks):
        """Draw facial landmarks on image"""
        if landmarks is None:
            return image

        # Convert landmarks to MediaPipe format
        mp_landmarks = self.mp_face_mesh.FaceLandmark()
        mp_landmarks.landmark = []

        for point in landmarks:
            landmark = mp_landmarks.landmark.add()
            landmark.x = point[0]
            landmark.y = point[1]
            landmark.z = point[2]

        # Draw landmarks
        annotated_image = image.copy()
        self.mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=mp_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        return annotated_image


class FaceMatcher:
    """Compare facial landmarks for face recognition"""

    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def calculate_distance(self, landmarks1, landmarks2):
        """Calculate Euclidean distance between two sets of landmarks"""
        if landmarks1 is None or landmarks2 is None:
            return float('inf')

        # Ensure same number of landmarks
        min_len = min(len(landmarks1), len(landmarks2))
        landmarks1 = landmarks1[:min_len]
        landmarks2 = landmarks2[:min_len]

        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((landmarks1 - landmarks2) ** 2, axis=1))
        return np.mean(distance)

    def match_faces(self, input_landmarks, known_landmarks_dict):
        """Match input landmarks against known faces"""
        if input_landmarks is None:
            return None, float('inf')

        best_match = None
        best_distance = float('inf')

        for name, landmarks_list in known_landmarks_dict.items():
            for landmarks in landmarks_list:
                distance = self.calculate_distance(input_landmarks, landmarks)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name

        return best_match, best_distance

    def is_same_person(self, distance):
        """Check if distance indicates same person"""
        return distance < self.threshold


class FaceRecognitionSystem:
    """Main face recognition system using MediaPipe"""

    def __init__(self):
        self.landmark_extractor = FaceLandmarkExtractor()
        self.face_matcher = FaceMatcher()

        # Data storage
        self.data_dir = "face_data_mediapipe"
        self.landmarks_file = os.path.join(self.data_dir, "face_landmarks.pkl")
        self.known_faces = {}

        self.create_directories()
        self.load_known_faces()

    def create_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(os.path.join(self.data_dir, "images")):
            os.makedirs(os.path.join(self.data_dir, "images"))

    def load_known_faces(self):
        """Load known face landmarks from pickle file"""
        if os.path.exists(self.landmarks_file):
            try:
                with open(self.landmarks_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                print(f"Error loading face landmarks: {e}")

    def save_known_faces(self):
        """Save known face landmarks to pickle file"""
        try:
            with open(self.landmarks_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
            print("Face landmarks saved successfully")
        except Exception as e:
            print(f"Error saving face landmarks: {e}")

    def add_new_face(self, name, image_paths):
        """Add new faces to the system"""
        landmarks_list = []

        for image_path in image_paths:
            try:
                # Extract landmarks
                landmarks = self.landmark_extractor.extract_landmarks_from_file(image_path)

                if landmarks is not None:
                    landmarks_list.append(landmarks)
                    print(f"Successfully extracted landmarks from {image_path}")
                else:
                    print(f"No face detected in {image_path}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        if not landmarks_list:
            return False, "No valid faces detected in the provided images"

        # Store landmarks
        if name in self.known_faces:
            self.known_faces[name].extend(landmarks_list)
        else:
            self.known_faces[name] = landmarks_list

        # Save to file
        self.save_known_faces()

        return True, f"Successfully added {name} with {len(landmarks_list)} face samples"

    def remove_face(self, name):
        """Remove a face from the system"""
        if name in self.known_faces:
            del self.known_faces[name]
            self.save_known_faces()
            return True, f"Successfully removed {name} from the database"
        else:
            return False, f"{name} not found in database"

    def recognize_face(self, image):
        """Recognize face in image"""
        # Extract landmarks
        landmarks = self.landmark_extractor.extract_landmarks(image)

        if landmarks is None:
            return "Unknown", 0.0, "No face detected"

        # Match against known faces
        best_match, distance = self.face_matcher.match_faces(landmarks, self.known_faces)

        if best_match is None:
            return "Unknown", 0.0, "No match found"

        # Calculate confidence (inverse of distance)
        confidence = max(0, 1 - (distance / self.face_matcher.threshold))

        if self.face_matcher.is_same_person(distance):
            return best_match, confidence, "Success"
        else:
            return "Unknown", confidence, "Distance too high"

    def get_known_faces(self):
        """Get list of known faces"""
        return list(self.known_faces.keys())


class VideoCapture:
    """Handle video capture for real-time recognition"""

    def __init__(self):
        self.cap = None
        self.is_running = False

    def start_capture(self):
        """Start video capture"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        self.is_running = True

    def stop_capture(self):
        """Stop video capture"""
        self.is_running = False
        if self.cap:
            self.cap.release()

    def read_frame(self):
        """Read frame from camera"""
        if not self.is_running or not self.cap:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()


class FaceRecognitionGUI:
    """GUI for the face recognition system"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition System - MediaPipe")
        self.root.geometry("1200x700")

        self.face_system = FaceRecognitionSystem()
        self.video_capture = VideoCapture()
        self.is_running = False

        self.setup_gui()

    def setup_gui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Video frame
        self.video_frame = ttk.LabelFrame(main_frame, text="Live Video Feed")
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = ttk.Label(self.video_frame, text="Click 'Start Recognition' to begin")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side="right", fill="y")

        # Recognition controls
        recog_frame = ttk.LabelFrame(control_frame, text="Recognition Controls")
        recog_frame.pack(fill="x", pady=(0, 10))

        ttk.Button(recog_frame, text="Start Recognition",
                   command=self.start_recognition).pack(fill="x", padx=5, pady=5)
        ttk.Button(recog_frame, text="Stop Recognition",
                   command=self.stop_recognition).pack(fill="x", padx=5, pady=5)

        # Face management
        face_frame = ttk.LabelFrame(control_frame, text="Face Management")
        face_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(face_frame, text="Name:").pack(anchor="w", padx=5, pady=(5, 0))
        self.name_entry = ttk.Entry(face_frame)
        self.name_entry.pack(fill="x", padx=5, pady=(0, 5))

        ttk.Button(face_frame, text="Add New Face",
                   command=self.add_face).pack(fill="x", padx=5, pady=5)
        ttk.Button(face_frame, text="Remove Face",
                   command=self.remove_face).pack(fill="x", padx=5, pady=5)

        # Face list
        list_frame = ttk.LabelFrame(control_frame, text="Registered Faces")
        list_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.face_listbox = tk.Listbox(list_frame)
        self.face_listbox.pack(fill="both", expand=True, padx=5, pady=5)

        # Recognition results
        results_frame = ttk.LabelFrame(control_frame, text="Recognition Results")
        results_frame.pack(fill="x", pady=(0, 10))

        self.result_label = ttk.Label(results_frame, text="No recognition yet")
        self.result_label.pack(padx=5, pady=5)

        self.confidence_label = ttk.Label(results_frame, text="")
        self.confidence_label.pack(padx=5, pady=5)

        # Status
        status_frame = ttk.LabelFrame(control_frame, text="System Status")
        status_frame.pack(fill="x")

        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(padx=5, pady=5)

        self.update_face_list()

    def update_face_list(self):
        """Update the face listbox"""
        self.face_listbox.delete(0, tk.END)
        for name in self.face_system.get_known_faces():
            self.face_listbox.insert(tk.END, name)

    def add_face(self):
        """Add a new face to the system"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return

        # Open file dialog to select images
        image_files = filedialog.askopenfilenames(
            title="Select face images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if image_files:
            success, message = self.face_system.add_new_face(name, image_files)

            if success:
                messagebox.showinfo("Success", message)
                self.name_entry.delete(0, tk.END)
                self.update_face_list()
            else:
                messagebox.showerror("Error", message)

    def remove_face(self):
        """Remove a face from the system"""
        selection = self.face_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "Please select a face to remove")
            return

        name = self.face_listbox.get(selection[0])
        success, message = self.face_system.remove_face(name)

        if success:
            messagebox.showinfo("Success", message)
            self.update_face_list()
        else:
            messagebox.showerror("Error", message)

    def start_recognition(self):
        """Start the face recognition process"""
        if self.is_running:
            return

        try:
            self.video_capture.start_capture()
            self.is_running = True
            self.status_label.config(text="Recognition Active")

            # Start recognition in a separate thread
            self.recognition_thread = threading.Thread(target=self.recognition_loop)
            self.recognition_thread.daemon = True
            self.recognition_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Could not start camera: {str(e)}")

    def stop_recognition(self):
        """Stop the face recognition process"""
        self.is_running = False
        self.video_capture.stop_capture()
        self.status_label.config(text="Stopped")
        self.video_label.config(text="Click 'Start Recognition' to begin")

    def recognition_loop(self):
        """Main recognition loop"""
        while self.is_running:
            frame = self.video_capture.read_frame()
            if frame is None:
                continue

            # Recognize face
            name, confidence, status = self.face_system.recognize_face(frame)

            # Draw results on frame
            cv2.putText(frame, f"Name: {name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert frame to PhotoImage for tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_tk = ImageTk.PhotoImage(frame_pil)

            # Update video label
            self.video_label.config(image=frame_tk)
            self.video_label.image = frame_tk

            # Update results
            self.root.after(0, lambda: self.update_results(name, confidence, status))

            # Small delay
            time.sleep(0.03)

    def update_results(self, name, confidence, status):
        """Update recognition results"""
        self.result_label.config(text=f"Recognized: {name}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}")

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

    def __del__(self):
        """Cleanup"""
        if self.video_capture:
            self.video_capture.stop_capture()


def main():
    """Main function to run the face recognition system"""
    print("Starting Face Recognition System (MediaPipe)...")
    print("Features:")
    print("- Facial landmark detection using MediaPipe")
    print("- 478 landmark points per face")
    print("- Lightweight and fast processing")
    print("- Real-time face recognition")
    print("- Modular and extensible architecture")
    print("- User-friendly GUI interface")

    app = FaceRecognitionGUI()
    app.run()


if __name__ == "__main__":
    main()