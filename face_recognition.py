from deepface import DeepFace 
import cv2
import os

# Define the directory containing known faces
known_faces_dir = "known_faces"
known_faces = {}

# Load images of known faces
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(known_faces_dir, filename)
        known_faces[filename] = path

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Start face recognition
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (DeepFace expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Loop through each known face
        recognized = False
        for name, image_path in known_faces.items():
            result = DeepFace.verify(img1_path=image_path, img2_path=frame_rgb, enforce_detection=False)
            
            if result['verified']:
                # Display the name if a match is found
                cv2.putText(frame, f"{name.split('.')[0]}: Match", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                recognized = True
                break

        # If no match was found, display "Unknown"
        if not recognized:
            cv2.putText(frame, "Unknown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    except Exception as e:
        print(f"Error in face verification: {e}")

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
