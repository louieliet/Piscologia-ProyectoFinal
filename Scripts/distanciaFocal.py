import cv2
import mediapipe as mp
import numpy as np
import math

# Configuración de la cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Configuración de la detección facial
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Constantes para la triangulación
focal_length = 875 # Valor aproximado para la cámara de la laptop
known_distance = 50 # Distancia en cm de la persona a la cámara para calibración
known_width = 16 # Ancho en cm del rostro de la persona para calibración
face_points = [33, 133, 362, 263] # Puntos clave de la cara (nariz, barbilla, ojos)

while True:
    # Capturar el frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break
    
    # Procesar el frame para detectar la cara
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    # Dibujar la malla facial si se detecta una cara
    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        mp_drawing = mp.solutions.drawing_utils 
        mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_CONTOURS)

        # Calcular la distancia de la persona a la cámara
        points = []
        for p in face_points:
            landmark = face.landmark[p]
            x, y, z = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]), landmark.z
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            
        if len(points) == len(face_points):
            pixel_width = abs(points[0][0] - points[3][0])
            distance = (known_width * focal_length) / pixel_width
            cv2.putText(frame, f"{distance:.2f} cm", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Mostrar el frame en una ventana
    cv2.imshow('Face Distance', frame)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
