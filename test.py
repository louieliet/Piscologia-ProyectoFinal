import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Inicializar la webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Convertir imagen a BGR a RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar imagen con MediaPipe Holistic
        results = holistic.process(image)
        
        # Dibujar puntos clave y marcadores de MediaPipe Holistic en la imagen
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        
        # Mostrar imagen procesada en una ventana
        cv2.imshow('MediaPipe Holistic', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Esperar a que se presione la tecla 'q' para cerrar la ventana
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Liberar la webcam y cerrar la ventana
cap.release()
cv2.destroyAllWindows()
