import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import pickle
from PIL import Image

filename = "svm_model.pickle"

# Load model from sklearn with pickle
print("Loading model...")
svm_model = pickle.load(open(filename, "rb"))
print("Model succesfully loaded!!!")

img_saved = False
image_path = 'image.png'

def preprocess_image(image_path, threshold=0.01):
    image = Image.open(image_path)
    # Resize to 28x28 pixels
    image = image.resize((28, 28))
    # Convert to grayscale
    image = image.convert('L')
    # Normalize and binarize the image using the specified threshold
    image = np.asarray(image)
    image = image.astype('float32') / 255.0
    binarized_image = (image > threshold).astype(np.float32)
    # Flatten the image
    extended_image = extend_image(binarized_image)
    flattened_image = extended_image.flatten()
    # Reshape to match the input shape of the SVM model
    flattened_image = flattened_image.reshape(1, -1)
    return flattened_image

def extend_image(binarized_image):
    extended_image = np.copy(binarized_image)

    # Up
    shift_up = np.zeros_like(binarized_image)
    shift_up[:-1, :] = binarized_image[1:, :]

    # Down
    shift_down = np.zeros_like(binarized_image)
    shift_down[1:, :] = binarized_image[:-1, :]

    # Left
    shift_left = np.zeros_like(binarized_image)
    shift_left[:, :-1] = binarized_image[:, 1:]

    # Right
    shift_right = np.zeros_like(binarized_image)
    shift_right[:, 1:] = binarized_image[:, :-1]

    extended_image = np.maximum(extended_image, shift_up)
    extended_image = np.maximum(extended_image, shift_down)
    extended_image = np.maximum(extended_image, shift_left)
    extended_image = np.maximum(extended_image, shift_right)
    
    return extended_image


# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Inicializar MediaPipe Drawing para dibujar las anotaciones
mp_drawing = mp.solutions.drawing_utils

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

# Lista para almacenar las coordenadas de la punta del dedo índice
index_finger_positions = []
is_index_finger_up = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar la imagen y detectar las manos
    results = hands.process(rgb_frame)

    # Dibujar las anotaciones en la imagen
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Obtener la coordenada de la punta del dedo índice (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Verificar si el dedo índice está levantado (más arriba que la base del dedo)
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            if index_finger_tip.y < index_finger_mcp.y:
                index_finger_positions.append((x, y))
                is_index_finger_up = True
                cv2.putText(frame, f'Index Tip: ({x}, {y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                if is_index_finger_up:
                    # Crear una nueva ventana con el recorrido del dedo índice
                    if len(index_finger_positions) > 1:
                        # Encontrar las coordenadas mínimas y máximas
                        x_coords, y_coords = zip(*index_finger_positions)
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        # Crear una matriz de ceros del tamaño correspondiente
                        matrix_width = x_max - x_min + 1
                        matrix_height = y_max - y_min + 1
                        matrix = np.zeros((matrix_height, matrix_width), dtype=int)
                        
                        # Marcar las posiciones del recorrido del dedo con 1 en la matriz
                        for (x, y) in index_finger_positions:
                            matrix[y - y_min, x - x_min] = 1
                        
                        # Mostrar la matriz
                        plt.figure()
                        plt.imshow(matrix, cmap='gray')
                        plt.axis('off')  # Oculta los ejes
                        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)  # Guarda la imagen
                        plt.show()
                        img_saved = True
                    
                    index_finger_positions.clear()
                    is_index_finger_up = False

    # Mostrar la imagen
    cv2.imshow('Hand Detector', frame)


    if (img_saved):
        preprocessed_image = preprocess_image(image_path)
        predicted_digit = svm_model.predict(preprocessed_image)
        print("Predicted Digit: ", predicted_digit)
        img_saved = False

    # Salir del loop al presionar la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()
