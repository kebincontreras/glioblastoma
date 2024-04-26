import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import os
import random
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Función para cargar y preprocesar datos
def load_and_preprocess_data(patient_paths):
    images = []
    labels = []
    count_eramus = 0  # Contador para pacientes de Eramus
    for path in patient_paths:
        if 'Eramus' in path and count_eramus < 700:
            image_path = os.path.join(path, 'T1_axial_slices.nii.gz')
            label = 1  # Todos los pacientes tienen la enfermedad en este dataset
            img = nib.load(image_path)
            data = img.get_fdata()
            data = np.expand_dims(data, axis=-1)  # Asegúrate que el formato coincide con el modelo
            images.append(data)
            labels.append(label)
            count_eramus += 1  # Incrementa el contador de pacientes de Eramus
        elif 'Eramus' not in path:
            image_path = os.path.join(path, 'image.nii')
            label_path = os.path.join(path, 'label.txt')
            with open(label_path, 'r') as file:
                label = int(file.read().strip())
            if label == 0:  # Solo procesar imágenes con etiqueta 0
                img = nib.load(image_path)
                data = img.get_fdata()
                data = np.expand_dims(data, axis=-1)
                images.append(data)
                labels.append(label)
    return np.array(images), np.array(labels)



# Cargar pacientes
data_dir_eramus = 'C:\\Users\\USUARIO\\Desktop\\Mestria\\DatasetEramus'
patients_eramus = [os.path.join(data_dir_eramus, name) for name in os.listdir(data_dir_eramus) if os.path.isdir(os.path.join(data_dir_eramus, name))]
data_dir_rsna = 'C:\\Users\\USUARIO\\Desktop\\Mestria\\Dataset_RSNA_MICCAI_1'
patients_rsna = [os.path.join(data_dir_rsna, f'Patient_{i}') for i in range(1, 200)]

# Cargar y procesar datos de ambos conjuntos
rsna_images, rsna_labels = load_and_preprocess_data(patients_rsna)
eramus_images, eramus_labels = load_and_preprocess_data(patients_eramus)

# Unir todos los pacientes, imágenes y etiquetas
total_patients = patients_eramus + patients_rsna
total_images = np.concatenate((rsna_images, eramus_images), axis=0)
total_labels = np.concatenate((rsna_labels, eramus_labels), axis=0)

# Mezclar los datos
indices = list(range(len(total_images)))
random.shuffle(indices)
shuffled_images = total_images[indices]
shuffled_labels = total_labels[indices]

# Dividir el conjunto de datos
train_val_images, test_images, train_val_labels, test_labels = train_test_split(shuffled_images, shuffled_labels, test_size=0.9, random_state=20)
train_images, val_images, train_labels, val_labels = train_test_split(train_val_images, train_val_labels, test_size=1/5, random_state=20)

# Continuar con la configuración de la GPU, carga del modelo, y entrenamiento



# Configurar el uso de GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Establece la memoria de la GPU para que se expanda según sea necesario
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Opcionalmente, establecer un límite máximo de memoria (en GB)
        # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU instead.")

# Cargar el modelo preentrenado
model_path = 'C:\\Users\\USUARIO\\Desktop\\Mestria\\best_model.h5'
model = load_model(model_path)

# Configurar el optimizador y compilar el modelo
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    epochs=200,
                    batch_size=32)  # Ajusta el tamaño del lote según las capacidades de tu máquina

model.save('C:\\Users\\USUARIO\\Desktop\\Mestria\\modelo_full_entrenado.h5')

# Graficar las curvas de aprendizaje
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.title('Curva de Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.title('Curva de Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()

# Evaluar el modelo
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f'Pérdida en el conjunto de prueba: {test_loss}')
print(f'Precisión en el conjunto de prueba: {test_accuracy}')
