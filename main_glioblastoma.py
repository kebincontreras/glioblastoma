import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

'''
def load_and_preprocess_data(patient_paths):
    images = []
    labels = []
    for path in patient_paths:
        # Determinar automáticamente el tipo de dataset
        if 'Eramus' in path:
            image_path = os.path.join(path, 'T1_axial_slices.nii.gz')
            label = 1  # Todos los pacientes tienen la enfermedad en este dataset
            dataset_type = 'Eramus'
        else:  # Asume que cualquier otro path proviene de RSNA_MICCAI
            image_path = os.path.join(path, 'image.nii')
            label_path = os.path.join(path, 'label.txt')
            with open(label_path, 'r') as file:
                label = int(file.read().strip())
            dataset_type = 'RSNA_MICCAI'

        img = nib.load(image_path)
        data = img.get_fdata()
        data = np.expand_dims(data, axis=-1)  # Asegúrate que el formato coincide con el modelo
        images.append(data)
        labels.append(label)
    return np.array(images), np.array(labels)
'''

def load_and_preprocess_data(patient_paths):
    images = []
    labels = []
    for path in patient_paths:
        # Determinar automáticamente el tipo de dataset
        if 'Eramus' in path:
            image_path = os.path.join(path, 'T1_axial_slices.nii.gz')
            label = 1  # Todos los pacientes tienen la enfermedad en este dataset
            dataset_type = 'Eramus'
            # Cargar y procesar la imagen, ya que siempre tiene la etiqueta 1
            img = nib.load(image_path)
            data = img.get_fdata()
            data = np.expand_dims(data, axis=-1)  # Asegúrate que el formato coincide con el modelo
            images.append(data)
            labels.append(label)
        else:
            # Asume que cualquier otro path proviene de RSNA_MICCAI
            image_path = os.path.join(path, 'image.nii')
            label_path = os.path.join(path, 'label.txt')
            with open(label_path, 'r') as file:
                label = int(file.read().strip())
            dataset_type = 'RSNA_MICCAI'
            if label == 0:  # Solo procesar si la etiqueta es 0
                img = nib.load(image_path)
                data = img.get_fdata()
                data = np.expand_dims(data, axis=-1)  # Asegúrate que el formato coincide con el modelo
                images.append(data)
                labels.append(label)
            # Si la etiqueta es 1, no hagas nada (continuar al próximo path)
    return np.array(images), np.array(labels)

# Cargar pacientes
data_dir_eramus = 'C:\\Users\\USUARIO\\Desktop\\Mestria\\DatasetEramus'

patients_eramus = [os.path.join(data_dir_eramus, name) for name in os.listdir(data_dir_eramus) if os.path.isdir(os.path.join(data_dir_eramus, name))]

data_dir_rsna = 'C:\\Users\\USUARIO\\Desktop\\Mestria\\Dataset_RSNA_MICCAI_1'
patients_rsna = [os.path.join(data_dir_rsna, f'Patient_{i}') for i in range(1, 200)]

# Dividir el conjunto de Eramus
train_val_patients, test_patients_eramus = train_test_split(patients_eramus, test_size=0.9, random_state=42)  # 80% para test
train_patients_eramus, val_patients_eramus = train_test_split(train_val_patients, test_size=1/2, random_state=42)  # 1/8 de 80% es 10%

# Añadir pacientes de RSNA_MICCAI al conjunto de entrenamiento
train_patients = train_patients_eramus + patients_rsna

# Carga y prepara los datos
train_images, train_labels = load_and_preprocess_data(train_patients)
val_images, val_labels = load_and_preprocess_data(val_patients_eramus)
test_images, test_labels = load_and_preprocess_data(test_patients_eramus)



import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

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




