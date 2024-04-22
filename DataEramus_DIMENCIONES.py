import os
import nibabel as nib
import matplotlib.pyplot as plt

# Directorio base donde están almacenados los datos
base_path = r"C:\Users\USUARIO\Downloads\Dataset_Eramus\Dataset_Eramus"

# Lista para guardar las dimensiones
dimensions = []

# Recorremos cada subdirectorio en el directorio base
for i in range(1, 775):  # desde EGD-0001 a EGD-0774
    folder_name = f"EGD-0{str(i).zfill(3)}"
    folder_path = os.path.join(base_path, folder_name)
    image_path = os.path.join(folder_path, "T1.nii.gz")
    
    if os.path.exists(image_path):
        # Cargamos la imagen NIfTI
        img = nib.load(image_path)
        # Obtenemos las dimensiones de la imagen
        dimensions.append(img.shape)
    else:
        print(f"Archivo no encontrado: {image_path}")

# Generar el histograma de las dimensiones
# Aplanar la lista de dimensiones a una lista simple para el histograma
flat_dimensions = [dim for sublist in dimensions for dim in sublist]

plt.hist(flat_dimensions, bins=50, alpha=0.75)
plt.title('Histograma de dimensiones de imágenes MRI')
plt.xlabel('Dimensiones')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.show()
