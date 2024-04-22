import os
import nibabel as nib
import matplotlib.pyplot as plt

# Directorio base donde están almacenados los datos
base_path = r"C:\\Users\\USUARIO\\Desktop\\DatasetEramus"

# Listas para guardar las dimensiones
dimensions = []

# Recorremos cada subdirectorio en el directorio base
for i in range(1, 775):  # desde EGD-0001 a EGD-0774
    folder_name = f"EGD-0{str(i).zfill(3)}"
    folder_path = os.path.join(base_path, folder_name)
    image_path = os.path.join(folder_path, "T1_axial_slices.nii.gz")
    
    if os.path.exists(image_path):
        # Cargamos la imagen NIfTI
        img = nib.load(image_path)
        # Obtenemos y almacenamos las dimensiones de la imagen
        dimensions.append(img.shape)
    else:
        print(f"Archivo no encontrado: {image_path}")

# Comprobamos las dimensiones
expected_dims = (512, 512, 5)
all_match = all(dim == expected_dims for dim in dimensions)

# Graficar las dimensiones encontradas
fig, ax = plt.subplots()
# Contar frecuencia de cada conjunto de dimensiones
dim_counts = {str(dim): dimensions.count(dim) for dim in set(map(tuple, dimensions))}
ax.bar(dim_counts.keys(), dim_counts.values(), color='blue')
ax.set_title('Frecuencia de las Dimensiones de las Imágenes')
ax.set_xlabel('Dimensiones')
ax.set_ylabel('Número de Imágenes')
plt.xticks(rotation=45)
plt.show()

# Imprimir si todas las imágenes cumplen con las dimensiones esperadas
if all_match:
    print("Todas las imágenes tienen las dimensiones esperadas: 512x512x5")
else:
    print("Algunas imágenes no tienen las dimensiones esperadas.")
