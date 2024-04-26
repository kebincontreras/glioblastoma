import os
import nibabel as nib
import matplotlib.pyplot as plt

# Directorio base donde están almacenados los datos
base_path = r"C:\Users\USUARIO\Downloads\Dataset_Eramus\Dataset_Eramus"

# Listas para guardar las dimensiones por separado
dim1 = []
dim2 = []
dim3 = []

# Recorremos cada subdirectorio en el directorio base
for i in range(1, 775):  # desde EGD-0001 a EGD-0774
    folder_name = f"EGD-0{str(i).zfill(3)}"
    folder_path = os.path.join(base_path, folder_name)
    image_path = os.path.join(folder_path, "T1.nii.gz")
    
    if os.path.exists(image_path):
        # Cargamos la imagen NIfTI
        img = nib.load(image_path)
        # Obtenemos las dimensiones de la imagen y las guardamos en listas separadas
        if len(img.shape) == 3:  # Asegurarse de que la imagen tiene tres dimensiones
            dim1.append(img.shape[0])
            dim2.append(img.shape[1])
            dim3.append(img.shape[2])
        else:
            print(f"La imagen en {image_path} no tiene tres dimensiones")
    else:
        print(f"Archivo no encontrado: {image_path}")

# Función para plotear las dimensiones
def plot_dimension(dim_list, dimension_label):
    plt.figure()
    plt.hist(dim_list, bins=50, alpha=0.75, color='blue')
    plt.title(f'Histograma de la {dimension_label} dimensión de imágenes MRI')
    plt.xlabel(dimension_label)
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

# Imprimir y graficar las dimensiones
print("Distribución de la primera dimensión:")
plot_dimension(dim1, "primera")

print("Distribución de la segunda dimensión:")
plot_dimension(dim2, "segunda")

print("Distribución de la tercera dimensión:")
plot_dimension(dim3, "tercera")


