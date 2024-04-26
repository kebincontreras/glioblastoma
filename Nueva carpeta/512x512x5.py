import os
import numpy as np
import nibabel as nib

# Directorio base donde están almacenados los datos
base_path = r"C:\Users\USUARIO\Desktop\DatasetEramus"

# Dimensiones deseadas después del padding
target_dims = (512, 512, 9)

# Función para seleccionar cortes y aplicar padding
def adjust_image_dims(data):
    # Seleccionar 5 cortes axiales equidistantes
    selected_slices = np.linspace(0, data.shape[2] - 1, target_dims[2], dtype=int)
    data = data[:, :, selected_slices]
    
    # Aplicar padding para ajustar a las dimensiones deseadas
    padded_data = np.zeros(target_dims)
    
    # Calcular los márgenes de padding para centrar la imagen
    pad_height = (target_dims[0] - data.shape[0]) // 2
    pad_width = (target_dims[1] - data.shape[1]) // 2
    
    # Aplicar el padding en altura y anchura
    padded_data = np.pad(data, ((pad_height, target_dims[0] - data.shape[0] - pad_height),
                                (pad_width, target_dims[1] - data.shape[1] - pad_width),
                                (0, 0)), mode='constant', constant_values=0)
    
    return padded_data

# Recorremos cada subdirectorio en el directorio base
for i in range(1, 775):  # desde EGD-0001 a EGD-0774
    folder_name = f"EGD-0{str(i).zfill(3)}"
    folder_path = os.path.join(base_path, folder_name)
    image_path = os.path.join(folder_path, "T1.nii.gz")
    
    if os.path.exists(image_path):
        # Cargamos la imagen NIfTI
        img = nib.load(image_path)
        data = img.get_fdata()
        
        # Ajustar dimensiones y aplicar padding
        padded_data = adjust_image_dims(data)
        
        # Crear nueva imagen NIfTI con las dimensiones ajustadas
        new_img = nib.Nifti1Image(padded_data, img.affine)  # usar la matriz affine original
        
        # Sobrescribir la imagen original
        nib.save(new_img, image_path)
        print(f"Imagen ajustada y sobrescrita en {image_path}")
    else:
        print(f"Archivo no encontrado: {image_path}")
