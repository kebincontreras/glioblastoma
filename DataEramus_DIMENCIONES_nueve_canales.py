import os
import nibabel as nib
import numpy as np

# Directorio base donde están almacenados los datos
base_path = r"C:\\Users\\USUARIO\Desktop\\DatasetEramus"

# Función para verificar y ajustar los cosenos direccionales si es necesario
def adjust_affine(affine):
    # Aquí puedes agregar condiciones específicas si los cosenos no son estándar
    # Esta es una implementación básica y podría necesitar ajustes específicos
    return affine

# Función para extraer 9 cortes axiales del centro
def extract_axial_slices(img_data, num_slices=9):
    z_dimension = img_data.shape[2]
    middle = z_dimension // 2
    start_slice = max(middle - num_slices // 2, 0)  # Asegúrate de no salirte del rango inferior
    end_slice = min(start_slice + num_slices, z_dimension)  # Asegúrate de no salirte del rango superior
    axial_slices = img_data[:, :, start_slice:end_slice]
    return axial_slices

# Recorremos cada subdirectorio en el directorio base
for i in range(1, 775):  # desde EGD-0001 a EGD-0774
    folder_name = f"EGD-0{str(i).zfill(3)}"
    folder_path = os.path.join(base_path, folder_name)
    image_path = os.path.join(folder_path, "T1.nii.gz")
    
    if os.path.exists(image_path):
        # Cargamos la imagen NIfTI
        img = nib.load(image_path)
        data = img.get_fdata()
        affine = img.affine
        
        # Ajustar la matriz affine si es necesario
        affine = adjust_affine(affine)
        
        # Extraer 9 cortes axiales del centro
        axial_slices = extract_axial_slices(data)
        
        # Crear nueva imagen NIfTI con los cortes axiales
        new_img = nib.Nifti1Image(axial_slices, affine)
        new_image_path = os.path.join(folder_path, "T1_axial_slices.nii.gz")
        nib.save(new_img, new_image_path)
        print(f"Guardado {new_image_path}")
    else:
        print(f"Archivo no encontrado: {image_path}")

