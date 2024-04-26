import os

# Directorio base donde están almacenados los datos
base_path = r"C:\Users\USUARIO\Downloads\Dataset_Eramus\Dataset_Eramus"

# Nombre del archivo adicional que deseas eliminar
additional_file_name = 'T1_axial_slices.nii.gz'

# Recorremos cada subdirectorio en el directorio base
for i in range(1, 775):  # desde EGD-0001 a EGD-0774
    folder_name = f"EGD-0{str(i).zfill(3)}"
    folder_path = os.path.join(base_path, folder_name)
    additional_file_path = os.path.join(folder_path, additional_file_name)
    
    if os.path.exists(additional_file_path):
        # Eliminar el archivo adicional
        os.remove(additional_file_path)
        print(f"Archivo eliminado: {additional_file_path}")
    else:
        print(f"Archivo no encontrado o ya eliminado: {additional_file_path}")


