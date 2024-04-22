import os
import json
import nibabel as nib

# Ruta de la carpeta principal donde están los datos
base_path = "C:\\Users\\USUARIO\\Downloads\\Dataset_Eramus\\Dataset_Eramus"

# Crear la carpeta de resultados si no existe
results_path = os.path.join(base_path, "Resultados")
os.makedirs(results_path, exist_ok=True)

# Iterar sobre cada carpeta de paciente
for i in range(1, 775):  # Asumiendo que van del 001 al 774
    patient_id = f"EGD-0{str(i).zfill(3)}"  # Formatear el número con ceros a la izquierda
    patient_folder = os.path.join(base_path, patient_id)
    
    # Rutas al archivo NIfTI y al JSON
    nifti_path = os.path.join(patient_folder, "T1.nii.gz")
    json_path = os.path.join(patient_folder, "metadata.json")
    
    # Cargar los metadatos desde JSON
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    # Cargar el archivo NIfTI
    img = nib.load(nifti_path)
    header = img.header
    
    # Modificar el header del NIfTI con la información del JSON
    header['descrip'] = json.dumps(metadata)  # Convertir el JSON a string y guardar en descrip
    
    # Guardar el archivo NIfTI modificado
    new_folder = os.path.join(results_path, patient_id)
    os.makedirs(new_folder, exist_ok=True)
    new_nifti_path = os.path.join(new_folder, "T1_modified.nii.gz")
    nib.save(img, new_nifti_path)

print("Proceso completado. Los archivos modificados han sido guardados.")
