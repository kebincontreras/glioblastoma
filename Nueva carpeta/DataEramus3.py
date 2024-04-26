import os
import json
import nibabel as nib
import numpy as np 

# Ruta de la carpeta principal donde están los datos
base_path = "C:\\Users\\USUARIO\\Downloads\\Dataset_Eramus\\Dataset_Eramus\\Resultados"

# Iterar sobre cada carpeta de paciente en la carpeta "Resultados"
for i in range(1, 775):  # Asumiendo que van del 001 al 774
    patient_id = f"EGD-0{str(i).zfill(3)}"  # Formatear el número con ceros a la izquierda
    patient_folder = os.path.join(base_path, patient_id)
    nifti_path = os.path.join(patient_folder, "T1_modified.nii.gz")
    
    # Verificar si el archivo NIfTI modificado existe antes de intentar abrirlo
    if os.path.exists(nifti_path):
        img = nib.load(nifti_path)
        header = img.header
        
        # Leer la descripción donde se almacenaron los metadatos
        descrip = header.get('descrip', b'')
        print(f"Tipo de 'descrip': {type(descrip)}")
        print(f"Contenido de 'descrip': {descrip}")
        
        # Asegurarse de que 'descrip' sea un array de bytes y convertirlo a cadena de texto
        if isinstance(descrip, numpy.ndarray):
            descrip = descrip.tobytes().decode('utf-8')
        
        if descrip:
            # Convertir el contenido del descrip (JSON en texto) a un diccionario
            try:
                metadata = json.loads(descrip)
                # Extraer la edad y el género del paciente desde el JSON incorporado
                age = metadata['Clinical_data']['Age']
                gender = metadata['Clinical_data']['Sex']
                print(f"Paciente ID: {patient_id}, Edad: {age}, Género: {gender}")
            except json.JSONDecodeError:
                print(f"Los metadatos de 'descrip' para el paciente {patient_id} no están en formato JSON o no se pudieron decodificar correctamente.")
        else:
            print(f"No hay metadatos almacenados en 'descrip' para el paciente {patient_id}")
    else:
        print(f"No se encontró el archivo NIfTI modificado para el paciente ID: {patient_id}")

print("Proceso de impresión completado.")


