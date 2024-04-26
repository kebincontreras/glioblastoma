import os

# Ruta de la carpeta principal donde se encuentran las subcarpetas
root_dir = "C:\\Users\\USUARIO\\Desktop\\DatasetEramus"

# Recorrer cada subcarpeta dentro del directorio
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    # Verificar si es un directorio y si sigue el patrón EGD-0###
    if os.path.isdir(subdir_path) and subdir.startswith("EGD-0"):
        # Recorrer cada archivo dentro de la subcarpeta
        for file in os.listdir(subdir_path):
            # Si el archivo no es T1.nii.gz ni metadata.json, se elimina
            if file != "T1.nii.gz" and not file.startswith("metadata"):
                file_path = os.path.join(subdir_path, file)
                try:
                    os.remove(file_path)
                    print(f"Archivo eliminado: {file_path}")
                except OSError as e:
                    print(f"Error al eliminar archivo {file_path}: {e.strerror}")

print("Limpieza de archivos completada.")
