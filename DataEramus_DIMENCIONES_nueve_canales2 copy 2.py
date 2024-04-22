import os
import nibabel as nib

# Directorio base donde están almacenados los datos
base_path = r"C:\Users\USUARIO\Downloads\Dataset_Eramus\Dataset_Eramus"

# Dimensiones esperadas
expected_dimensions = (197, 233, 9)

# Flag para seguir si todas las imágenes cumplen
all_match = True

# Recorremos cada subdirectorio en el directorio base
for i in range(1, 775):  # desde EGD-0001 a EGD-0774
    folder_name = f"EGD-0{str(i).zfill(3)}"
    folder_path = os.path.join(base_path, folder_name)
    image_path = os.path.join(folder_path, "T1.nii.gz")
    
    if os.path.exists(image_path):
        # Cargamos la imagen NIfTI
        img = nib.load(image_path)
        # Obtenemos las dimensiones de la imagen
        dimensions = img.shape
        
        # Comprobar si las dimensiones son las esperadas
        if dimensions != expected_dimensions:
            print(f"La imagen en {image_path} tiene dimensiones incorrectas: {dimensions}")
            all_match = False
    else:
        print(f"Archivo no encontrado: {image_path}")

# Si todas las imágenes cumplen con las dimensiones esperadas
if all_match:
    print("Todas las imágenes tienen las dimensiones esperadas: 197x233x9")
else:
    print("Algunas imágenes no tienen las dimensiones esperadas.")

