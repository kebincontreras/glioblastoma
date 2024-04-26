import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Configurar PyTorch para usar la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ", device)

def load_and_preprocess_data(patient_paths):
    images = []
    labels = []
    for path in patient_paths:
        if 'Eramus' in path:
            image_path = os.path.join(path, 'T1_axial_slices.nii.gz')
            label = 1
        else:
            image_path = os.path.join(path, 'image.nii')
            label_path = os.path.join(path, 'label.txt')
            with open(label_path, 'r') as file:
                label = int(file.read().strip())

        img = nib.load(image_path)
        data = img.get_fdata()
        data = np.expand_dims(data, axis=0)  # Añadir canal en el frente
        images.append(data)
        labels.append(label)

    # Convertir la lista de arrays a un único array de NumPy antes de hacer la conversión a tensor
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    return images, labels


# Cargar pacientes y dividir los datos
# (Asegúrate de actualizar las rutas y los nombres de directorios)
data_dir_eramus = 'C:\\Users\\USUARIO\\Desktop\\Mestria\\DatasetEramus'
patients_eramus = [os.path.join(data_dir_eramus, name) for name in os.listdir(data_dir_eramus) if os.path.isdir(os.path.join(data_dir_eramus, name))]

data_dir_rsna = 'C:\\Users\\USUARIO\\Desktop\\Mestria\\Dataset_RSNA_MICCAI_1'
patients_rsna = [os.path.join(data_dir_rsna, f'Patient_{i}') for i in range(1, 200)]

# Dividir el conjunto de Eramus
train_val_patients, test_patients_eramus = train_test_split(patients_eramus, test_size=0.8, random_state=42)
train_patients_eramus, val_patients_eramus = train_test_split(train_val_patients, test_size=1/2, random_state=42)

# Añadir pacientes de RSNA_MICCAI al conjunto de entrenamiento
train_patients = train_patients_eramus + patients_rsna

# Carga y prepara los datos
train_images, train_labels = load_and_preprocess_data(train_patients)
val_images, val_labels = load_and_preprocess_data(val_patients_eramus)
test_images, test_labels = load_and_preprocess_data(test_patients_eramus)



# Configurar DataLoaders
batch_size = 64  # Define el tamaño de lote que prefieras

train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_images, val_labels)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(test_images, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# Cargar el modelo
model_torch = ModeloPyTorch().to(device)
model_torch.load_state_dict(torch.load('C:\\Users\\USUARIO\\Desktop\\Mestria\\best_model.pth'))

# Configurar el optimizador y la función de pérdida
import torch.optim as optim
optimizer = optim.Adam(model_torch.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Función para un paso de entrenamiento
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(dataloader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

# Similar para validación
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss = running_loss / len(dataloader)
    val_acc = 100 * correct / total
    return val_loss, val_acc



# Ejecutar entrenamiento y validación
num_epochs = 10  # Define el número de épocas que prefieras
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model_torch, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model_torch, val_loader, criterion, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}')
