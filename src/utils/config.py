import torch

model_path = 'data/best_model.pth'
dog_model_path = 'data/dogModel.pt'
dog_model_state_path = 'data/dogModelState.pt'
training_data_path = 'data/custom_dataset'

learning_rate = 1e-3
batch_size = 64
epochs = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

labels_map = {
    0: 'Clean',
    1: 'Messy',
}
