import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import argparse

def evaluate(model, dataloader, device):
    model.eval()  # set the model to evaluation mode
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():  # disable gradient computation to save memory
        for data in tqdm(dataloader, desc='Evaluating'):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted += predicted.cpu().tolist()
            all_labels += labels.cpu().tolist()
    accuracy = 100 * correct / total
    print('Accuracy of the network on the test images: %d %%' % accuracy)
    confusion = confusion_matrix(all_labels, all_predicted)
    print('Confusion matrix:')
    print(confusion)
    return accuracy, confusion


parser = argparse.ArgumentParser(description='Evaluate a Swin Transformer model on the mosq_data dataset')
parser.add_argument('--data-dir', type=str, default='./mosq_data',
                    help='path to the root directory of the dataset')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for evaluation (default: 32)')
parser.add_argument('--num-workers', type=int, default=4,
                    help='number of workers for data loading (default: 4)')
parser.add_argument('--model-file', type=str, default='swin_tiny_patch4_window7_224.pth',
                    help='path to the model file (default: swin_tiny_patch4_window7_224.pth)')
args = parser.parse_args()

testdir = args.data_dir + '/val'
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder(testdir, test_transforms)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args.model_file)
model.to(device)

accuracy, confusion = evaluate(model, testloader, device)

classes = test_dataset.classes
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion)
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(classes)):
    for j in range(len(classes)):
        ax.text(j, i, confusion[i, j], ha="center", va="center", color="w")
