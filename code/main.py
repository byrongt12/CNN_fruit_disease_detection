from model import *
from dataset import *

print('Getting data...')
X, Y, key_list, val_list = getData()

print('Splitting data...')
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

train_ds = FruitDataset(X_train['new_path'], X_train['new_bb'], y_train, transforms=True)
valid_ds = FruitDataset(X_val['new_path'], X_val['new_bb'], y_val)

print('Creating batches...')
batch_size = 16
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)

# --------------------AlexNet-------------------------------------------------#
print('Building model...')
model = AlexNet().cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.006, momentum=0.005)  # vs Adam  vs momentum=0.005
#
print('Training...')
alex_train, alex_val_acc = train_epocs(model, optimizer, train_dl, valid_dl, epochs=20)
print('Done!')

# --------------------VGG11-------------------------------------------------#
print('Building model...')
model = VGG11().cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.006, momentum=0.005)  # vs Adam  vs momentum=0.005

print('Training...')
vgg_train, vgg_val_acc = train_epocs(model, optimizer, train_dl, valid_dl, epochs=20)
print('Done!')

# --------------------ResNet18-------------------------------------------------#
print('Building model...')
model = ResNet18().cuda()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(parameters, lr=0.006, momentum=0.005)  # vs Adam  vs momentum=0.005

print('Training...')
res_train, res_val_acc = train_epocs(model, optimizer, train_dl, valid_dl, epochs=20)
print('Done!')


# --------------------Plot-------------------------------------------------#

plt.plot(alex_train, label='AlexNet')
plt.plot(vgg_train, label='VGG-11')
plt.plot(res_train, label='ResNet18')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(alex_val_acc, label='AlexNet')
plt.plot(vgg_val_acc, label='VGG-11')
plt.plot(res_val_acc, label='ResNet18')
plt.title('Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()









