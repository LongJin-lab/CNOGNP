import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
import argparse
import random
# from timm.models.vision_transformer import VisionTransformer
import torchvision.models as models




# -------------------- 参数设置 --------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--ft_lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--ft_epochs', default=90, type=int, help='number of epochs to train')  #150 resnet20
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--arch', default='r18', type=str, help='model name')
parser.add_argument('--load_path', default='./r18_imagenet_final.pth', type=str, help='path to load pretrain model')
parser.add_argument('--save_path', default='./r18_imagenet_sgd_ft.pth', type=str, help='path to save final model')
parser.add_argument('--data_path', default='/scratch/lab415/datasets/imagenet/', type=str, help='path to dataset')
# --- Use parse_known_args() for Jupyter compatibility ---
args, unknown = parser.parse_known_args()
# -------------------------------------------------------



# -------------------- 设备配置 --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# -------------------- 数据准备 --------------------
print('==> Preparing data..')
# ImageNet normalization
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224), # Standard for ImageNet
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

transform_test = transforms.Compose([
    transforms.Resize(256), # Standard for ImageNet
    transforms.CenterCrop(224), # Standard for ImageNet
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# Ensure your ImageNet data is structured as:
# args.data_path/train/class1/img1.jpeg
# args.data_path/train/class2/img2.jpeg
# ...
# args.data_path/val/class1/img3.jpeg
# args.data_path/val/class2/img4.jpeg
# ...
train_dir = os.path.join(args.data_path, 'train')
val_dir = os.path.join(args.data_path, 'val') # Or 'validation' or 'test' depending on your ImageNet structure

if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
    raise FileNotFoundError(f"ImageNet train or val directory not found at {args.data_path}. Please check the path and structure.")

trainset = torchvision.datasets.ImageFolder(
    root=train_dir, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True) # Increased num_workers, added pin_memory

testset = torchvision.datasets.ImageFolder(
    root=val_dir, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True) # Increased num_workers, added pin_memory

# -------------------- 模型加载 --------------------
print('==> Building and loading pre-trained model..')
# net = ResNet20()
if args.arch == 'r18':
    net = models.resnet18(pretrained=False, num_classes=1000)
if args.arch == 'r34':
    net = models.resnet34(pretrained=False, num_classes=1000)
if args.arch == 'r50':
    net = models.resnet50(pretrained=False, num_classes=1000)

if os.path.exists(args.load_path):
    try:
        print(f"Loading checkpoint from '{args.load_path}'")
        checkpoint = torch.load(args.load_path, map_location=device)
        # Adjust based on how the model was saved (state_dict vs full model)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
             net.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict) and not ('state_dict' in checkpoint): # Directly saved state_dict
             net.load_state_dict(checkpoint)
        else: # Saved the entire model object
             net = checkpoint
        print("Pre-trained model loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Proceeding with initialized ResNet20 (training from scratch).")
        # Apply weight initialization if not loading weights
        net.apply(_weights_init)
else:
    print(f"Checkpoint file not found at '{args.load_path}'.")
    print("Proceeding with initialized ResNet20 (training from scratch).")
    # Apply weight initialization if not loading weights
    net.apply(_weights_init)

net = net.to(device)
# Optional DataParallel
# if device == 'cuda' and torch.cuda.device_count() > 1:
#     print(f"Let's use {torch.cuda.device_count()} GPUs!")
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True # Good if input sizes don't change

# -------------------- 损失函数 和 sgd 优化器 --------------------
criterion = nn.CrossEntropyLoss()


# --- 添加 SGD 优化器定义 ---
optimizer = optim.SGD(net.parameters(),       # 使用模型的参数
                      lr=args.ft_lr,          # 使用 fine-tuning 的学习率
                      momentum=0,           # 动量
                      weight_decay=5e-4)      # 权重衰减 (如果需要的话)
# ---------------------------


# No learning rate scheduler needed for fixed LR fine-tuning

# -------------------- 评估函数 (Same as before) --------------------
def evaluate(loader, set_name="Test", model=net): # Pass model explicitly
    model.eval() # Set model to evaluation mode
    eval_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad(): # Disable gradient calculation
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs) # Use the passed model
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_time = time.time() - start_time
    avg_loss = eval_loss / len(loader)
    accuracy = 100. * correct / total
    print(f'{set_name.ljust(5)} | Loss: {avg_loss:.4f} | Acc: {accuracy:.3f}% ({correct}/{total}) | Time: {epoch_time:.2f}s')
    return avg_loss, accuracy


# -------------------- 初始评估 (评估加载的模型) --------------------
print("\n==> Evaluating loaded model before fine-tuning...")
initial_train_loss, initial_train_acc = evaluate(trainloader, "Train", net)
initial_test_loss, initial_test_acc = evaluate(testloader, "Test", net)
print("--------------------------------------------------")


def train_sgd(epoch): # <--- 修改函数名 (或者你喜欢的其他名字)
    print(f'\n--- SGD Fine-tuning Epoch: {epoch+1}/{args.ft_epochs} ---') # 更新打印信息
    net.train() # Set model to training mode
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr'] # Get LR (should be fixed)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # --- 标准 SGD 步骤 ---
        optimizer.zero_grad()       # <--- 添加：梯度清零
        outputs = net(inputs)       # <--- 前向传播 (只需要一次)
        loss = criterion(outputs, targets) # <--- 计算损失
        loss.backward()             # <--- 反向传播 (只需要一次)
        optimizer.step()            # <--- 添加：更新权重
        # --- 结束标准 SGD 步骤 ---


        # --- 删除 SAM 相关步骤 ---
        # # 1. First forward/backward pass ...
        # loss.backward()
        # optimizer.first_step(zero_grad=True) # <-- 删除

        # # 2. Second forward/backward pass ...
        # with torch.enable_grad():
        #      criterion(net(inputs), targets).backward() # <-- 删除
        # optimizer.second_step(zero_grad=True) # <-- 删除
        # --- 结束删除 SAM 步骤 ---


        # (损失和准确率的累积部分保持不变)
        train_loss += loss.item()
        _, predicted = outputs.max(1) # Use predictions from the forward pass
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # (打印 epoch 结果的部分保持不变)
    epoch_time = time.time() - start_time
    epoch_loss = train_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    print(f'Train | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.3f}% ({correct}/{total}) | LR: {current_lr:.5f} | Time: {epoch_time:.2f}s')
    return epoch_loss, epoch_acc

# --- 修改 "Fine-tuning 主循环" ---
print("==> Starting SGD Fine-tuning...") # 更新打印信息
finetuning_start_time = time.time()

for epoch in range(args.ft_epochs):
    # 1. Fine-tune with SGD for one epoch
    # train_loss, train_acc = train_sam(epoch) # 原来的调用
    train_loss, train_acc = train_sgd(epoch) # <--- 使用新的函数名

    # (评估部分保持不变)
    test_loss, test_acc = evaluate(testloader, "Test", net)

total_finetuning_time = time.time() - finetuning_start_time
print(f"\n==> Finished SGD Fine-tuning in {total_finetuning_time:.2f} seconds.") # 更新打印信息


# -------------------- 保存最终 Fine-tuned 模型 --------------------
print(f'==> Saving final fine-tuned model to {args.save_path}')
save_dir = os.path.dirname(args.save_path)
if save_dir and not os.path.exists(save_dir): # Check if save_dir is not empty
    os.makedirs(save_dir)
# Save only the model state_dict is usually preferred
torch.save(net.state_dict(), args.save_path)
# If you need to save optimizer state as well (e.g., to resume SAM training):
# torch.save({
#     'epoch': args.ft_epochs, # Or the actual last epoch number
#     'model_state_dict': net.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(), # Save SAM state
# }, args.save_path)
print("Final fine-tuned model saved.")

# -------------------- 评估最终 Fine-tuned 模型 --------------------
print("\n==> Evaluating final fine-tuned model (after {} epochs)...".format(args.ft_epochs))
print("--- Final Training Set Evaluation ---")
final_train_loss, final_train_acc = evaluate(trainloader, "Train", net)
print("--- Final Test Set Evaluation ---")
final_test_loss, final_test_acc = evaluate(testloader, "Test", net)

print("\n===== Initial Model Performance =====")
print(f"Initial Training Loss: {initial_train_loss:.4f}")
print(f"Initial Training Acc:  {initial_train_acc:.3f}%")
print(f"Initial Test Loss:     {initial_test_loss:.4f}")
print(f"Initial Test Acc:      {initial_test_acc:.3f}%")
print("====================================")

print("\n===== Final Fine-tuned Model Performance =====")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Training Acc:  {final_train_acc:.3f}%")
print(f"Final Test Loss:     {final_test_loss:.4f}")
print(f"Final Test Acc:      {final_test_acc:.3f}%")
print("==========================================")