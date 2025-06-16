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

# --- SAM Optimizer Definition (Paste the SAM class definition from above here) ---
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        # Ensure base_optimizer is a type, not an instance
        if not isinstance(base_optimizer, type):
            raise ValueError("base_optimizer must be a class type, e.g., torch.optim.SGD")
        # Instantiate the base_optimizer with the parameters and kwargs
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                # Calculate ascent direction
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                # Restore original weights before base optimizer step
                p.data = self.state[p]["old_p"]
        # The gradients computed in the second forward/backward pass are already set
        self.base_optimizer.step()  # do the actual update step using gradients computed on perturbed weights
        if zero_grad: self.zero_grad()

    # step() function using closure is not the standard way SAM is used in loops
    # It's more common to manually call first_step and second_step in the training loop

    def _grad_norm(self):
        # Tolenrant to device placement
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
             for p in group["params"]:
                 if p.grad is not None:
                     # Use p.grad.detach() to avoid modifying gradients during norm calculation if adaptive=True
                     param_grad = p.grad.detach()
                     param_norm = ((torch.abs(p.detach()) if group["adaptive"] else 1.0) * param_grad).norm(p=2)
                     norms.append(param_norm.to(shared_device))
        if not norms: # Handle case where no parameters have gradients
            return torch.tensor(0.0, device=shared_device)
        # Stack norms before calculating the final norm
        total_norm = torch.norm(torch.stack(norms), p=2)
        return total_norm

    # Overwrite zero_grad to also zero base_optimizer's gradients
    def zero_grad(self, set_to_none: bool = False):
        super(SAM, self).zero_grad(set_to_none=set_to_none)
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    # Need to handle state dict loading/saving properly for both SAM and base_optimizer
    def state_dict(self):
        # Combine SAM state and base optimizer state
        sam_state = super(SAM, self).state_dict()
        base_state = self.base_optimizer.state_dict()
        return {"sam_state": sam_state, "base_optimizer_state": base_state}

    def load_state_dict(self, state_dict):
        # Load states separately
        sam_state = state_dict["sam_state"]
        base_state = state_dict["base_optimizer_state"]
        super(SAM, self).load_state_dict(sam_state)
        self.base_optimizer.load_state_dict(base_state)
        # Ensure param_groups are synchronized after loading state
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)



# -------------------- 参数设置 --------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--ft_lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--ft_epochs', default=90, type=int, help='number of epochs to train')  #150 resnet20
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--arch', default='r18', type=str, help='model name')
parser.add_argument('--sam_rho', default=0.01, type=float, help='sam_rho')
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

# -------------------- 损失函数 和 SAM 优化器 --------------------
criterion = nn.CrossEntropyLoss()

# Define the base optimizer (SGD) with the fine-tuning learning rate
# SAM will use this base optimizer internally
base_optimizer = torch.optim.SGD  # Pass the class, not an instance
optimizer = SAM(net.parameters(), base_optimizer, rho=args.sam_rho, adaptive=False, # Set adaptive=True if needed
                lr=args.ft_lr, momentum=0, weight_decay=5e-4)

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


# -------------------- SAM Fine-tuning 训练函数 --------------------
def train_sam(epoch):
    print(f'\n--- SAM Fine-tuning Epoch: {epoch+1}/{args.ft_epochs} ---')
    net.train() # Set model to training mode
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    current_lr = optimizer.param_groups[0]['lr'] # Get LR (should be fixed)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # --- SAM specific steps ---
        # 1. First forward/backward pass to compute gradients on original weights
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.first_step(zero_grad=True) # Perturbs weights and zeros grads

        # 2. Second forward/backward pass on perturbed weights
        # Ensure gradients are enabled for the second pass's backward
        with torch.enable_grad():
             criterion(net(inputs), targets).backward()
        optimizer.second_step(zero_grad=True) # Restores original weights and performs update step
        # --- End SAM steps ---

        # Accumulate loss (using loss from the first step for reporting)
        train_loss += loss.item()
        _, predicted = outputs.max(1) # Use predictions from the first step
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_time = time.time() - start_time
    epoch_loss = train_loss / len(trainloader)
    epoch_acc = 100. * correct / total
    print(f'Train | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.3f}% ({correct}/{total}) | LR: {current_lr:.5f} | Time: {epoch_time:.2f}s')
    return epoch_loss, epoch_acc

# -------------------- Fine-tuning 主循环 --------------------
print("==> Starting SAM Fine-tuning...")
finetuning_start_time = time.time()

for epoch in range(args.ft_epochs):
    # 1. Fine-tune with SAM for one epoch
    train_loss, train_acc = train_sam(epoch)

    # 2. Evaluate on the test set after this epoch
    test_loss, test_acc = evaluate(testloader, "Test", net) # Evaluate the updated model

    # No scheduler.step() needed as LR is fixed

total_finetuning_time = time.time() - finetuning_start_time
print(f"\n==> Finished SAM Fine-tuning in {total_finetuning_time:.2f} seconds.")


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