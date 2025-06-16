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
import copy # For deep copying models and states
import math # For infinity
# from timm.models.vision_transformer import VisionTransformer
import torchvision.models as models

# -------------------- CNOGNP & Script Parameters --------------------
parser = argparse.ArgumentParser(description='PyTorch ImageNet Fine-tuning with CNOGNP on Weights')
# --- CNOGNP Specific ---
parser.add_argument('--lambda_gnp', default=0.1, type=float, help='Gradient norm penalty coefficient (λ)')
# --- Parameters from CNO (Mapping from image) ---
parser.add_argument('--num_particles', default=2, type=int, help='Number of particles (N)')
parser.add_argument('--cnognp_epochs', default=10, type=int, help='Number of CNOGNP iterations (κ_max)') # Renamed epoch arg
parser.add_argument('--w', default=1, type=float, help='Inertia weight (ω)')
parser.add_argument('--c1', default=0.00001, type=float, help='Cognitive learning factor (c1)')
parser.add_argument('--c2', default=0.00001, type=float, help='Social learning factor (c2)')
parser.add_argument('--eta', default=0.001, type=float, help='Scale factor / Learning rate for inner SGD step (η)')
# --- Parameters consistent with others ---
parser.add_argument('--initial_noise_level', default=0.0001, type=float, help='Std deviation of noise added to initial particle weights')
parser.add_argument('--inner_sgd_momentum', default=0, type=float, help='Momentum for the inner SGD step')
parser.add_argument('--inner_sgd_wd', default=5e-4, type=float, help='Weight decay for the inner SGD step')
parser.add_argument('--arch', default='vit-t', type=str)
parser.add_argument('--load_path', default='./vit-t_cifar100_final_290.pth', type=str, help='Path to load the pre-trained model')
parser.add_argument('--save_path', default='./vit-t_cifar100_cnognp_ft_290.pth', type=str, help='Path to save the best model found by CNOGNP')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for SGD training and evaluation')
parser.add_argument('--data_path', default='/scratch/lab415/datasets/imagenet/', type=str, help='Path to dataset')
# --- Use parse_known_args() for Jupyter compatibility ---
args, unknown = parser.parse_known_args()

print(args)



# -------------------- Device Configuration --------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    cudnn.benchmark = True

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

# -------------------- Loss Function --------------------
criterion = nn.CrossEntropyLoss()

# -------------------- Helper Functions --------------------

# Standard evaluation function (for final results)
def evaluate(loader, model, set_name="Test"):
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = eval_loss / len(loader)
    accuracy = 100. * correct / total
    print(f'{set_name.ljust(5)} Eval | Loss: {avg_loss:.4f} | Acc: {accuracy:.3f}% ({correct}/{total})')
    return avg_loss, accuracy

# Function for CNO Line 7: Train ONE SGD epoch and return the *new state* and the loss
def train_one_sgd_epoch_and_get_state(initial_state_dict, train_loader, criterion, device, lr, momentum, weight_decay, model_name):
    if args.arch == 'r18':
        model = models.resnet18(pretrained=False, num_classes=1000).to(device)
    if args.arch == 'r34':
        model = models.resnet34(pretrained=False, num_classes=1000).to(device)
    if args.arch == 'r50':
        model = models.resnet50(pretrained=False, num_classes=1000).to(device)
    model.load_state_dict(copy.deepcopy(initial_state_dict))
    model.train()
    train_loss = 0
    total_grad_norm = 0 # Accumulate norm per batch
    num_batches = 0
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    start_time = time.time()
    print(f"      Starting 1-epoch SGD (CNOGNP Line 7, η={lr})... ", end="")
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        batch_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                batch_grad_norm += param_norm.item() ** 2
        batch_grad_norm = batch_grad_norm ** 0.5
        total_grad_norm += batch_grad_norm
    avg_loss = train_loss / len(train_loader)
    avg_grad_norm = total_grad_norm / len(train_loader) # Average the norm calculated per batch
    epoch_time = time.time() - start_time
    print(f"Done. Avg Loss during SGD: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    return copy.deepcopy(model.state_dict()), avg_loss, avg_grad_norm

# Function for CNOGNP Line 10: Evaluate fitness (Loss + λ*||g||₂) on train set
def evaluate_fitness_loss_and_grad_norm(avg_loss, avg_grad_norm, lambda_gnp):
    # CNOGNP Fitness
    fitness = avg_loss + lambda_gnp * avg_grad_norm
    print(f"Done. Fitness: {fitness:.4f} (Avg Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f})")
    return fitness

# --- Other helper functions (add_noise, initialize_velocity, update_cno_velocity, update_particle_position) ---
# --- remain EXACTLY the same as in the CNO implementation ---
def add_noise_to_model(model, noise_level, device):
    print('add noise')
    with torch.no_grad():
        for param in model.parameters(): param.add_(torch.randn_like(param) * noise_level)
    return model
def initialize_velocity(model):
    velocity = {}
    with torch.no_grad():
      for name, param in model.named_parameters():
          if param.requires_grad: velocity[name] = torch.zeros_like(param)
    return velocity
def update_cno_velocity(velocity_dict, z_bar_i_state, pbest_state, gbest_state, c1, c2, device):
    with torch.no_grad():
        for name, param_vel in velocity_dict.items():
            if name not in z_i_state: continue
            r1 = random.random()
            r2 = random.random()

            # Ensure all tensors are on the correct device
            z_bar_i_param = z_bar_i_state[name].to(device)
            pbest_param = pbest_state[name].to(device)
            gbest_param = gbest_state[name].to(device)
            current_vel = param_vel.to(device)

            # CNO Velocity Update (Line 8)
            cognitive_term = c1 * r1 * (pbest_param - z_bar_i_param)
            social_term = c2 * r2 * (gbest_param - z_bar_i_param)

            new_vel = cognitive_term + social_term
            velocity_dict[name].copy_(new_vel) # Update velocity in place
def update_particle_position(model_to_update, z_bar_i_state, velocity_dict, device):
    new_state = copy.deepcopy(z_bar_i_state)
    with torch.no_grad():
        for name, param in new_state.items():
             if name in velocity_dict: param.add_(velocity_dict[name].to(device))
    model_to_update.load_state_dict(new_state)

# -------------------- Load Pre-trained Model --------------------
print('==> Loading pre-trained model...')
# initial_model = ResNet20().to(device)
if args.arch == 'r18':
    initial_model = models.resnet18(pretrained=False, num_classes=1000).to(device)
if args.arch == 'r34':
    initial_model = models.resnet34(pretrained=False, num_classes=1000).to(device)
if args.arch == 'r50':
    initial_model = models.resnet50(pretrained=False, num_classes=1000).to(device)

# (Loading logic remains the same as CNO)
if os.path.exists(args.load_path):
    try:
        checkpoint = torch.load(args.load_path, map_location=device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint: initial_model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict): initial_model.load_state_dict(checkpoint)
        else: initial_model = checkpoint
        print(f"Loaded pre-trained weights from '{args.load_path}'")
    except Exception as e: print(f"Error loading checkpoint: {e}. Exiting."); exit()
else: print(f"Pre-trained model file not found at '{args.load_path}'. Exiting."); exit()

# Evaluate the loaded model once (using standard evaluate)
print("\n==> Evaluating loaded pre-trained model:")
initial_test_loss, initial_test_acc = evaluate(testloader, initial_model, "Test")
initial_train_loss, initial_train_acc = evaluate(trainloader, initial_model, "Train")


# -------------------- CNOGNP Initialization --------------------
print(f"\n==> Initializing {args.num_particles} CNOGNP particles...")
particles = []
gbest_state_dict = None
gbest_fitness = float('inf') # Use math.inf for clarity

for i in range(args.num_particles):
    print(f"  Initializing particle {i+1}/{args.num_particles}...")
    particle_model = copy.deepcopy(initial_model).to(device)
    if i > 0 or args.num_particles == 1:
         particle_model = add_noise_to_model(particle_model, args.initial_noise_level, device)
    velocity = initialize_velocity(particle_model)
    pbest_state_dict = copy.deepcopy(particle_model.state_dict())
    pbest_fitness = float('inf')
    particles.append({
        'id': i, 'model': particle_model, 'velocity': velocity,
        'pbest_state_dict': pbest_state_dict, 'pbest_fitness': pbest_fitness,
        'current_fitness': float('inf')
    })

# --- Initial Fitness Evaluation (using CNOGNP fitness function) ---
print("\n==> Performing initial fitness evaluation (using CNOGNP fitness)...")
current_epoch_best_fitness = float('inf')
current_epoch_best_particle_idx = -1
for i, particle in enumerate(particles):
    print(f"  Evaluating initial fitness for particle {i+1}/{args.num_particles}:")
    # Use the NEW fitness function
    fitness = evaluate_fitness_loss_and_grad_norm(
        999, 999, args.lambda_gnp
    )
    particle['current_fitness'] = fitness
    particle['pbest_fitness'] = fitness # Initial pbest fitness

    if fitness < current_epoch_best_fitness:
        current_epoch_best_fitness = fitness
        current_epoch_best_particle_idx = i

# # Update global best (gbest) based on the initial evaluation
# if current_epoch_best_particle_idx != -1 :
#      initial_best_particle = particles[current_epoch_best_particle_idx]
#      print(f"\nInitial Global Best Fitness (particle {current_epoch_best_particle_idx+1}): {current_epoch_best_fitness:.4f}")
#      gbest_fitness = current_epoch_best_fitness
#      gbest_state_dict = copy.deepcopy(initial_best_particle['pbest_state_dict'])
# else:
#      print("\nWarning: No valid fitness found in initial evaluation.")
#      # Fallback: use the originally loaded model as gbest
#      gbest_fitness = evaluate_fitness_loss_and_grad_norm(avg_loss, avg_grad_norm, args.lambda_gnp)
#      gbest_state_dict = copy.deepcopy(initial_model.state_dict())
#      print(f"Using loaded model as initial gbest (Fitness: {gbest_fitness:.4f})")


# -------------------- CNOGNP Main Loop --------------------
print(f"\n==> Starting CNOGNP Fine-tuning for {args.cnognp_epochs} epochs...")
cnognp_start_time = time.time()

# Use args.cnognp_epochs here
for cnognp_epoch in range(args.cnognp_epochs):
    print(f"\n--- CNOGNP Epoch {cnognp_epoch + 1}/{args.cnognp_epochs} ---")
    epoch_start_time = time.time()
    current_epoch_best_fitness = float('inf')
    current_epoch_best_particle_idx = -1

    for i, particle in enumerate(particles):
        print(f"  Processing Particle {i+1}/{args.num_particles}:")
        z_i_state = copy.deepcopy(particle['model'].state_dict())



        # --- CNOGNP Line 7: Perform SGD step ---
        z_bar_i_state, sgd_run_loss, avg_grad_norm = train_one_sgd_epoch_and_get_state(
            z_i_state, trainloader, criterion, device,
            args.eta, args.inner_sgd_momentum, args.inner_sgd_wd, args.arch
        )
        new_state = copy.deepcopy(z_bar_i_state)
        particle['model'].load_state_dict(new_state)
        current_fitness = evaluate_fitness_loss_and_grad_norm(
            sgd_run_loss, avg_grad_norm, args.lambda_gnp
        )
        particle['current_fitness'] = current_fitness

        # --- CNOGNP Lines 11-13: Update PBest ---
        if current_fitness < particle['pbest_fitness']:
            print(f"      New pbest for particle {i+1}: {current_fitness:.4f} (was {particle['pbest_fitness']:.4f})")
            particle['pbest_fitness'] = current_fitness
            particle['pbest_state_dict'] = copy.deepcopy(particle['model'].state_dict())
        else:
            print(f"      Fitness {current_fitness:.4f} not better than pbest {particle['pbest_fitness']:.4f}")

        if current_fitness < current_epoch_best_fitness:
             current_epoch_best_fitness = current_fitness
             current_epoch_best_particle_idx = i

        # --- CNOGNP Lines 14-16: Update GBest ---
        print("  Updating gbest...")
        if current_epoch_best_particle_idx != -1 and current_epoch_best_fitness < gbest_fitness:
            print(f"    New Global Best! Fitness: {current_epoch_best_fitness:.4f} (was {gbest_fitness:.4f}) from particle {current_epoch_best_particle_idx+1}'s pbest")
            gbest_fitness = current_epoch_best_fitness
            gbest_state_dict = copy.deepcopy(particles[current_epoch_best_particle_idx]['pbest_state_dict'])
        else:
            print(f"    No new gbest found this epoch. Best this epoch: {current_epoch_best_fitness:.4f}, Current gbest: {gbest_fitness:.4f}")

        # --- CNOGNP Line 8: Update Velocity ---
        update_cno_velocity(
            particle['velocity'], z_bar_i_state,
            particle['pbest_state_dict'], gbest_state_dict,
            args.c1, args.c2, device
        )

        # --- CNOGNP Line 9: Update Position ---
        update_particle_position(
             particle['model'], z_bar_i_state, particle['velocity'], device
        )

        # --- CNOGNP Line 10: Evaluate Fitness (Loss + λ*||g||₂) ---
        # Use the NEW fitness function
    epoch_time = time.time() - epoch_start_time
    print(f"--- CNOGNP Epoch {cnognp_epoch + 1} finished. Time: {epoch_time:.2f}s ---")
        

    


total_cnognp_time = time.time() - cnognp_start_time
print(f"\n==> Finished CNOGNP Fine-tuning in {total_cnognp_time:.2f} seconds ({total_cnognp_time/3600:.2f} hours).")


# -------------------- Final Evaluation --------------------
print("\n==> Evaluating the best model found by CNOGNP...")
# final_best_model = ResNet20().to(device)
if args.arch == 'r18':
    final_best_model = models.resnet18(pretrained=False, num_classes=1000).to(device)
if args.arch == 'r34':
    final_best_model = models.resnet34(pretrained=False, num_classes=1000).to(device)
if args.arch == 'r50':
    final_best_model = models.resnet50(pretrained=False, num_classes=1000).to(device)


if gbest_state_dict is not None:
    final_best_model.load_state_dict(gbest_state_dict)
else:
    print("Error: Global best state dictionary was not set. Cannot evaluate.")
    exit()

# Use standard evaluate for final Loss/Acc comparison
print("--- Final Training Set Evaluation (using standard evaluate) ---")
final_train_loss, final_train_acc = evaluate(trainloader, final_best_model, "Train")
print("--- Final Test Set Evaluation (using standard evaluate) ---")
final_test_loss, final_test_acc = evaluate(testloader, final_best_model, "Test")
particle_eval_results = {} # Optional: dictionary to store results per particle
for i, particle in enumerate(particles):
    print(f"\n--- Evaluating Final State of Particle {i+1}/{args.num_particles} ---")
    # The model in particle['model'] holds the final state after all updates
    particle_model = particle['model']

    # Use standard evaluate for training set
    print(f"Particle {i+1} Train Set Evaluation:")
    train_loss, train_acc = evaluate(trainloader, particle_model, f"P{i+1} Train")

    # Use standard evaluate for test set
    print(f"Particle {i+1} Test  Set Evaluation:") # Added padding for alignment
    test_loss, test_acc = evaluate(testloader, particle_model, f"P{i+1} Test ")

    particle_eval_results[f'particle_{i+1}'] = {'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc}

print("\n===== Initial Model Performance =====")
print(f"Initial Training Loss: {initial_train_loss:.4f}")
print(f"Initial Training Acc:  {initial_train_acc:.3f}%")
print(f"Initial Test Loss:     {initial_test_loss:.4f}")
print(f"Initial Test Acc:      {initial_test_acc:.3f}%")
print("====================================")

print("\n===== CNOGNP Fine-tuned Model Performance =====")
print(f"Achieved Global Best Fitness (Min Loss+λ||g||₂ during CNOGNP): {gbest_fitness:.4f}")
print(f"Final Eval Training Loss: {final_train_loss:.4f}") # Standard Loss
print(f"Final Eval Training Acc:  {final_train_acc:.3f}%")
print(f"Final Eval Test Loss:     {final_test_loss:.4f}") # Standard Loss
print(f"Final Eval Test Acc:      {final_test_acc:.3f}%")
print("===========================================")

# -------------------- Save Final Model --------------------
print(f'==> Saving final CNOGNP best model to {args.save_path}')
save_dir = os.path.dirname(args.save_path)
if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir)
if gbest_state_dict is not None:
    torch.save(gbest_state_dict, args.save_path)
    print("Final best model saved.")
else:
    print("Error: Global best state dictionary was not set. Model not saved.")