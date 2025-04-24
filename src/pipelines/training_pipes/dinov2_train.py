import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        # teacher and student: list of multi-crops outputs
        student_logits = [s / self.student_temp for s in student_output]
        teacher_probs = [(t - self.center) / self.teacher_temp for t in teacher_output]
        teacher_probs = [p.softmax(dim=-1).detach() for p in teacher_probs]

        total_loss = 0
        n_loss_terms = 0
        for t in teacher_probs:
            for s in student_logits:
                loss = torch.sum(-t * nn.functional.log_softmax(s, dim=-1), dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1
        total_loss /= n_loss_terms

        # update center
        batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        return total_loss


def get_dino_augmentations(img_size=224):
    # Multi-crop: 2 global + 6 local crops
    global_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.GaussianBlur(kernel_size=23),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size // 2, scale=(0.05, 0.4)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    def multi_crop_transform(img):
        crops = []
        # 2 global
        for _ in range(2):
            crops.append(global_transform(img))
        # 6 local
        for _ in range(6):
            crops.append(local_transform(img))
        return crops
    return multi_crop_transform


def train_dino(
    data_dir,
    arch='vit_base_patch16_224',
    out_dim=65536,
    epochs=100,
    batch_size=64,
    lr=0.0005,
    weight_decay=0.04,
    momentum_teacher=0.996,
    device='cuda'
):
    # Dataset & DataLoader
    transform = get_dino_augmentations()
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)

    # Student & Teacher networks
    student = timm.create_model(arch, pretrained=False, num_classes=out_dim)
    teacher = timm.create_model(arch, pretrained=False, num_classes=out_dim)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    student.to(device)
    teacher.to(device)

    # Loss, optimizer
    criterion = DINOLoss(out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for images, _ in loader:
            # images: list of lists of crops
            crops = list(zip(*images))
            views = [torch.stack(v).to(device) for v in images]
            # forward
            student_outputs = [student(v) for v in views]
            with torch.no_grad():
                teacher_outputs = [teacher(v) for v in views[:2]]  # only global views for teacher

            loss = criterion(student_outputs, teacher_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update teacher via momentum
            m = momentum_teacher * (1 - (1 - momentum_teacher) * (epoch / epochs))
            for param_s, param_t in zip(student.parameters(), teacher.parameters()):
                param_t.data = param_t.data * m + param_s.data * (1 - m)

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")

    torch.save(student.state_dict(), 'dino_student.pth')
    torch.save(teacher.state_dict(), 'dino_teacher.pth')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DINOv2 Self-Supervised Training')
    parser.add_argument('data_dir', type=str, help='Path to ImageNet or image folder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    train_dino(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
