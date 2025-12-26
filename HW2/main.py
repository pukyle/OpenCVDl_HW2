import sys
import os
import cv2
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchsummary import summary
from torchvision import transforms
import matplotlib.pyplot as plt

# Import the model structure from your train.py
from train import get_resnet18_cifar10

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set window title (Replace with your ID and Name)
        self.setWindowTitle("Hw2_ResNet18_AN4126018_部政佑") 
        self.setGeometry(100, 100, 900, 600)

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Buttons ---
        btn_layout = QVBoxLayout()
        
        # 2.1 Load and Show Image
        self.btn_load = QPushButton("2.1 Load and Show Image")
        self.btn_load.clicked.connect(self.load_image)
        btn_layout.addWidget(self.btn_load)

        # 2.2 Show Model Structure
        self.btn_structure = QPushButton("2.2 Show Model Structure")
        self.btn_structure.clicked.connect(self.show_model_structure)
        btn_layout.addWidget(self.btn_structure)

        # 2.3 Show Acc and Loss
        self.btn_acc_loss = QPushButton("2.3 Show Acc and Loss")
        self.btn_acc_loss.clicked.connect(self.show_acc_loss)
        btn_layout.addWidget(self.btn_acc_loss)

        # 2.4 Inference
        self.btn_inference = QPushButton("2.4 Inference")
        self.btn_inference.clicked.connect(self.inference)
        btn_layout.addWidget(self.btn_inference)
        
        btn_layout.addStretch() 

        # Prediction Label
        self.label_prediction = QLabel("Predict: ")
        self.label_prediction.setStyleSheet("font-size: 16px; font-weight: bold;")
        btn_layout.addWidget(self.label_prediction)

        main_layout.addLayout(btn_layout, 1)

        # --- Right Panel: Image Display ---
        self.label_display = QLabel("")
        self.label_display.setAlignment(Qt.AlignCenter)
        self.label_display.setStyleSheet("border: 1px solid black; background-color: white;")
        main_layout.addWidget(self.label_display, 4)

        self.file_path = ""
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # ==========================================
    # 2.1 Load and Show Image
    # ==========================================
    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', './', 'Image Files (*.png *.jpg *.jpeg)')
        if filename:
            self.file_path = filename
            # Read image using OpenCV
            img = cv2.imread(filename)
            if img is None:
                print("Error: Could not read image.")
                return

            # Convert BGR to RGB for display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image to 32x32 (for display context, though usually we display larger)
            # But the slide says "Resize images to 32x32", so we follow it for display or processing.
            # Here we display the original or a scaled version for better visibility on GUI
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Scale pixmap to fit label
            scaled_pixmap = pixmap.scaled(self.label_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_display.setPixmap(scaled_pixmap)
            
            self.label_prediction.setText("Predict: ")

    # ==========================================
    # 2.2 Show Model Structure
    # ==========================================
    def show_model_structure(self):
        try:
            print("\n" + "="*30)
            print("2.2 Show ResNet-18 Structure")
            print("="*30)
            
            # Load the modified ResNet18
            model = get_resnet18_cifar10()
            print(model) 
            
            print("="*30 + "\n")
            
        except Exception as e:
            print(f"Error: {e}")

    # ==========================================
    # 2.3 Show Acc and Loss
    # ==========================================
    def show_acc_loss(self):
        img_path = "Loss&Acc.jpg"
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            cv2.imshow("Training Result (ResNet18)", img)
        else:
            QMessageBox.warning(self, "Error", "Result image not found! Did you run train.py?")

    # ==========================================
    # 2.4 Inference
    # ==========================================
    def inference(self):
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        try:
            # 1. Load Model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_resnet18_cifar10().to(device)
            
            model_path = "model/weight.pth"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                QMessageBox.critical(self, "Error", f"Model not found: {model_path}")
                return
            
            model.eval()

            # 2. Preprocessing
            img_cv = cv2.imread(self.file_path)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL for transforms
            img_pil = transforms.ToPILImage()(img_rgb)
            
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # 3. Inference
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                max_prob, predicted_id = torch.max(probabilities, 1)
                
            prob_val = max_prob.item()
            pred_index = predicted_id.item()
            
            # 4. Threshold Logic (For "Others")
            threshold = 0.5 # Adjust this value if needed
            
            if prob_val < threshold:
                result_text = "Others"
                confidence_str = ""
            else:
                result_text = self.classes[pred_index]
                confidence_str = f"({prob_val:.4f})"

            # 5. Update GUI
            self.label_prediction.setText(f"Predict: {result_text} {confidence_str}")

            # 6. Show Histogram
            probs_np = probabilities.cpu().numpy()[0]
            
            plt.figure(figsize=(10, 5))
            bars = plt.bar(self.classes, probs_np, color='blue')
            plt.title(f"Inference Result - Predicted: {result_text}")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45)
            
            # Add text labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", f"Inference failed: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())