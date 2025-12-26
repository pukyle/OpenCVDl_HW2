import sys
import os
import cv2
import torch
import torch.nn.functional as F
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from torchsummary import summary
from torchvision import transforms
import matplotlib.pyplot as plt

from train import LeNet5

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set window title format: Hw2_StudentId_Name
        self.setWindowTitle("Hw2_AN4126018_部政佑") 
        self.setGeometry(100, 100, 900, 600)

        # Main Layout (Left: Buttons, Right: Image)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- Left Panel: Buttons & Prediction Label ---
        btn_layout = QVBoxLayout()
        
        # 1. Load Image
        self.btn_load_image = QPushButton("Load Image")
        self.btn_load_image.clicked.connect(self.load_image)
        btn_layout.addWidget(self.btn_load_image)

        # 2. Show Architecture
        self.btn_show_structure = QPushButton("1.1 Show Architecture")
        self.btn_show_structure.clicked.connect(self.show_model_structure)
        btn_layout.addWidget(self.btn_show_structure)

        # 3. Show Accuracy & Loss
        self.btn_show_acc_loss = QPushButton("1.2 Show Acc Loss")
        self.btn_show_acc_loss.clicked.connect(self.show_acc_loss)
        btn_layout.addWidget(self.btn_show_acc_loss)

        # 4. Predict
        self.btn_predict = QPushButton("1.3 Predict")
        self.btn_predict.clicked.connect(self.predict_image)
        btn_layout.addWidget(self.btn_predict)
        
        # Add stretch to push the prediction label to the bottom
        btn_layout.addStretch() 

        # Prediction Result Label (Bottom of Left Panel)
        self.label_prediction = QLabel("Predict:")
        self.label_prediction.setStyleSheet("font-size: 18px; font-weight: bold;")
        btn_layout.addWidget(self.label_prediction)

        # Add Left Panel to Main Layout (Ratio 1)
        main_layout.addLayout(btn_layout, 1)

        # --- Right Panel: Image Display ---
        self.label_display = QLabel("")
        self.label_display.setAlignment(Qt.AlignCenter)
        self.label_display.setStyleSheet("border: 1px solid black; background-color: white;")
        
        # Add Right Panel to Main Layout (Ratio 4)
        main_layout.addWidget(self.label_display, 4)

        # Variable to store current image path
        self.file_path = ""

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', './', 'Image Files (*.png *.jpg *.jpeg)')
        if filename:
            self.file_path = filename
            pixmap = QPixmap(filename)
            scaled_pixmap = pixmap.scaled(self.label_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_display.setPixmap(scaled_pixmap)
            self.label_prediction.setText("Predict:")

    def show_model_structure(self):
        try:
            print("\n" + "="*30)
            print("1.1 Show Architecture")
            print("="*30)
            model = LeNet5()
            summary(model, (1, 32, 32), device="cpu")
            print("="*30 + "\n")
        except Exception as e:
            print(f"Error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show architecture: {str(e)}")

    def show_acc_loss(self):
        # Define file paths for both results
        path_sigmoid = "Loss&Acc_Sigmoid.jpg"
        path_relu = "Loss&Acc_Relu.jpg"
        
        found_any = False

        # Show Sigmoid Result
        if os.path.exists(path_sigmoid):
            img_sig = cv2.imread(path_sigmoid)
            cv2.imshow("Training Result (Sigmoid)", img_sig)
            found_any = True
        else:
            print(f"Warning: {path_sigmoid} not found.")

        # Show ReLU Result
        if os.path.exists(path_relu):
            img_relu = cv2.imread(path_relu)
            cv2.imshow("Training Result (ReLU)", img_relu)
            found_any = True
        else:
            print(f"Warning: {path_relu} not found.")

        # Error handling if neither exists
        if not found_any:
            QMessageBox.warning(self, "Error", "Result images not found! Did you run train.py?")

    def predict_image(self):
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        try:
            # Load Model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = LeNet5(activation='relu').to(device)
            
            model_path = "model/Weight_Relu.pth"
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                QMessageBox.critical(self, "Error", f"Model not found: {model_path}")
                return
            
            model.eval()

            # Preprocessing
            img_cv = cv2.imread(self.file_path, cv2.IMREAD_GRAYSCALE)
            
            # Invert colors (White background -> Black background)
            img_inverted = cv2.bitwise_not(img_cv)
            
            img_pil = transforms.ToPILImage()(img_inverted)
            
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()

            # Update GUI
            self.label_prediction.setText(f"Predict: {predicted_class}")

            # Show Histogram
            probs_np = probabilities.cpu().numpy()[0]
            classes = [str(i) for i in range(10)]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(classes, probs_np, color='skyblue', edgecolor='black')
            plt.title("Probability of each class")
            plt.xlabel("Class")
            plt.ylabel("Probability")
            plt.ylim(0, 1.1)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.2f}',
                             ha='center', va='bottom')
            
            plt.show()

        except Exception as e:
            print(e)
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())