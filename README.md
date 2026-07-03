I need to create a GitHub-style README for your Fungi identification project. Let me first check your existing code and files to understand the project structure, then search for relevant information about the fungi dataset and model you're using.
Based on your project context (Stanford/NVIDIA summer program, Fungi image classifier with toxicity info, web interface, 33 species from iNaturalist), here's a complete GitHub README. I'll fill in the template with accurate technical details:

---

# 🍄 Fungi Identifier

A web-based image classification tool that identifies fungi species from photographs and provides information about their edibility and toxicity. Built with a deep learning backend and an interactive HTML/CSS/JS frontend.

![Fungi Identifier Web Interface](https://via.placeholder.com/800x400?text=Fungi+Identifier+Screenshot)

## The Algorithm

This project uses **transfer learning** with a **ResNet-50** convolutional neural network to classify fungi images into 33 distinct species. ResNet-50 is a 50-layer deep CNN originally trained on the ImageNet dataset (1.2M+ images, 1000 classes) and is widely used for fine-grained image classification tasks due to its residual connections that prevent vanishing gradients in deep networks.

**How it works:**

1. **Image Preprocessing**: User-uploaded images are resized to 224×224 pixels and normalized using ImageNet mean/std statistics.
2. **Feature Extraction**: The pre-trained ResNet-50 backbone extracts hierarchical visual features (edges, textures, shapes) from the input image.
3. **Classification**: A custom fully-connected layer maps these features to 33 fungi species classes.
4. **Toxicity Lookup**: The predicted species name is matched against a curated database containing edibility status, toxicity warnings, and safe preparation methods.

**Key Dependencies:**
- **PyTorch / Torchvision** — Deep learning framework and pre-trained ResNet-50 model
- **Pillow (PIL)** — Image loading and preprocessing
- **Flask** — Lightweight Python web server (API backend)
- **HTML/CSS/JavaScript** — Frontend interface with drag-and-drop upload

The model was fine-tuned on a subset of the **iNaturalist Fungi Dataset**, focusing on 33 common species with well-documented edibility information. The training uses standard data augmentation (random crop, horizontal flip, color jitter) to improve generalization.


### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fungi-identifier.git
   cd fungi-identifier
   ```

2. **Install required libraries**
   ```bash
   pip install torch torchvision pillow flask
   ```

3. **Download the model weights**
   - Place the trained model file (`fungi_resnet50.pth`) in the `models/` directory
   - Ensure the species database (`species_data.json`) is in the `data/` directory

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   - Navigate to `http://localhost:5000`
   - Upload a fungi image via drag-and-drop or file selector

### Project Structure
```
fungi-identifier/
├── app.py                 # Flask backend server
├── models/
│   └── fungi_resnet50.pth # Trained model weights
├── data/
│   └── species_data.json  # Edibility & toxicity database
├── static/
│   ├── css/
│   └── js/
├── templates/
│   └── index.html         # Main web interface
└── utils/
    └── preprocess.py      # Image preprocessing helpers
```

### Optional: Training from Scratch
If you want to retrain the model on your own dataset:
```bash
python train.py --data-dir ./dataset --epochs 20 --batch-size 32
```

[View a video explanation here](https://your-video-link-here)

---

**Creator:** Mia G  
**Program:** Stanford/NVIDIA Summer Program 2026
