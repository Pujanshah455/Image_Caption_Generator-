# 🖼️ Image Caption Generator 🤖

A **Streamlit** web application that generates intelligent captions and scene descriptions for images using state-of-the-art **Vision-Language AI models** — BLIP, CLIP, and ViT — powered by HuggingFace Transformers.

![app-demo](https://github.com/yourusername/image-caption-generator/assets/demo.gif)

## 🔍 Features

✅ Upload or URL input for any image  
✅ Generate descriptive captions using **BLIP**  
✅ Analyze scene and context using **CLIP**  
✅ Classify objects using **ViT (Vision Transformer)**  
✅ Combine all outputs into one **enhanced caption**  
✅ Visualize confidence levels with interactive charts  
✅ Customizable settings and export results  

---

## 🚀 Live Demo

Try it instantly with Streamlit:  
👉 [Run on Streamlit Cloud](https://share.streamlit.io/yourusername/image-caption-generator)

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python UI framework)
- **Backend Models**:
  - 🤖 [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base) for image captioning
  - 🧠 [CLIP](https://huggingface.co/openai/clip-vit-base-patch32) for text-image similarity
  - 🔍 [ViT](https://huggingface.co/google/vit-base-patch16-224) for object classification
- **Visualization**: Plotly
- **Libraries**: PyTorch, Transformers, PIL, Requests, NumPy, OpenCV

---

## 📦 Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator
2️⃣ Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
If you don’t have a requirements.txt, install manually:

bash
Copy
Edit
pip install streamlit torch torchvision transformers pillow requests numpy opencv-python accelerate plotly
▶️ Run the App
bash
Copy
Edit
streamlit run Image_Caption_Generator.py
📸 Screenshot


📁 Folder Structure
bash
Copy
Edit
image-caption-generator/
│
├── Image_Caption_Generator.py   # Main Streamlit app
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies (optional)
└── assets/                      # (Optional) sample images, demo GIFs
💡 How it Works
Upload or select an image.

The app uses:

BLIP to generate a base caption

CLIP to evaluate scene/text similarity

ViT to classify objects in the image

All outputs are combined into an enhanced caption.

Confidence levels are visualized in interactive charts.

✨ Sample Output
Uploaded Image: Dog on a mountain

BLIP Caption: A dog standing on top of a mountain
Top CLIP Scene: A photo of an animal (98.3%)
ViT Prediction: Eskimo dog (85.6%)
Enhanced Caption:

This appears to be an animal. The image shows a dog standing on top of a mountain with elements suggesting Eskimo dog.

📄 Export Feature
📝 Download a summary of model outputs and enhanced caption as a .txt file directly from the UI.

