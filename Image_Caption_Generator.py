#!/usr/bin/env python3
"""
Streamlit Image Caption Generator - Vision + Language AI
Uses BLIP, CLIP, ViT, and transformers for image captioning

Requirements:
pip install streamlit torch torchvision transformers pillow requests numpy opencv-python
pip install accelerate plotly

Run with: streamlit run image_caption_app.py
"""

import streamlit as st
import torch
import requests
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    ViTImageProcessor, ViTForImageClassification
)
import io
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .tech-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache all vision-language models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    
    with st.spinner("🔄 Loading AI models... This may take a few minutes on first run."):
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # BLIP for image captioning
            status_text.text("Loading BLIP model...")
            models['blip_processor'] = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            models['blip_model'] = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
            progress_bar.progress(33)
            
            # CLIP for image-text understanding
            status_text.text("Loading CLIP model...")
            models['clip_processor'] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            models['clip_model'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            progress_bar.progress(66)
            
            # ViT for image classification
            status_text.text("Loading ViT model...")
            models['vit_processor'] = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            models['vit_model'] = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
            progress_bar.progress(100)
            
            status_text.text("✅ All models loaded successfully!")
            progress_bar.empty()
            status_text.empty()
            
            return models, device
            
        except Exception as e:
            st.error(f"❌ Error loading models: {e}")
            st.info("💡 Make sure you have installed: pip install torch transformers pillow")
            return None, None

class ImageCaptionGenerator:
    """Streamlit Image Caption Generator"""
    
    def __init__(self, models, device):
        self.models = models
        self.device = device
    
    def generate_blip_caption(self, image):
        """Generate caption using BLIP model"""
        try:
            inputs = self.models['blip_processor'](image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out = self.models['blip_model'].generate(**inputs, max_length=50, num_beams=5)
            
            caption = self.models['blip_processor'].decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            return f"Error: {e}"
    
    def analyze_with_clip(self, image, custom_queries=None):
        """Analyze image with CLIP for image-text similarity"""
        default_queries = [
            "a photo of a person", "a photo of an animal", "a photo of nature",
            "a photo of food", "a photo of a vehicle", "a photo of a building",
            "indoor scene", "outdoor scene", "close-up photo", "landscape photo"
        ]
        
        text_queries = custom_queries if custom_queries else default_queries
        
        try:
            inputs = self.models['clip_processor'](text=text_queries, images=image, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clip_model'](**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get all results
            results = []
            for i, (prob, query) in enumerate(zip(probs[0], text_queries)):
                results.append({
                    'category': query,
                    'confidence': prob.item()
                })
            
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            return results
            
        except Exception as e:
            return [{'category': f'Error: {e}', 'confidence': 0}]
    
    def classify_with_vit(self, image, top_k=5):
        """Classify image using Vision Transformer"""
        try:
            inputs = self.models['vit_processor'](images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.models['vit_model'](**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Get top predictions
            top_predictions = torch.topk(predictions, top_k)
            results = []
            
            for score, idx in zip(top_predictions.values[0], top_predictions.indices[0]):
                label = self.models['vit_model'].config.id2label[idx.item()]
                results.append({
                    'class': label,
                    'confidence': score.item()
                })
            
            return results
        except Exception as e:
            return [{'class': f'Error: {e}', 'confidence': 0}]
    
    def generate_enhanced_caption(self, blip_caption, clip_results, vit_results):
        """Generate enhanced caption combining all model insights"""
        scene_type = clip_results[0]['category'].replace('a photo of ', '')
        main_object = vit_results[0]['class']
        
        enhanced_parts = []
        
        if clip_results[0]['confidence'] > 0.3:
            enhanced_parts.append(f"This appears to be {scene_type}")
        
        enhanced_parts.append(f"The image shows {blip_caption.lower()}")
        
        if vit_results[0]['confidence'] > 0.1:
            enhanced_parts.append(f"with elements suggesting {main_object}")
        
        return ". ".join(enhanced_parts).capitalize() + "."

def create_confidence_chart(results, title, color):
    """Create confidence visualization"""
    labels = [item.get('category', item.get('class', 'Unknown'))[:30] for item in results[:5]]
    confidences = [item['confidence'] for item in results[:5]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=labels,
            orientation='h',
            marker_color=color,
            text=[f"{conf:.2%}" for conf in confidences],
            textposition='inside'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Confidence",
        yaxis_title="Categories",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Image Caption Generator</h1>
        <p>Vision + Language AI using BLIP, CLIP, ViT & Transformers</p>
        <div>
            <span class="tech-badge">BLIP</span>
            <span class="tech-badge">CLIP</span>
            <span class="tech-badge">ViT</span>
            <span class="tech-badge">Transformers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models, device = load_models()
    
    if models is None:
        st.stop()
    
    generator = ImageCaptionGenerator(models, device)
    
    # Sidebar
    st.sidebar.header("🎛️ Configuration")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    use_custom_queries = st.sidebar.checkbox("Use Custom CLIP Queries")
    custom_queries = []
    
    if use_custom_queries:
        st.sidebar.write("Enter custom queries (one per line):")
        custom_query_text = st.sidebar.text_area(
            "Custom Queries",
            "a photo of a cat\na photo of a dog\nindoor scene\noutdoor scene",
            height=100
        )
        custom_queries = [q.strip() for q in custom_query_text.split('\n') if q.strip()]
    
    vit_top_k = st.sidebar.slider("ViT Top-K Results", 3, 10, 5)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # Image upload options
        upload_option = st.radio(
            "Choose input method:",
            ["Upload File", "Enter URL", "Use Sample"]
        )
        
        image = None
        
        if upload_option == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp']
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        
        elif upload_option == "Enter URL":
            image_url = st.text_input("Enter image URL:")
            
            if image_url:
                try:
                    response = requests.get(image_url, stream=True)
                    image = Image.open(response.raw)
                except Exception as e:
                    st.error(f"Error loading image: {e}")
        
        elif upload_option == "Use Sample":
            sample_images = {
                "Dog": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee?w=500",
                "Mountain": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=500",
                "Food": "https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=500",
                "City": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=500"
            }
            
            selected_sample = st.selectbox("Choose a sample image:", list(sample_images.keys()))
            
            if st.button("Load Sample Image"):
                try:
                    response = requests.get(sample_images[selected_sample], stream=True)
                    image = Image.open(response.raw)
                except Exception as e:
                    st.error(f"Error loading sample image: {e}")
        
        # Display image
        if image is not None:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            st.image(image, caption=f"Input Image ({image.size[0]}x{image.size[1]})", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Generate Captions", type="primary"):
                with st.spinner("🤖 Analyzing image with AI models..."):
                    
                    # Generate results
                    blip_caption = generator.generate_blip_caption(image)
                    clip_results = generator.analyze_with_clip(image, custom_queries if custom_queries else None)
                    vit_results = generator.classify_with_vit(image, vit_top_k)
                    enhanced_caption = generator.generate_enhanced_caption(blip_caption, clip_results, vit_results)
                    
                    # Store results in session state
                    st.session_state.results = {
                        'blip_caption': blip_caption,
                        'clip_results': clip_results,
                        'vit_results': vit_results,
                        'enhanced_caption': enhanced_caption
                    }
    
    with col2:
        st.header("📊 Results")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Enhanced Caption
            st.markdown("""
            <div class="result-card">
                <h3>🎯 Enhanced Caption</h3>
                <p style="font-size: 1.1em; font-weight: 500;">{}</p>
            </div>
            """.format(results['enhanced_caption']), unsafe_allow_html=True)
            
            # Individual model results
            st.subheader("🔍 Model Analysis")
            
            # BLIP Caption
            st.markdown(f"**📝 BLIP Caption:** {results['blip_caption']}")
            
            # CLIP Analysis Chart
            if results['clip_results']:
                st.subheader("🎯 CLIP Scene Analysis")
                clip_fig = create_confidence_chart(
                    results['clip_results'], 
                    "Scene Type Confidence", 
                    "#667eea"
                )
                st.plotly_chart(clip_fig, use_container_width=True)
            
            # ViT Classification Chart
            if results['vit_results']:
                st.subheader("🏷️ ViT Object Classification")
                vit_fig = create_confidence_chart(
                    results['vit_results'], 
                    "Object Classification Confidence", 
                    "#764ba2"
                )
                st.plotly_chart(vit_fig, use_container_width=True)
            
            # Detailed metrics
            st.subheader("📈 Detailed Metrics")
            
            col1_metrics, col2_metrics, col3_metrics = st.columns(3)
            
            with col1_metrics:
                st.markdown("""
                <div class="metric-card">
                    <h4>BLIP</h4>
                    <p>Caption Length</p>
                    <h2>{}</h2>
                </div>
                """.format(len(results['blip_caption'].split())), unsafe_allow_html=True)
            
            with col2_metrics:
                top_clip_conf = results['clip_results'][0]['confidence'] if results['clip_results'] else 0
                st.markdown("""
                <div class="metric-card">
                    <h4>CLIP</h4>
                    <p>Top Confidence</p>
                    <h2>{:.1%}</h2>
                </div>
                """.format(top_clip_conf), unsafe_allow_html=True)
            
            with col3_metrics:
                top_vit_conf = results['vit_results'][0]['confidence'] if results['vit_results'] else 0
                st.markdown("""
                <div class="metric-card">
                    <h4>ViT</h4>
                    <p>Top Confidence</p>
                    <h2>{:.1%}</h2>
                </div>
                """.format(top_vit_conf), unsafe_allow_html=True)
            
            # Export results
            st.subheader("💾 Export Results")
            
            export_data = {
                "Enhanced Caption": results['enhanced_caption'],
                "BLIP Caption": results['blip_caption'],
                "Top CLIP Result": f"{results['clip_results'][0]['category']} ({results['clip_results'][0]['confidence']:.2%})" if results['clip_results'] else "N/A",
                "Top ViT Result": f"{results['vit_results'][0]['class']} ({results['vit_results'][0]['confidence']:.2%})" if results['vit_results'] else "N/A"
            }
            
            export_text = "\n".join([f"{k}: {v}" for k, v in export_data.items()])
            
            st.download_button(
                label="📄 Download Results as Text",
                data=export_text,
                file_name="image_caption_results.txt",
                mime="text/plain"
            )
        
        else:
            st.info("👆 Upload an image and click 'Generate Captions' to see results!")
    
    # About section
    with st.expander("ℹ️ About This App"):
        st.write("""
        This Image Caption Generator uses state-of-the-art Vision-Language models:
        
        - **BLIP** (Bootstrapped Language-Image Pre-training): Generates natural language captions
        - **CLIP** (Contrastive Language-Image Pre-training): Analyzes image-text relationships
        - **ViT** (Vision Transformer): Classifies objects and scenes in images
        - **Transformers**: The underlying framework powering all models
        
        The app combines insights from all three models to create enhanced, detailed captions
        that provide comprehensive understanding of your images.
        """)

if __name__ == "__main__":
    main()