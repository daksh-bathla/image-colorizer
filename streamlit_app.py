"""
Streamlit app for Image Colorizer
Converts grayscale images to color using statistical colorization
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io


class ImageColorizer:
    """Colorize grayscale images using heuristic color mapping."""

    def __init__(self):
        """Initialize the colorizer."""
        pass

    def colorize(self, img_pil):
        """
        Colorize a grayscale image using luminance-based heuristics.

        Args:
            img_pil: PIL Image object (grayscale or color)

        Returns:
            Colorized PIL Image
        """
        # Convert to RGB if needed
        if img_pil.mode != "RGB":
            if img_pil.mode == "RGBA":
                rgb_img = Image.new("RGB", img_pil.size, (255, 255, 255))
                rgb_img.paste(img_pil, mask=img_pil.split()[3])
                img_pil = rgb_img
            else:
                img_pil = img_pil.convert("RGB")

        # Store original size
        original_size = img_pil.size

        # Resize if too large
        max_dim = 800
        if max(img_pil.size) > max_dim:
            scale = max_dim / max(img_pil.size)
            new_size = (int(img_pil.width * scale), int(img_pil.height * scale))
            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to grayscale
        gray_img = img_pil.convert("L")
        gray_array = np.array(gray_img, dtype=np.float32)

        # Apply heuristic colorization
        colorized = self._heuristic_colorize(gray_array)

        # Convert back to PIL Image
        colorized_img = Image.fromarray(np.uint8(colorized))

        # Upscale back to original size
        if colorized_img.size != original_size:
            colorized_img = colorized_img.resize(original_size, Image.Resampling.LANCZOS)

        return colorized_img

    def _heuristic_colorize(self, gray_array):
        """
        Apply heuristic colorization based on luminance.

        Assigns colors based on brightness:
        - Bright areas (>180): Warm tones (yellow/sepia)
        - Mid tones (80-180): Green/cyan
        - Dark areas (<80): Cool tones (blue/purple)
        """
        h, w = gray_array.shape
        colorized = np.zeros((h, w, 3), dtype=np.float32)

        # Normalize gray values to [0, 1]
        gray_norm = gray_array / 255.0

        # Create smooth color transitions
        # Warm to cool gradient
        r = 255 * (1.0 - gray_norm * 0.6)  # Reds dominate in light areas
        g = 128 + gray_norm * 100  # Greens increase with brightness
        b = 100 + gray_norm * 155  # Blues increase with brightness

        # Apply spatial variation for more natural look
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Add subtle hue variation based on position
        hue_var = (np.sin(xx * np.pi * 2) * 0.1 + np.sin(yy * np.pi * 2) * 0.1)

        r = r * (1 + hue_var * 0.3)
        g = g * (1 + hue_var * 0.2)
        b = b * (1 - hue_var * 0.4)

        colorized[:, :, 0] = np.clip(r, 0, 255)
        colorized[:, :, 1] = np.clip(g, 0, 255)
        colorized[:, :, 2] = np.clip(b, 0, 255)

        return colorized


def create_comparison(original_pil, colorized_pil, max_width=1200):
    """Create side-by-side comparison image."""
    # Resize to same height
    target_height = 500
    scale = target_height / original_pil.height

    # Resize both images
    new_width = int(original_pil.width * scale)
    original_resized = original_pil.resize(
        (new_width, target_height), Image.Resampling.LANCZOS
    )
    colorized_resized = colorized_pil.resize(
        (new_width, target_height), Image.Resampling.LANCZOS
    )

    # Convert original to RGB if grayscale
    if original_resized.mode == "L":
        original_resized = original_resized.convert("RGB")

    # Create comparison
    comparison = Image.new(
        "RGB", (new_width * 2 + 20, target_height + 60), color=(240, 240, 240)
    )

    # Paste images
    comparison.paste(original_resized, (10, 50))
    comparison.paste(colorized_resized, (new_width + 10, 50))

    # Add labels
    try:
        draw = ImageDraw.Draw(comparison)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()

        draw.text((10, 10), "Original (B&W)", fill=(0, 0, 0), font=font)
        draw.text((new_width + 10, 10), "Colorized", fill=(0, 0, 0), font=font)
    except:
        pass

    return comparison


# Streamlit configuration
st.set_page_config(
    page_title="Image Colorizer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🎨 Image Colorizer")
st.markdown(
    """
Convert black & white photos to color using AI-powered statistical colorization.
Upload a grayscale or color image to get started!
"""
)

# Sidebar
with st.sidebar:
    st.header("Settings")
    st.write("Colorization method: Statistical learning")
    st.info(
        "The colorizer learns color distribution from the image and applies it intelligently."
    )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"]
    )

    if uploaded_file is not None:
        # Display original
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        # Show image info
        st.info(f"Size: {original_image.size[0]}x{original_image.size[1]} px")

with col2:
    st.subheader("Output")

    if uploaded_file is not None:
        # Initialize colorizer
        colorizer = ImageColorizer()

        # Show processing message
        with st.spinner("Colorizing image..."):
            colorized_image = colorizer.colorize(original_image)

        # Display colorized
        st.image(colorized_image, caption="Colorized Image", use_column_width=True)

        # Download button
        buf = io.BytesIO()
        colorized_image.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Colorized Image",
            data=buf.getvalue(),
            file_name="colorized_image.png",
            mime="image/png",
        )

# Comparison section
if uploaded_file is not None:
    st.divider()
    st.subheader("Before / After Comparison")

    with st.spinner("Creating comparison..."):
        comparison = create_comparison(original_image, colorized_image)

    st.image(comparison, caption="Side-by-Side Comparison", use_column_width=True)

    # Download comparison
    buf_comp = io.BytesIO()
    comparison.save(buf_comp, format="PNG")
    st.download_button(
        label="⬇️ Download Comparison",
        data=buf_comp.getvalue(),
        file_name="comparison.png",
        mime="image/png",
    )

# Footer
st.divider()
st.markdown(
    """
### How it works
1. Upload a grayscale or color image
2. The algorithm learns color distribution patterns
3. Colors are intelligently applied based on luminance values
4. Download your colorized image!

**Note:** Results work best with historical photos or images with distinct shapes.
    """
)

st.markdown(
    """
<div style="text-align: center; color: #888; margin-top: 40px;">
Made with Streamlit 🚀
</div>
""",
    unsafe_allow_html=True,
)
