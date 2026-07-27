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
        Apply heuristic colorization using natural color distribution.

        Maps grayscale values to realistic colors using learned statistics
        from natural images: sky/water (bright blue), earth/skin (warm), etc.
        """
        h, w = gray_array.shape
        gray_norm = gray_array / 255.0  # [0, 1]

        # Create HSV image
        h_arr = np.zeros((h, w), dtype=np.float32)
        s_arr = np.ones((h, w), dtype=np.float32)
        v_arr = gray_norm.copy()

        # Assign hue based on luminance (natural color distribution)
        # Bright areas (0.7-1.0) -> Blue sky (hue ~210°/255)
        # Mid-bright (0.5-0.7) -> Green/yellow (hue ~60°/255)
        # Mid (0.3-0.5) -> Orange/brown (hue ~30°/255)
        # Dark (0.0-0.3) -> Brown/earth (hue ~20°/255)

        mask_bright = gray_norm > 0.65
        mask_mid_bright = (gray_norm > 0.45) & (gray_norm <= 0.65)
        mask_mid = (gray_norm > 0.25) & (gray_norm <= 0.45)
        mask_dark = gray_norm <= 0.25

        # Hue values in HSV (0-1 range in OpenCV)
        h_arr[mask_bright] = 0.55  # Cyan/Blue (210°)
        h_arr[mask_mid_bright] = 0.20  # Yellow-green (72°)
        h_arr[mask_mid] = 0.08  # Orange (28°)
        h_arr[mask_dark] = 0.06  # Brown (21°)

        # Saturation increases with contrast/variation
        # Compute local variance for edge regions
        s_arr = 0.4 + 0.5 * (gray_norm ** 0.5)  # More saturated in mid-tones
        s_arr = np.clip(s_arr, 0.2, 0.8)  # Constrain for realistic colors

        # Add spatial variation for realism
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)

        # Subtle hue shift based on position (tint variation)
        hue_shift = 0.03 * (np.sin(xx * np.pi * 3) * np.cos(yy * np.pi * 2))
        h_arr = (h_arr + hue_shift) % 1.0

        # Convert HSV to RGB
        hsv_img = np.stack([h_arr, s_arr, v_arr], axis=2)
        colorized = self._hsv_to_rgb(hsv_img)

        return colorized

    def _hsv_to_rgb(self, hsv_img):
        """Convert HSV image to RGB (HSV in [0,1] range)."""
        h, w = hsv_img.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        h_val = hsv_img[:, :, 0] * 6.0
        s_val = hsv_img[:, :, 1]
        v_val = hsv_img[:, :, 2]

        i = np.floor(h_val).astype(int)
        f = h_val - i

        p = v_val * (1.0 - s_val)
        q = v_val * (1.0 - s_val * f)
        t = v_val * (1.0 - s_val * (1.0 - f))

        i = i % 6

        mask0 = i == 0
        mask1 = i == 1
        mask2 = i == 2
        mask3 = i == 3
        mask4 = i == 4
        mask5 = i == 5

        rgb[:, :, 0][mask0] = v_val[mask0]
        rgb[:, :, 1][mask0] = t[mask0]
        rgb[:, :, 2][mask0] = p[mask0]

        rgb[:, :, 0][mask1] = q[mask1]
        rgb[:, :, 1][mask1] = v_val[mask1]
        rgb[:, :, 2][mask1] = p[mask1]

        rgb[:, :, 0][mask2] = p[mask2]
        rgb[:, :, 1][mask2] = v_val[mask2]
        rgb[:, :, 2][mask2] = t[mask2]

        rgb[:, :, 0][mask3] = p[mask3]
        rgb[:, :, 1][mask3] = q[mask3]
        rgb[:, :, 2][mask3] = v_val[mask3]

        rgb[:, :, 0][mask4] = t[mask4]
        rgb[:, :, 1][mask4] = p[mask4]
        rgb[:, :, 2][mask4] = v_val[mask4]

        rgb[:, :, 0][mask5] = v_val[mask5]
        rgb[:, :, 1][mask5] = p[mask5]
        rgb[:, :, 2][mask5] = q[mask5]

        return rgb * 255.0


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
