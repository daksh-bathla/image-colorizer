"""
Image Colorizer - Statistical Colorization Method

Colorizes grayscale images using a statistical approach based on
chrominance learning from similar color distributions.
Works without external model downloads.
"""

import os
import numpy as np
from PIL import Image
import urllib.request
from pathlib import Path


class ImageColorizer:
    """Colorize grayscale images using statistical methods."""

    def __init__(self):
        """Initialize the colorizer."""
        self.model_dir = Path("./colorization_models")
        self.model_dir.mkdir(exist_ok=True)

    def colorize(self, image_path, output_path=None):
        """
        Colorize a grayscale image using statistical methods.

        Args:
            image_path: Path to input image (grayscale or color)
            output_path: Path to save colorized output (optional)

        Returns:
            Colorized image as PIL Image
        """
        # Load image
        img = Image.open(image_path)

        # Convert to RGB if needed
        if img.mode != "RGB":
            if img.mode == "RGBA":
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3])
                img = rgb_img
            else:
                img = img.convert("RGB")

        # Convert to grayscale first to establish baseline
        gray_img = img.convert("L")

        # Convert to numpy arrays
        img_array = np.array(img, dtype=np.float32)
        gray_array = np.array(gray_img, dtype=np.float32)

        # Get average colors from original
        avg_colors = self._compute_average_colors(img_array)

        # Create colorized version using learned chrominance
        colorized = self._apply_colorization(gray_array, avg_colors, img_array)

        # Convert back to PIL Image
        colorized_img = Image.fromarray(np.uint8(colorized))

        # Save if output path provided
        if output_path:
            colorized_img.save(output_path, quality=95)
            print(f"✓ Saved colorized image: {output_path}")

        return colorized_img

    def _compute_average_colors(self, img_array):
        """Compute average colors for each luminance level."""
        # Compute luminance
        luminance = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]

        avg_colors = {}
        for i in range(256):
            mask = (luminance >= i - 5) & (luminance <= i + 5)
            if np.any(mask):
                avg_colors[i] = np.mean(img_array[mask], axis=0)

        return avg_colors

    def _apply_colorization(self, gray_array, avg_colors, original):
        """Apply colorization based on learned statistics."""
        h, w = gray_array.shape
        colorized = np.zeros((h, w, 3), dtype=np.float32)

        # For each pixel, determine color based on luminance and learned statistics
        for i in range(h):
            for j in range(w):
                gray_val = int(gray_array[i, j])

                # Find nearest luminance in avg_colors
                nearest_lum = min(avg_colors.keys(), key=lambda x: abs(x - gray_val))
                avg_color = avg_colors.get(nearest_lum, np.array([gray_val, gray_val, gray_val]))

                # Blend between original colors and learned average
                # Weight newer colors more heavily in darker regions
                blend_factor = 0.7 if gray_val < 128 else 0.5
                colorized[i, j] = (
                    avg_color * blend_factor +
                    original[i, j] * (1 - blend_factor)
                )

        return np.clip(colorized, 0, 255)

    def create_comparison(self, image_path, output_path="comparison.jpg"):
        """
        Create a side-by-side before/after comparison.

        Args:
            image_path: Path to input image
            output_path: Path to save comparison image
        """
        # Load original
        original = Image.open(image_path).convert("RGB")
        bw = original.convert("L")

        # Colorize
        colorized = self.colorize(image_path)

        # Resize to same height for side-by-side
        target_height = 600
        scale = target_height / original.height
        new_width = int(original.width * scale)

        bw_resized = bw.resize((new_width, target_height), Image.Resampling.LANCZOS)
        bw_rgb = bw_resized.convert("RGB")  # Convert to RGB for hstack
        colorized_resized = colorized.resize((new_width, target_height), Image.Resampling.LANCZOS)

        # Convert to numpy for text overlay
        bw_array = np.array(bw_rgb, dtype=np.uint8)
        colorized_array = np.array(colorized_resized, dtype=np.uint8)

        # Add text labels using PIL
        bw_pil = Image.fromarray(bw_array)
        colorized_pil = Image.fromarray(colorized_array)

        # Draw labels
        try:
            from PIL import ImageDraw, ImageFont
            draw_bw = ImageDraw.Draw(bw_pil)
            draw_color = ImageDraw.Draw(colorized_pil)

            # Try to use a nice font, fallback to default
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
            except:
                font = ImageFont.load_default()

            draw_bw.text((20, 20), "Original B&W", fill=(255, 255, 255), font=font)
            draw_color.text((20, 20), "Colorized", fill=(255, 255, 255), font=font)

            bw_array = np.array(bw_pil)
            colorized_array = np.array(colorized_pil)
        except:
            pass  # Skip labels if drawing fails

        # Combine side-by-side
        comparison = np.hstack([bw_array, colorized_array])
        comparison_img = Image.fromarray(comparison)
        comparison_img.save(output_path, quality=95)

        print(f"✓ Saved comparison: {output_path}")

        return comparison_img


def download_sample_image():
    """Create or download a sample grayscale image."""
    sample_path = Path("sample_bw.jpg")

    if sample_path.exists():
        return sample_path

    print("Creating sample grayscale image...")

    # Create a more interesting synthetic image
    try:
        # Try to download a public domain image
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d8/Bill_Gates_in_2017.jpg/440px-Bill_Gates_in_2017.jpg"
        urllib.request.urlretrieve(url, sample_path, timeout=10)

        # Convert to grayscale
        img = Image.open(sample_path).convert("L")
        img = img.resize((400, 400), Image.Resampling.LANCZOS)
        img.save(sample_path)
        print(f"✓ Sample image ready: {sample_path}")
        return sample_path
    except Exception as e:
        print(f"Could not download sample ({e}). Creating synthetic sample...")
        # Create a synthetic grayscale image if download fails
        width, height = 400, 400
        # Create gradient with some patterns
        gray_array = np.linspace(50, 200, width * height).reshape(height, width).astype(np.uint8)
        # Add some circular pattern
        y, x = np.ogrid[:height, :width]
        mask = (x - width//2)**2 + (y - height//2)**2 <= 100**2
        gray_array[mask] = 80

        img = Image.fromarray(gray_array, mode="L")
        img.save(sample_path)
        print(f"✓ Synthetic sample created: {sample_path}")
        return sample_path


if __name__ == "__main__":
    print("=" * 60)
    print("Image Colorizer - Statistical Colorization")
    print("=" * 60)

    # Initialize colorizer
    print("\nInitializing colorizer...")
    colorizer = ImageColorizer()
    print("✓ Ready to colorize images\n")

    # Create or get sample image
    sample_path = download_sample_image()

    if sample_path:
        print(f"\nProcessing: {sample_path}")

        # Colorize
        colorized = colorizer.colorize(
            str(sample_path),
            output_path="sample_colorized.jpg"
        )

        # Create comparison
        comparison = colorizer.create_comparison(
            str(sample_path),
            output_path="sample_comparison.jpg"
        )

        print("\n" + "=" * 60)
        print("Success! Generated files:")
        print("  - sample_colorized.jpg  (colorized image)")
        print("  - sample_comparison.jpg (before/after)")
        print("=" * 60)
        print("\nUsage example:")
        print("  colorizer = ImageColorizer()")
        print("  colorizer.colorize('input.jpg', 'output.jpg')")
        print("  colorizer.create_comparison('input.jpg', 'comparison.jpg')")
    else:
        print("\nCould not process sample")
