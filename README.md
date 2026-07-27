# Image Colorizer

Convert black & white photos to color using AI-powered statistical colorization.

## Features

- Upload grayscale or color images
- Intelligent colorization using statistical learning
- Side-by-side before/after comparison
- Download colorized images
- Fast processing with Streamlit

## Installation

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Deploy to Streamlit Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select this repository
5. Choose `streamlit_app.py` as the main file

## How It Works

1. Upload a grayscale or color image
2. The algorithm learns color distribution patterns from the image
3. Colors are intelligently applied based on luminance values
4. Download your colorized result

## Files

- `streamlit_app.py` - Web application
- `colorize_images.py` - CLI version of the colorizer
- `requirements.txt` - Python dependencies

## License

MIT
