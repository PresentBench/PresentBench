#!/usr/bin/env python3
"""
Usage:
    python -m utils.pdf_to_images
"""
from os import PathLike
import sys
import subprocess
import tempfile
import shutil
import re
import logging
from pathlib import Path
from typing import Optional, List
from PIL import Image

from .pptx_to_pdf import convert_images_to_pdf

# Set up logger
logger = logging.getLogger(__name__)



# ============================================================================
# Public API
# ============================================================================

def convert_pdf_to_images(pdf_path: PathLike, output_dir: Optional[PathLike] = None) -> Optional[List[Path]]:
    """
    Convert a PDF file to a list of PNG images (one per page).

    This uses the same backend strategy as `convert_pptx_to_images`:
    - Prefer `pdftoppm` from poppler-utils if available
    - Fallback to the `pdf2image` library if installed

    Args:
        pdf_path: Path to input PDF file
        output_dir: Directory to save images (optional)
                    If None, images will be saved to `<pdf_dir>/images`.

    Returns:
        List of paths to generated PNG images, or None if conversion failed.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF file: {pdf_path}")

    if output_dir is None:
        output_dir = pdf_path.parent / "images"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if pdftoppm is available (from poppler-utils)
    use_pdftoppm = False
    try:
        subprocess.run(['pdftoppm', '-v'],
                       capture_output=True, check=True)
        use_pdftoppm = True
        logger.info("Using pdftoppm for PDF to image conversion")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.info("pdftoppm not found, will try alternative method")

    temp_dir = None
    try:
        # Always use a temporary directory for conversion work
        temp_dir = Path(tempfile.mkdtemp(prefix='pdf_to_images_'))
        generated_pdf = pdf_path

        # Step: Render PDF pages to PNG images
        png_files = []

        if use_pdftoppm:
            # Use pdftoppm (more reliable)
            cmd_images = [
                'pdftoppm',
                '-png',
                '-r', '150',  # Resolution: 150 DPI
                # '-cropbox',
                str(generated_pdf),
                str(temp_dir / 'page')
            ]

            try:
                result = subprocess.run(
                    cmd_images,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300
                )

                # pdftoppm creates files like page-01.png, page-02.png, etc.
                png_files = sorted(temp_dir.glob('page-*.png'),
                                   key=lambda x: _extract_slide_number(x.name))

            except subprocess.CalledProcessError as e:
                logger.warning("pdftoppm failed, trying alternative method")
                logger.debug(f"  stderr: {e.stderr}")
                use_pdftoppm = False

        if not use_pdftoppm:
            # Fallback 1: Use pymupdf (fitz) — no poppler dependency
            try:
                import fitz  # pymupdf
                doc = fitz.open(str(generated_pdf))
                for i, page in enumerate(doc, 1):
                    mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    png_file = temp_dir / f'page-{i:03d}.png'
                    pix.save(str(png_file))
                    png_files.append(png_file)
                doc.close()
                logger.info("Using pymupdf for PDF to image conversion")
            except ImportError:
                logger.info("pymupdf not found, trying pdf2image")
                # Fallback 2: Use pdf2image library if available
                try:
                    from pdf2image import convert_from_path
                    images_pil = convert_from_path(str(generated_pdf), dpi=150)
                    for i, img in enumerate(images_pil, 1):
                        png_file = temp_dir / f'page-{i:03d}.png'
                        img.save(png_file, 'PNG')
                        png_files.append(png_file)
                except ImportError:
                    logger.error("Neither pymupdf nor pdf2image is available")
                    return None

        if not png_files:
            logger.error("No PNG images were generated from PDF")
            return None

        # move images to output directory
        moved_files = []
        for png_file in png_files:
            dest_file = output_dir / png_file.name
            shutil.move(str(png_file), str(dest_file))
            moved_files.append(dest_file)

        logger.info(f"Generated {len(moved_files)} page images and moved to {output_dir}")
        return moved_files

    except Exception as e:
        logger.error(f"Failed to convert PDF to images: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
    finally:
        # Clean up temporary directory regardless of success or failure
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {cleanup_error}")


def convert_pdf_to_image_pdf(pdf_path: PathLike, output_pdf_path: PathLike) -> Optional[Path]:
    """
    Convert a PDF to an image-based PDF:
    - First render each page of the input PDF to PNG images.
    - Then combine those images into a new PDF via `convert_images_to_pdf`.

    Args:
        pdf_path: Path to input PDF file.
        output_pdf_path: Path to output image-based PDF file.

    Returns:
        Path to the generated PDF file, or None if conversion failed.
    """
    pdf_path = Path(pdf_path)
    output_pdf_path = Path(output_pdf_path)

    temp_dir = None
    try:
        # Create a temporary directory for intermediate images
        temp_dir = Path(tempfile.mkdtemp(prefix='pdf_to_image_pdf_'))

        # Step 1: Convert PDF to images
        image_paths = convert_pdf_to_images(pdf_path, output_dir=temp_dir)
        if not image_paths:
            return None

        # Step 2: Combine images into a new PDF
        result = convert_images_to_pdf(image_paths, output_pdf_path)
        return result

    except Exception as e:
        logger.error(f"PDF to image-PDF conversion failed with unexpected error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
    finally:
        # Clean up temporary directory after using images
        if temp_dir and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")



# ============================================================================
# Helper Functions
# ============================================================================

def _extract_slide_number(filename: str) -> int:
    """
    Extract slide number from filename for sorting.
    Handles formats like "slide_1.png", "1.png", "page_01.png", etc.
    
    Args:
        filename: Name of the file
    
    Returns:
        Slide number, or 0 if not found
    """
    # Try to find numbers in the filename
    numbers = re.findall(r'\d+', filename)
    if numbers:
        # Return the last number found (usually the slide number)
        return int(numbers[-1])
    return 0


if __name__ == "__main__":
    pdf_path = Path("path/to/input.pdf")
    output_pdf_path = Path("path/to/output.pdf")
    convert_pdf_to_image_pdf(pdf_path, output_pdf_path)
