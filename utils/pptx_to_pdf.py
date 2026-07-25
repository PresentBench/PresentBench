#!/usr/bin/env python3
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

# Set up logger
logger = logging.getLogger(__name__)



# ============================================================================
# Public API
# ============================================================================

def convert_pptx_to_pdf(pptx_path: PathLike, pdf_path: Optional[PathLike] = None, to_image_pdf: bool = False) -> Optional[Path]:
    """
    Convert a PPTX file to PDF. Support Chinese characters.
    
    Args:
        pptx_path: Path to the input PPTX file
        pdf_path: Output PDF path (default: `<pptx_dir>/<pptx_stem>.pdf`)
        to_image_pdf: Whether to convert to pure-image PDF
    
    Returns:
        Path to the generated PDF file, or None if conversion failed
    """
    pptx_path = Path(pptx_path)
    
    if not pptx_path.exists():
        raise FileNotFoundError(f"PPTX file not found: {pptx_path}")
    if pptx_path.suffix.lower() != '.pptx':
        raise ValueError(f"File is not a PPTX file: {pptx_path}")

    # Determine output directory
    if pdf_path is None:
        pdf_path = pptx_path.parent / f"{pptx_path.stem}.pdf"
    else:
        pdf_path = Path(pdf_path)

    if to_image_pdf:
        return _convert_to_image_pdf(pptx_path, pdf_path)
    else:
        return _convert_with_libreoffice(pptx_path, pdf_path)


def convert_pptx_to_images(pptx_path: PathLike, output_dir: Optional[PathLike] = None) -> Optional[List[Path]]:
    """
    Convert PPTX file to a list of PNG images (one per slide).
    
    Args:
        pptx_path: Path to input PPTX file
        output_dir: Directory to save images (optional)
                    If provided, images will be moved here after successful conversion.
                    If None, images will remain in temporary directory and caller is responsible for cleanup.
    
    Returns:
        List of paths to generated PNG images, or None if conversion failed
    """
    pptx_path = Path(pptx_path)
    
    if not pptx_path.exists():
        raise FileNotFoundError(f"PPTX file not found: {pptx_path}")
    if pptx_path.suffix.lower() != '.pptx':
        raise ValueError(f"File is not a PPTX file: {pptx_path}")

    if output_dir is None:
        output_dir = pptx_path.parent / "images"
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
    
    # Always use a temporary directory for conversion work
    # This prevents overwriting existing files and allows us to verify conversion success
    temp_dir = None
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix='pptx_to_images_'))
        
        # Step 1: Convert PPTX to PDF first (more reliable)
        temp_pdf = temp_dir / f"{pptx_path.stem}_temp.pdf"
        
        # Use existing _convert_with_libreoffice function
        generated_pdf = _convert_with_libreoffice(pptx_path, temp_pdf)
        if not generated_pdf or not generated_pdf.exists():
            logger.error("Failed to convert PPTX to intermediate PDF")
            return None
        
        # Step 2: Render PDF pages to PNG images
        png_files = []
        
        if use_pdftoppm:
            # Use pdftoppm (more reliable)
            cmd_images = [
                'pdftoppm',
                '-png',
                '-r', '150',  # Resolution: 150 DPI
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
                logger.warning(f"pdftoppm failed, trying alternative method")
                logger.debug(f"  stderr: {e.stderr}")
                use_pdftoppm = False
        
        if not use_pdftoppm:
            # Fallback: Use pdf2image library if available
            try:
                from pdf2image import convert_from_path
                images_pil = convert_from_path(str(generated_pdf), dpi=150)
                for i, img in enumerate(images_pil, 1):
                    png_file = temp_dir / f'page-{i:03d}.png'
                    img.save(png_file, 'PNG')
                    png_files.append(png_file)
            except ImportError:
                logger.error("pdf2image library not found")
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
        
        logger.info(f"Generated {len(moved_files)} slide images and moved to {output_dir}")
        return moved_files
        
    except Exception as e:
        logger.error(f"Failed to convert PPTX to images: {e}")
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


def convert_images_to_pdf(image_paths: List[PathLike], pdf_path: PathLike) -> Optional[Path]:
    """
    Combine multiple images into a single PDF file.
    
    Args:
        image_paths: List of paths to image files
        pdf_path: Path to output PDF file
    
    Returns:
        Path to the generated PDF file, or None if conversion failed
    """
    if not image_paths:
        logger.error("No image paths provided")
        return None
    
    image_paths = [Path(img_path) for img_path in image_paths]
    if not all(img_path.exists() for img_path in image_paths):
        raise FileNotFoundError(f"Image files not found: {image_paths}")
    if not all(img_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] for img_path in image_paths):
        raise ValueError(f"Images are not PNG/JPG/JPEG files: {image_paths}")

    pdf_path = Path(pdf_path)
    
    try:
        # Open all images and convert to RGB (required for PDF)
        images = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                # Convert to RGB if necessary (PNG might be RGBA)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to open image {img_path}: {e}")
                continue
        
        if not images:
            logger.error("No valid images to combine into PDF")
            return None
        
        # Ensure output directory exists
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save all images as a single PDF
        # Use the first image's size as reference, or let each image use its own size
        images[0].save(
            str(pdf_path),
            'PDF',
            resolution=100.0,
            save_all=True,
            append_images=images[1:] if len(images) > 1 else []
        )
        
        logger.info(f"Successfully created PDF from {len(images)} slides: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logger.error(f"Failed to combine images into PDF: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


# ============================================================================
# Private Functions
# ============================================================================

def _convert_with_libreoffice(pptx_path: Path, pdf_path: Optional[Path]) -> Optional[Path]:
    """
    Convert PPTX to PDF using LibreOffice headless mode.
    
    Args:
        pptx_path: Path to input PPTX file
        pdf_path: Path to output PDF file
    
    Returns:
        Path to the generated PDF file, or None if conversion failed
    """
    # Check if LibreOffice is installed (try 'soffice' first, then 'libreoffice')
    libreoffice_cmd = None
    for cmd in ['soffice', 'libreoffice']:
        try:
            subprocess.run([cmd, '--version'],
                          capture_output=True, check=True)
            libreoffice_cmd = cmd
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    if libreoffice_cmd is None:
        logger.error("LibreOffice is not installed or not in PATH.")
        logger.error("Please install LibreOffice: sudo apt-get install libreoffice")
        return None
    
    # Get absolute paths
    pptx_abs = pptx_path.resolve()
    pdf_abs = pdf_path.resolve()

    # Create a temporary directory for LibreOffice output
    # This prevents overwriting existing files and allows us to verify conversion success
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix='pptx_to_pdf_')
        temp_dir_path = Path(temp_dir)
        
        # LibreOffice command
        # --headless: Run without GUI
        # --convert-to pdf: Convert to PDF
        # --outdir: Output directory (temporary directory)
        cmd = [
            libreoffice_cmd,
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', str(temp_dir_path),
            str(pptx_abs)
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            # LibreOffice creates PDF in the temp directory with the same name as input
            temp_pdf = temp_dir_path / f"{pptx_path.stem}.pdf"
            
            # Check if PDF was created in temp directory (this verifies conversion success)
            if temp_pdf.exists():
                # Ensure output directory exists
                pdf_abs.parent.mkdir(parents=True, exist_ok=True)
                
                # Move the PDF from temp directory to target location
                shutil.move(str(temp_pdf), str(pdf_abs))
                logger.info(f'Successfully converted "{pptx_path}" to PDF: "{pdf_abs}"')
                return pdf_abs
            else:
                logger.error(f"pptx-to-pdf conversion failed: PDF not found in temp directory: {temp_pdf}")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"pptx-to-pdf conversion failed with error:")
            logger.error(f"  Return code: {e.returncode}")
            logger.error(f"  stdout: {e.stdout}")
            logger.error(f"  stderr: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"pptx-to-pdf conversion failed with unexpected error: {e}")
            return None
    finally:
        # Clean up temporary directory
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")


def _convert_to_image_pdf(pptx_path: Path, pdf_path: Path) -> Optional[Path]:
    """
    Convert PPTX to PDF via images: first convert each slide to PNG, then combine into PDF.
    Uses a two-step approach: PPTX -> PNG images -> PDF
    
    Args:
        pptx_path: Path to input PPTX file
        pdf_path: Path to output PDF file
    
    Returns:
        Path to the generated PDF file, or None if conversion failed
    """
    temp_dir = None
    try:
        # Create a temporary directory for images
        temp_dir = Path(tempfile.mkdtemp(prefix='pptx_to_images_'))
        
        # Step 1: Convert PPTX to images
        image_paths = convert_pptx_to_images(pptx_path, output_dir=temp_dir)
        if not image_paths:
            return None
        
        # Step 2: Combine images into PDF
        result = convert_images_to_pdf(image_paths, pdf_path)
        return result
        
    except Exception as e:
        logger.error(f"Conversion failed with unexpected error: {e}")
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
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert PPTX files to PDF format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct conversion (default)
  python pptx_to_pdf.py slides.pptx
  python pptx_to_pdf.py slides.pptx output.pdf
  
  # Convert via images (each slide -> image -> PDF)
  python pptx_to_pdf.py slides.pptx --to-image-pdf
  python pptx_to_pdf.py slides.pptx output.pdf --to-image-pdf
        """
    )
    parser.add_argument('pptx_file', help='Path to the input PPTX file')
    parser.add_argument('output', nargs='?', help='Path to the output PDF file (optional)')
    parser.add_argument('--to-image-pdf', action='store_true',
                       help='Convert via images: first convert each slide to image, then combine into PDF')
    args = parser.parse_args()
    
    pptx_path = args.pptx_file
    pdf_path = args.output
    
    # Configure logging for command-line usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    result = convert_pptx_to_pdf(pptx_path, pdf_path, to_image_pdf=True if args.to_image_pdf else False)
    
    if result:
        logger.info(f"\nConversion successful!")
        logger.info(f"PDF saved to: {result}")
        sys.exit(0)
    else:
        logger.error(f"\nConversion failed!")
        sys.exit(1)
