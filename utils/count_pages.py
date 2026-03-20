"""
Utility: count pages/slides in a PDF or PPTX file.
"""
from pathlib import Path



def count_pages(file_path: str) -> int:
    """
    Count pages (PDF) or slides (PPTX).
    
    Args:
        file_path: Path to a PDF or PPTX file.
        
    Returns:
        Page/slide count.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    file_ext = file_path.suffix.lower()
    
    if file_ext == '.pdf':
        return _count_pdf_pages(file_path)
    elif file_ext == '.pptx':
        return _count_pptx_pages(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Only .pdf and .pptx are supported.")


def _count_pdf_pages(file_path: Path) -> int:
    """Count pages in a PDF."""
    try:
        import PyPDF2
    except ImportError:
        try:
            import pypdf as PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 or pypdf is required to process PDF files. "
                "Install with: pip install pypdf2  (or)  pip install pypdf"
            )
    
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        return len(pdf_reader.pages)


def _count_pptx_pages(file_path: Path) -> int:
    """Count slides in a PPTX."""
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError(
            "python-pptx is required to process PPTX files. "
            "Install with: pip install python-pptx"
        )
    
    prs = Presentation(file_path)
    return len(prs.slides)


def check_slide_count(slides_path: str, min_count: int, max_count: int) -> dict[str, str]:
    """
    Check whether the slide count is within a given range.
    
    Args:
        slides_path: Path to a PDF or PPTX file.
        min_count: Minimum slide count (inclusive).
        max_count: Maximum slide count (inclusive).
        
    Returns:
        Dict containing 'answer' and 'explanation'.
    """
    try:
        num_slides = count_pages(slides_path)
        if min_count <= num_slides <= max_count:
            answer = "yes"
            explanation = f"[judged by code] The number of slides is {num_slides}, which is within the required range of {min_count}-{max_count}."
        elif num_slides < min_count:
            answer = "no"
            explanation = f"[judged by code] The number of slides is {num_slides}, which is too few. The required range is {min_count}-{max_count} slides."
        else:  # num_slides > max_count
            answer = "no"
            explanation = f"[judged by code] The number of slides is {num_slides}, which is too many. The required range is {min_count}-{max_count} slides."

    except Exception as e:
        answer = "no"
        explanation = f"[judged by code] Error counting slides: {str(e)}"

    return {
        'answer': answer,
        'explanation': explanation,
    }



if __name__ == "__main__":
    # Example usage.
    import sys
    
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        try:
            page_count = count_pages(test_file)
            print(f"File '{test_file}' has {page_count} page(s)")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python count_pages.py <file_path>")
        print("Example: python count_pages.py example.pdf")
        print("         python count_pages.py example.pptx")

