#!/usr/bin/env python3
"""
Utility to convert DOCX files to PDF.
Supports batch conversion of all Press Release files in a specified folder.
"""

import os
from pathlib import Path

try:
    from docx2pdf import convert
    USE_DOCX2PDF = True
except ImportError:
    USE_DOCX2PDF = False
    print("Warning: docx2pdf is not installed; falling back to LibreOffice CLI if available")
    print("Install with: pip install docx2pdf")


def _convert_docx_to_pdf_libreoffice(docx_path, output_dir=None):
    """
    Convert DOCX to PDF using the LibreOffice command-line tool.
    
    Args:
        docx_path: DOCX file path.
        output_dir: Output directory; if None, use the DOCX file's directory.
    
    Returns:
        bool: Whether the conversion succeeded.
    """
    import subprocess
    
    if output_dir is None:
        output_dir = os.path.dirname(docx_path)
    
    try:
        # LibreOffice command.
        cmd = [
            'libreoffice',
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', output_dir,
            docx_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return True
        else:
            print(f"Error: LibreOffice conversion failed - {result.stderr}")
            return False
    except FileNotFoundError:
        print("Error: LibreOffice not found. Please ensure LibreOffice is installed.")
        return False
    except subprocess.TimeoutExpired:
        print(f"Error: conversion timed out - {docx_path}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False


def convert_docx_to_pdf(docx_path, pdf_path=None):
    """
    Convert a DOCX file to PDF.
    
    Args:
        docx_path: DOCX file path.
        pdf_path: Output PDF path; if None, auto-generate.
    
    Returns:
        bool: Whether the conversion succeeded.
    """
    if not os.path.exists(docx_path):
        print(f"Error: file does not exist - {docx_path}")
        return False
    
    if pdf_path is None:
        pdf_path = os.path.splitext(docx_path)[0] + '.pdf'
    
    try:
        if USE_DOCX2PDF:
            # Use docx2pdf library.
            convert(docx_path, pdf_path)
            print(f"✓ Converted: {os.path.basename(docx_path)} -> {os.path.basename(pdf_path)}")
            return True
        else:
            # Use LibreOffice.
            output_dir = os.path.dirname(pdf_path)
            if _convert_docx_to_pdf_libreoffice(docx_path, output_dir):
                # LibreOffice auto-generates the PDF file name; verify it exists.
                expected_pdf = os.path.join(output_dir, os.path.splitext(os.path.basename(docx_path))[0] + '.pdf')
                if os.path.exists(expected_pdf):
                    # Rename to the requested path if needed.
                    if expected_pdf != pdf_path:
                        os.rename(expected_pdf, pdf_path)
                    print(f"✓ Converted: {os.path.basename(docx_path)} -> {os.path.basename(pdf_path)}")
                    return True
            return False
    except Exception as e:
        print(f"Error: conversion failed - {docx_path}")
        print(f"  Details: {str(e)}")
        return False

