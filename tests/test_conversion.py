import io
import logging
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pypdf import PdfWriter

import judge_all
from utils import docx_to_pdf, pdf_to_images, pptx_to_pdf


class PptxToPdfTest(unittest.TestCase):
    def test_libreoffice_conversion_has_timeout_and_isolated_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "presentation.pptx"
            target = root / "presentation.pdf"
            source.touch()
            calls = []

            def fake_run(command, **kwargs):
                calls.append((command, kwargs))
                if '--convert-to' in command:
                    output_dir = Path(command[command.index('--outdir') + 1])
                    (output_dir / "presentation.pdf").touch()
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

            with patch("utils.pptx_to_pdf.subprocess.run", side_effect=fake_run):
                result = pptx_to_pdf._convert_with_libreoffice(source, target)

            self.assertEqual(result, target.resolve())
            convert_command, convert_kwargs = calls[1]
            self.assertTrue(
                any(arg.startswith('-env:UserInstallation=file://') for arg in convert_command)
            )
            self.assertEqual(convert_kwargs["timeout"], 300)


class PdfToImagesTest(unittest.TestCase):
    @staticmethod
    def _fake_fitz():
        class FakePixmap:
            def save(self, path):
                Path(path).write_bytes(b"png")

        class FakePage:
            def get_pixmap(self, **_kwargs):
                return FakePixmap()

        class FakeDocument:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter([FakePage()])

        return SimpleNamespace(
            Matrix=lambda *_args: object(),
            open=lambda *_args: FakeDocument(),
        )

    def test_uses_pymupdf_when_pdftoppm_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "input.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=72, height=72)
            with pdf_path.open("wb") as stream:
                writer.write(stream)

            with (
                patch("utils.pdf_to_images.subprocess.run", side_effect=FileNotFoundError),
                patch.dict(sys.modules, {"fitz": self._fake_fitz(), "pdf2image": None}),
            ):
                result = pdf_to_images.convert_pdf_to_images(pdf_path, root / "images")

            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].exists())

    def test_falls_back_to_pymupdf_when_pdftoppm_times_out(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "input.pdf"
            pdf_path.touch()

            def fake_run(command, **_kwargs):
                if '-v' in command:
                    return subprocess.CompletedProcess(command, 0)
                raise subprocess.TimeoutExpired(command, 300)

            with (
                patch("utils.pdf_to_images.subprocess.run", side_effect=fake_run),
                patch.dict(sys.modules, {"fitz": self._fake_fitz(), "pdf2image": None}),
            ):
                result = pdf_to_images.convert_pdf_to_images(pdf_path, root / "images")

            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].exists())

    def test_falls_back_to_pdf2image_when_pymupdf_rendering_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_path = root / "input.pdf"
            pdf_path.touch()

            class FakeImage:
                def save(self, path, _format):
                    Path(path).write_bytes(b"png")

            failing_fitz = SimpleNamespace(
                open=lambda *_args: (_ for _ in ()).throw(RuntimeError("render failed")),
            )
            fake_pdf2image = SimpleNamespace(
                convert_from_path=lambda *_args, **_kwargs: [FakeImage()],
            )

            with (
                patch("utils.pdf_to_images.subprocess.run", side_effect=FileNotFoundError),
                patch.dict(
                    sys.modules,
                    {"fitz": failing_fitz, "pdf2image": fake_pdf2image},
                ),
            ):
                result = pdf_to_images.convert_pdf_to_images(pdf_path, root / "images")

            self.assertIsNotNone(result)
            self.assertEqual(len(result), 1)
            self.assertTrue(result[0].exists())


class DocxToPdfTest(unittest.TestCase):
    def test_missing_file_is_reported_through_logging(self):
        stdout = io.StringIO()

        with redirect_stdout(stdout), self.assertLogs("utils.docx_to_pdf", level="ERROR") as logs:
            result = docx_to_pdf.convert_docx_to_pdf("/missing/document.docx")

        self.assertFalse(result)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("File does not exist", "\n".join(logs.output))


class JudgeAllLoggingTest(unittest.TestCase):
    def test_run_once_preserves_existing_root_handlers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_root = root / "data"
            case_dir = data_root / "domain" / "source" / "case"
            (case_dir / "generation_task").mkdir(parents=True)
            (case_dir / "material.md").touch()
            (case_dir / "generation_task" / "judge_prompt.json").touch()

            result_root = root / "results"
            result_dir = result_root / "domain" / "source" / "case" / "generation_task" / "results"
            result_dir.mkdir(parents=True)
            (result_dir / "slides_generation_failed.txt").touch()

            root_logger = logging.getLogger()
            original_handlers = list(root_logger.handlers)
            original_level = root_logger.level
            sentinel = logging.NullHandler()
            root_logger.addHandler(sentinel)
            try:
                with patch("judge_all.judge.main", return_value=None):
                    judge_all.run_once(
                        api_type="gemini",
                        model="model",
                        thinking_level=None,
                        type_name="domain",
                        data_item_dir=case_dir,
                        result_root=result_root,
                        data_root=data_root,
                    )

                self.assertIn(sentinel, root_logger.handlers)
                self.assertEqual(root_logger.level, original_level)
            finally:
                root_logger.handlers[:] = original_handlers
                root_logger.setLevel(original_level)


if __name__ == "__main__":
    unittest.main()
