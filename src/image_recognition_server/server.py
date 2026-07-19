"""
MCP Image Recognition Server.

Exposes two tools via the MCP stdio protocol:
- ``describe_image`` ― describe a base64-encoded image using vision AI
- ``describe_image_from_file`` ― describe an image file on disk using vision AI

Supports Anthropic and OpenAI vision providers, configurable via the
``VISION_PROVIDER`` env var with optional ``FALLBACK_PROVIDER``.
OCR (Tesseract) is opt-in via ``ENABLE_OCR=true``.
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from PIL import Image

# Load environment variables from project root BEFORE local imports
# so that vision client function defaults can read from os.getenv.
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

from .mcp_stdio import McpServer, ToolDef  # noqa: E402
from .utils.image import image_to_base64, validate_base64_image  # noqa: E402
from .utils.ocr import OCRError, extract_text_from_image  # noqa: E402
from .vision.anthropic import AnthropicVision  # noqa: E402
from .vision.openai import OpenAIVision  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_ENCODING = "utf-8"
ENCODING = os.getenv("MCP_OUTPUT_ENCODING", DEFAULT_ENCODING)

# ---------------------------------------------------------------------------
# Logging ― writes to file so stdout stays clean for MCP JSON-RPC
# ---------------------------------------------------------------------------
log_file_path = os.path.join(os.path.dirname(__file__), "mcp_server.log")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename=log_file_path,
    filemode="a",
)
logger = logging.getLogger(__name__)
logger.info(f"Using encoding: {ENCODING}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize_output(text: str) -> str:
    """Sanitize output string to replace problematic characters."""
    if text is None:
        return ""
    try:
        return text.encode(ENCODING, "replace").decode(ENCODING)
    except Exception as e:
        logger.error(f"Error during sanitization: {str(e)}", exc_info=True)
        return text


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = McpServer(
    "mcp-image-recognition",
    version="0.2.0",  # Android-compatible refactor (no pydantic/mcp deps)
)


# ---------------------------------------------------------------------------
# Vision client dispatch
# ---------------------------------------------------------------------------

def get_vision_client() -> Union[AnthropicVision, OpenAIVision]:
    """Get the configured vision client based on environment settings."""
    provider = os.getenv("VISION_PROVIDER", "anthropic").lower()

    try:
        if provider == "anthropic":
            return AnthropicVision()
        elif provider == "openai":
            return OpenAIVision()
        else:
            raise ValueError(f"Invalid vision provider: {provider}")
    except Exception as e:
        # Try fallback provider if configured
        fallback = os.getenv("FALLBACK_PROVIDER")
        if fallback and fallback.lower() != provider:
            logger.warning(
                f"Primary provider failed: {str(e)}. Trying fallback: {fallback}"
            )
            if fallback.lower() == "anthropic":
                return AnthropicVision()
            elif fallback.lower() == "openai":
                return OpenAIVision()
        raise


async def process_image_with_ocr(image_data: str, prompt: str) -> str:
    """Process image with both vision AI and OCR.

    Args:
        image_data: Base64 encoded image data
        prompt: Prompt for vision AI

    Returns:
        str: Combined description from vision AI and OCR

    Raises:
        ValueError: If vision API returns empty response or OCR fails
    """
    client = get_vision_client()

    # Handle both sync (Anthropic) and async (OpenAI) clients
    if isinstance(client, OpenAIVision):
        description = await client.describe_image(image_data, prompt)
    else:
        description = client.describe_image(image_data, prompt)

    if not description or description == "No description available.":
        raise ValueError("Vision API returned empty or default response")

    # OCR is opt-in via ENABLE_OCR env var
    ocr_enabled = os.getenv("ENABLE_OCR", "false").lower() == "true"
    if ocr_enabled:
        try:
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))

            if ocr_text := extract_text_from_image(image, ocr_required=True):
                description += (
                    f"\n\nAdditionally, this is the output of tesseract-ocr: "
                    f"{ocr_text}"
                )
        except OCRError as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise ValueError(f"OCR Error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during OCR: {str(e)}")
            raise

    # Prepend a version marker so callers can distinguish the refactored
    # stdlib MCP server (main-android) from the original FastMCP version (main).
    MARKER = "[MCP-STDLIB v0.2.0] "
    return MARKER + sanitize_output(description)


# ---------------------------------------------------------------------------
# Tool: describe_image
# ---------------------------------------------------------------------------

DESCRIBE_IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "image": {
            "type": "string",
            "description": "Base64-encoded image data (raw base64 string, not a data URI)",
        },
        "prompt": {
            "type": "string",
            "description": "Optional prompt to guide the vision AI description",
        },
    },
    "required": ["image"],
}


@mcp.tool(ToolDef(
    name="describe_image",
    description="Describe the contents of an image using vision AI. "
                "Supports Anthropic Claude and OpenAI GPT vision models.",
    inputSchema=DESCRIBE_IMAGE_SCHEMA,
))
async def describe_image(
    image: str,
    # Configurable default prompt via DEFAULT_IMAGE_PROMPT env var
    prompt: str = os.getenv("DEFAULT_IMAGE_PROMPT", "Please describe this image in detail."),
) -> str:
    """Describe the contents of an image using vision AI.

    Args:
        image: Base64-encoded image data
        prompt: Optional prompt to use for the description.

    Returns:
        str: Detailed description of the image
    """
    try:
        logger.info(f"Processing image description request with prompt: {prompt}")
        logger.debug(f"Image data length: {len(image)}")

        if not validate_base64_image(image):
            raise ValueError("Invalid base64 image data")

        result = await process_image_with_ocr(image, prompt)
        if not result:
            raise ValueError("Received empty response from processing")

        logger.info("Successfully processed image")
        return sanitize_output(result)
    except ValueError as e:
        logger.error(f"Input error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error describing image: {str(e)}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Tool: describe_image_from_file
# ---------------------------------------------------------------------------

DESCRIBE_IMAGE_FROM_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "filepath": {
            "type": "string",
            "description": "Absolute or relative path to the image file on disk",
        },
        "prompt": {
            "type": "string",
            "description": "Optional prompt to guide the vision AI description",
        },
    },
    "required": ["filepath"],
}


@mcp.tool(ToolDef(
    name="describe_image_from_file",
    description="Describe the contents of an image file on disk using vision AI. "
                "Reads the file, converts to base64, and delegates to describe_image.",
    inputSchema=DESCRIBE_IMAGE_FROM_FILE_SCHEMA,
))
async def describe_image_from_file(
    filepath: str,
    # Configurable default prompt via DEFAULT_IMAGE_PROMPT env var
    prompt: str = os.getenv("DEFAULT_IMAGE_PROMPT", "Please describe this image in detail."),
) -> str:
    """Describe the contents of an image file using vision AI.

    Args:
        filepath: Path to the image file
        prompt: Optional prompt to use for the description.

    Returns:
        str: Detailed description of the image
    """
    try:
        logger.info(f"Processing image file: {filepath}")

        image_data, mime_type = image_to_base64(filepath)
        logger.info(f"Successfully converted image to base64. MIME type: {mime_type}")
        logger.debug(f"Base64 data length: {len(image_data)}")

        # Delegate to the describe_image handler (above)
        result = await describe_image(image=image_data, prompt=prompt)

        if not result:
            raise ValueError("Received empty response from processing")

        return sanitize_output(result)
    except FileNotFoundError:
        logger.error(f"Image file not found: {filepath}")
        raise
    except ValueError as e:
        logger.error(f"Input error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error processing image file: {str(e)}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
