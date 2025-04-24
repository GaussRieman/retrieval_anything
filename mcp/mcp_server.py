# math_server.py
from mcp.server.fastmcp import FastMCP

import cv2

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

@mcp.tool()
def create_random_list(length: int) -> list[int]:
    """Create a random list of integers"""
    import random
    return [random.randint(0, 100) for _ in range(length)]

@mcp.tool()
def sort_list(numbers: list) -> list:
    """Sort a list of numbers"""
    return sorted(numbers)


@mcp.tool()
def read_and_convert_image(image_path: str, operation: str) -> str:
    """Read an image and convert it to a  gray image"""
    if image_path is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    if operation == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif operation == "invert":
        img = cv2.bitwise_not(img)
    elif operation == "blur":
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif operation == "edge":
        img = cv2.Canny(img, 100, 200)
    elif operation == "resize":
        img = cv2.resize(img, (100, 100))
    image_path_new = image_path.replace(".jpg", "_converted.jpg")
    cv2.imwrite(image_path_new, img)
    return image_path_new


if __name__ == "__main__":
    print("Starting Math MCP server...")
    mcp.run(transport="stdio")