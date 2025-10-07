# Project Information

This application uses the Whisper AI model to create subtitles from videos.

Notes:
* Python is used whenever possible
* A virtual environment has been created
* The 'faster-whisper' module has been installed using pip.
* Videos are all in English
* Videos are in mp4 format



---
description: 'Python coding conventions and guidelines'
applyTo: '**/*.py'
---

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Prefer classes to standalone functions whenever appropriate

## General Instructions

- Always prioritize readability and clarity.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Ensure lines do not exceed 79 characters.
- For functions and methods, each argument should be on a separate line

## Example of Function Formatting

```python
def calculate_area(
    radius: float
) -> float:
    """
    Calculate the area of a circle given the radius.
    
    Parameters:
    radius (float): The radius of the circle.
    
    Returns:
    float: The area of the circle, calculated as π * radius^2.
    """

    import math
    return math.pi * radius ** 2
```

## Example of Class Formatting

```python
class SampleClass:
    """
    A sample class used to show docstrings and code formatting

    Attributes:
        config (str): A string of configuration information

    Methods:
        __init__:
            Initialize QueryProcessor
        process:
            Process a user query
    """

    def __init__(
        self,
        config: str,
    ) -> None:
        """
        Initialize the class

        Args:
            config (str): Configuration

        Returns:
            None
        """

        self.config = config

    async def process(
        self,
        user_query: str
    ) -> None:
        """
        Process a user query.

        Args:
            user_query (str): The user's query

        Returns:
            None
        """

        try:
            print("This is where code is added")

        except Exception as e:
            print(f"❌ A problem has occured: {e}\n")

```
