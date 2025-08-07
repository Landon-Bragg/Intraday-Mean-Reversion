# setup.py
"""
Setup script for professional package installation
Run with: pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="intraday-mean-reversion",
    version="1.0.0",
    description="Professional-grade intraday mean reversion trading strategy",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/intraday-mean-reversion",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.1.0",
        "yfinance>=0.2.0",
        "plotly>=5.10.0",
        "dash>=2.6.0",
        "pyyaml>=6.0",
        "reportlab>=3.6.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
        ],
        "professional": [
            "bloomberg-api-sdk>=1.0.0",
            "refinitiv-data>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    entry_points={
        "console_scripts": [
            "mean-reversion=main:main",
        ],
    },
)