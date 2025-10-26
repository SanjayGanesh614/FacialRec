"""
Setup script for FaceGuard AI - Advanced Facial Recognition System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    pass

setup(
    name="faceguard-ai",
    version="1.0.0",
    author="FaceGuard AI Team",
    author_email="team@faceguard-ai.com",
    description="Advanced Facial Recognition System with Military-Grade Encryption",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/faceguard-ai/faceguard-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "faceguard-demo=scripts.run_demo:main",
            "faceguard-setup=scripts.start_system:main",
            "faceguard-test=tests.test_setup:main",
            "faceguard-cli=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "src.web": ["templates/*.html", "static/*.css"],
        "docs": ["*.md"],
        "data": ["sample_images/*"],
    },
    zip_safe=False,
    keywords=[
        "facial-recognition",
        "computer-vision", 
        "biometrics",
        "security",
        "encryption",
        "machine-learning",
        "opencv",
        "flask",
        "web-interface",
        "privacy-preserving",
    ],
    project_urls={
        "Bug Reports": "https://github.com/faceguard-ai/faceguard-ai/issues",
        "Source": "https://github.com/faceguard-ai/faceguard-ai",
        "Documentation": "https://faceguard-ai.readthedocs.io/",
    },
)