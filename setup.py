from setuptools import setup, find_packages

# Charger les dépendances depuis requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    required_packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="network_analysis_ir",  # Nom du package
    version="0.1.0",  # Version initiale
    packages=find_packages(where="src"),  # Recherche les packages dans src/
    package_dir={"": "src"},  # Spécifie que le code source est dans src/
    install_requires=required_packages,  # Utilisation du requirements.txt
    entry_points={
        "console_scripts": [
            "network-analysis=app.app:main",  # Permet d'exécuter l'application en ligne de commande
        ]
    },
    author="M.MANSOUR",
    author_email="mehdi.mansour@univ-lyon2.fr",
    description="Un package d'analyse de graphes et de recherche "
                "d'information dans des corpora d'articles scientifiques",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ds314159/network_analysis_ir",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",  # Version minimum de Python requise
)
