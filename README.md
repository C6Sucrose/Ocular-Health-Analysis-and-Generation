# Ocular-Health-Analysis-and-Generation

This project delves into the deep learning-based analysis of ocular health using color fundus images. It encompasses two main aspects: discriminative classification of ocular conditions using EfficientNet, and generative image-to-image translation using CycleGAN to transform diseased eye images into healthy-looking ones.

## Features

*   **Image Preprocessing**: Scripts for preparing and organizing raw ocular fundus images into suitable datasets for both classification and generative tasks.
*   **EfficientNet-Based Classification**: Utilizes the EfficientNet-B0 architecture for robust classification of ocular health states, incorporating advanced training techniques like learning rate scheduling and early stopping.
*   **Generative Image Translation (CycleGAN)**: Implements a CycleGAN model to perform unpaired image-to-image translation, specifically converting diseased retinal images to healthy ones and vice-versa.
*   **Graphical User Interfaces (GUIs)**:
    *   A standalone application (`Final_gui.exe`) for interacting with the classification model.
    *   A separate GUI (`finalgan_gui.exe`) for demonstrating the generative model's capabilities.

## Installation

To set up the project environment, follow these steps:

1.  **Prerequisites**: Ensure you have Python 3.x and `pip` (Python package installer) installed on your system.

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/C6Sucrose/Ocular-Health-Analysis-and-Generation.git
    cd deep-generative-analysis-of-ocular-health-main
    ```

3.  **Install Dependencies**:
    Navigate to the `App` directory and install the required Python packages. Note that the CycleGAN part might have additional dependencies that are handled within its notebook/scripts.
    ```bash
    pip install -r App/requirements.txt
    ```

## Usage

### Data Preparation

The project requires preprocessed image data.
*   For **classification**, the project expects a `preprocessed_images` directory at the root level, structured with `train` and `val` subdirectories, each containing class-specific image folders.
*   For **generative analysis (CycleGAN)**, the data should be organized into `data/trainA`, `data/trainB`, `data/valA`, `data/valB`, `data/testA`, `data/testB` directories, where 'A' typically represents diseased images and 'B' represents healthy images.

You can use the scripts in `preprocessing scripts/` and the data organization steps within `cyclegan-for-dga-of-ocular-health.ipynb` to prepare your datasets.

### Training Classification Models

To train the deep learning models for classification, execute the Python scripts:

*   **Original Training Script**:
    ```bash
    python original_train_efficientnet_augmented.py
    ```
*   **Training with Learning Rate Scheduler**: This script includes an improved learning rate scheduler for potentially better convergence and performance.
    ```bash
    python train_efficientnetlrsched_augmented.py
    ```

### Training Generative Models (CycleGAN)

The CycleGAN model training is detailed in a Jupyter notebook. You would typically run the cells in the notebook or extract the `train.py` script:

*   **Run CycleGAN Training**:
    ```bash
    python train.py
    ```
    (Ensure `utils/` and `models/` directories are set up as described in `cyclegan-for-dga-of-ocular-health.ipynb` and `train.py` is in the root or accessible via `sys.path`).

### Running Classification GUI

To run the compiled graphical user interface for classification:

*   Navigate to the `App` directory and execute the `Final_gui.exe` file:
    ```bash
    ./App/Final_gui.exe
    ```

### Running Generative Prediction

To perform image translation using the trained CycleGAN model, you can use the `predict.py` script:

*   **Run CycleGAN Prediction**:
    ```bash
    python predict.py
    ```
    (Ensure `predict.py` is configured with correct model and input/output paths, and `utils/` and `models/` are accessible).

### Running Generative GUI

To run the compiled graphical user interface for the generative model:

*   Execute the `finalgan_gui.exe` file:
    ```bash
    ./finalgan_gui.exe
    ```

## Project Structure

*   `A_dataset_of_color_fundus_images_for_the_detection.pdf`: Research paper detailing the dataset used in this project.
*   `Eye_health_scans_Final.pdf`: Supplementary documentation or research related to eye health scans.
*   `Project_Notes.pdf`: Additional project notes or documentation.
*   `original_train_efficientnet_augmented.py`: The initial Python script for training the EfficientNet classification model.
*   `train_efficientnetlrsched_augmented.py`: An enhanced training script for classification that incorporates a `ReduceLROnPlateau` learning rate scheduler.
*   `cyclegan-for-dga-of-ocular-health.ipynb`: Jupyter notebook containing the implementation, training, and prediction logic for the CycleGAN model.
*   `Final_gui.py`: Python script for the classification GUI.
*   `finalgan_gui.exe`: Compiled GUI application for the generative model.
*   `App/`: Contains the compiled classification GUI application (`Final_gui.exe`) and its Python dependencies (`requirements.txt`).
*   `preprocessing scripts/`: A directory housing scripts such as `datasplits.py` and `preprocess_and_split__jpg_.py` for data handling and preparation.
*   `Saved Models/`: This directory stores the trained classification model weights, including `best_efficientnet_b0_model.pth`.
*   `models/`: (Created by CycleGAN notebook/scripts) Contains PyTorch model definitions for CycleGAN generators (`generator_A2B.py`, `generator_B2A.py`) and discriminators (`discriminator_A.py`, `discriminator_B.py`).
*   `utils/`: (Created by CycleGAN notebook/scripts) Contains utility scripts for CycleGAN, such as `image_pool.py`, `transforms.py`, and `dataset.py`.
*   `checkpoints/`: (Created by CycleGAN training) Stores trained CycleGAN model weights (e.g., `netG_A2B_best.pth`).
*   `results/`: (Created by CycleGAN training/prediction) Stores generated images and training plots.
*   `data/`: (Created by CycleGAN notebook/scripts) Contains subdirectories for CycleGAN training, validation, and testing data (e.g., `trainA`, `trainB`).

## Dataset Information

The project utilizes color fundus images for both classification and generative tasks.

*   **Classification Dataset**: Expected in a `preprocessed_images` directory with `train` and `val` subdirectories, each containing class-labeled image folders.
*   **Generative Dataset**: Organized into 'A' (diseased) and 'B' (healthy) domains within the `data/` directory, split into `train`, `val`, and `test` sets.

For detailed information regarding the datasets, please refer to the provided PDF documents: `A_dataset_of_color_fundus_images_for_the_detection.pdf` and `Eye_health_scans_Final.pdf`.

## Model Architectures

### Classification Model

The core of this project's classification system is built upon the **EfficientNet-B0** architecture. This model leverages transfer learning, initializing with pre-trained weights to accelerate training and improve performance on the ocular health classification task.

Key training components include:
*   **Loss Function**: Cross-Entropy Loss, suitable for multi-class classification.
*   **Optimizer**: Adam Optimizer, known for its efficiency in deep learning tasks.
*   **Learning Rate Scheduler**: `torch.optim.lr_scheduler.ReduceLROnPlateau`, which dynamically adjusts the learning rate based on validation loss, helping the model converge more effectively.

### Generative Model (CycleGAN)

The generative component uses a **CycleGAN** architecture for unpaired image-to-image translation. This involves two generator networks and two discriminator networks:

*   **Generators**: ResNet-based architectures (`Generator_A2B`, `Generator_B2A`) responsible for translating images between the diseased (A) and healthy (B) domains.
*   **Discriminators**: PatchGAN discriminators (`Discriminator_A`, `Discriminator_B`) that learn to distinguish between real and fake images in each domain.

Key training components for CycleGAN include:
*   **Loss Functions**:
    *   **GAN Loss (MSELoss)**: Encourages generators to produce realistic images that fool the discriminators.
    *   **Cycle Consistency Loss (L1Loss)**: Ensures that translating an image from one domain to another and back recovers the original image, promoting meaningful translations.
    *   **Identity Loss (L1Loss)**: Regularizes the generators to preserve color composition when translating images within the same domain.
*   **Optimizer**: Adam Optimizer for both generators and discriminators, with separate learning rates (TTUR - Two Time-scale Update Rule).
*   **Learning Rate Scheduler**: `torch.optim.lr_scheduler.LambdaLR` for linear decay of the learning rate.
*   **Image Pool**: Utilized during discriminator training to stabilize the training process by using a history of generated images.

## Saved Models

*   **Classification Models**: Trained model weights for the EfficientNet classifier are saved in the `Saved Models/` directory. The primary saved model is `best_efficientnet_b0_model.pth`.
*   **Generative Models**: Trained CycleGAN model weights are saved in the `checkpoints/` directory. Key models include `netG_A2B_best.pth` (generator from diseased to healthy) and `netG_B2A_best.pth` (generator from healthy to diseased).
