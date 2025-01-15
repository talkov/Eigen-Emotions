

😄 Eigen Emotions: Detecting Emotions From Facial Expressions
Eigen Emotions is a computer vision project that detects emotions from facial expressions using classical image processing techniques and high-level linear algebra. By combining the Viola-Jones algorithm for face detection and a novel eigen-based approach for emotion classification, this system showcases the potential of non-deep-learning methods in affective computing.
<img src="https://github.com/talkov/eigen-emotions-/blob/main/logo.jpg" alt="Eigen Emotions logo">

📖 Overview
Key Features:
Face Detection: Utilizes the Viola-Jones algorithm with Haar-like features for efficient and accurate face detection.
Emotion Recognition: Employs eigenfaces, derived through Principal Component Analysis (PCA), to classify emotions.
Emotions Detected: Happy, Angry, Sad, and Surprised.
Real-Time Testing: Demonstrated on live users in a library setting, highlighting practical applicability.

🛠️ Methodology
Face Detection:
Preprocessing: Normalizes image lighting using mean intensity adjustment.
Algorithm: Viola-Jones, leveraging Haar-like features and integral images for rapid detection.
Emotion Classification:
Preprocessing:
Converts images to grayscale.
Adjusts mean intensity for lighting normalization.
Enhances contrast using histogram equalization.
Eigenfaces:
PCA is used to derive eigenfaces for each emotion.
A scoring system selects the most relevant eigenfaces for classification.
Classification:
Feature extraction with eigenfaces.
Multi-class Support Vector Machine (SVM) with a one-vs-rest strategy for emotion recognition.

🔍 Results
Real-World Testing:
High accuracy for happy and surprised emotions.
Challenges with sad and angry emotions when expressions were subtle.
Overall success rate improved with user prompts for more expressive emotions.
Scalability: The algorithm performs well with reduced training data, demonstrating robustness.

📂 Project Structure
Face Detection:
Viola-Jones implementation with Haar-like features and integral images.
Emotion Recognition:
PCA-based eigenface generation.
SVM classifier for multi-class emotion detection.
Preprocessing:
Grayscale conversion, histogram equalization, and intensity normalization.

📊 Example Outputs
Here are some visual results from the project:
![image](https://github.com/user-attachments/assets/e3411347-0178-48a4-bf43-e8af474edd08)
![image](https://github.com/user-attachments/assets/30526234-fe10-4439-8e5c-9ffb94055a90)
![image](https://github.com/user-attachments/assets/4f61f296-4bff-49c6-ab6d-f8a471a373d8)

Confusion matrix for handpicking method:
![image](https://github.com/user-attachments/assets/227d1839-ddef-469b-a5f1-f185dbb2ce7d)

Confusion matrix for scoring method
![image](https://github.com/user-attachments/assets/7c504856-21fc-4d07-a5f5-4d41ed732469)

👥 Contributors
This project was developed by:

Tal Kozakov
Omer Hazan
Yaniv Rosenberg
Ofir Bar Lev
📚 References
M. Turk and A. Pentland, "Face Recognition Using Eigenfaces," 1991.
P. Viola and M. Jones, "Rapid Object Detection Using a Boosted Cascade of Simple Features," 2001.
Feel free to explore the repository and contribute! 🚀
