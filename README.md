# 🧠 Handwritten Digit Recognition with CNN

This project demonstrates a deep learning-based solution to recognize handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 📌 Objective

To build a machine learning model capable of accurately classifying grayscale images of handwritten digits using deep learning.

---

## 📚 Dataset

- **MNIST Dataset**: 70,000 labeled grayscale images of handwritten digits (28x28 pixels).
  - 60,000 training images
  - 10,000 testing images
- Automatically loaded via `tensorflow.keras.datasets`.

---

## 📦 Requirements

> These libraries are pre-installed in Google Colab. If you're running locally, install them with the following:

```bash
pip install tensorflow matplotlib seaborn scikit-learn numpy

---

## 🛠️ Tools & Technologies

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib & Seaborn**
- **Scikit-learn**

---

## 🔄 Workflow

1. **Data Loading**: Load and visualize MNIST digit samples.
2. **Preprocessing**:
   - Normalize pixel values to `[0, 1]`
   - Reshape input for CNN
   - One-hot encode labels
3. **Model Building**:
   - Convolutional layers
   - MaxPooling
   - Dense and Dropout layers
4. **Model Training**: Train on 90% of training data, validate on 10%.
5. **Evaluation**:
   - Accuracy and loss graphs
   - Confusion matrix and classification report
6. **Prediction**: Test the model on unseen digit samples.

---

## ✅ Results

- Achieved over **98% accuracy** on test data.
- Confusion matrix and metrics demonstrate high precision across all classes.

---


# 🖊️ Handwritten Character Recognition using CNN

This project implements a Convolutional Neural Network (CNN) model using TensorFlow and Keras to recognize handwritten characters from a labeled dataset. It includes data preprocessing, model training, evaluation, and performance reporting.

---

## 📌 Project Highlights

- ✅ Built with TensorFlow/Keras
- 🧠 Recognizes handwritten **digits and characters**
- 📊 Displays accuracy/loss plots
- 🧪 Evaluates model performance using `classification_report`

---

## 🧾 Dataset

The dataset used is based on the [EMNIST](https://www.nist.gov/itl/products-and-services/emnist-dataset) format, extracted from a ZIP file stored on Google Drive.



### File Structure (after extraction):


- Activation: ReLU for hidden layers, Softmax for output
- Optimizer: Adam
- Loss: Categorical Crossentropy

---

## 📈 Results

- Achieves >95% validation accuracy (varies by dataset and epochs)
- Provides a full classification report with precision, recall, and F1-score for each class

---

## 📦 Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

> All dependencies are installed in Colab automatically.

---

## 📸 Sample Output

![Model Accuracy Plot](your-sample-plot.png)

---

## 📚 Future Work

- Deploy as a web app using Flask or Streamlit
- Add handwritten input capture using HTML5 Canvas or OpenCV
- Extend model to support cursive handwriting

---

## 🙌 Acknowledgements

- Dataset: EMNIST by NIST
- Libraries: TensorFlow, scikit-learn, Matplotlib, NumPy

---

## 👨‍💻 Author

**Bitan Karak**  
Feel free to ⭐ star the repo and share your feedback!
