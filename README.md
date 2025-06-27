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
