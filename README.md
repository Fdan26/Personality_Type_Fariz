# Personality Type Prediction (Introvert vs Extrovert)

This project uses a neural network built with **TensorFlow/Keras** to predict personality types — **Introvert** or **Extrovert** — based on behavioral data.

## 📊 Dataset

The dataset is sourced from Kaggle:

[Extrovert vs Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data)

It contains behavioral features such as:
| Feature                     | Description                                                                  |
| --------------------------- | ---------------------------------------------------------------------------- |
| `Time_spent_Alone`          | Number of hours an individual typically spends alone daily                   |
| `Stage_fear`                | Whether the person experiences stage fear (`Yes` = 1, `No` = 0)              |
| `Social_event_attendance`   | Frequency (scale 0–10) of attending social events                            |
| `Going_outside`             | How often the individual goes outside (scale 0–10)                           |
| `Drained_after_socializing` | Whether the individual feels drained after socializing (`Yes` = 1, `No` = 0) |
| `Friends_circle_size`       | Number of close friends                                                      |
| `Post_frequency`            | Frequency of posting on social media (e.g., scale 0–10 or posts per day)     |


## 🧠 Model

A simple feedforward neural network was built using TensorFlow:

- Input layer with 7 features
- Two hidden layers using ReLU activation
- Output layer with 2 neurons using sigmoid activation (for binary classification)

### Model Architecture

```python
model = keras.Sequential([
    keras.Input(shape=(7,)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])
```
### How to run this application?
- Clone this github repository and run the following command:
```cmd
streamlit run main.py
```