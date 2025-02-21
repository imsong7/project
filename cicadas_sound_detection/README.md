# Detecting the Hyalessa fuscata

## Intro
- **Objectives**: By training **Residual Network 50**, if you give it a random voice data file, it will determine whether the data is a cicada call.
- **Data**: Donga Science Used 190 pieces of data from the 10th and 11th periods of the “Earth Love Exploration Team”, an ecological research citizen science project jointly operated by the science magazine ‘Children’s Science Donga’.

## Processing
- Convert to WAV
- Process with STFT for spectrograms (3,000–20,000Hz, 12 time segments)
- Resize to 224×224
- Label data
- Split into training/testing sets, augment training data
- Using Model: Res50

## Result
- The accuracy was **94.17%**, the test loss was 0.1831, and the precision and recall were also 0.84 and 0.94, respectively.
- The Confusion Matrix came out as [[1206 75] [25 408]], and although there was an imbalance in the data labels, the result of calculating the F1 score considering the imbalance in the labels was 0.89.

## Discussion
- The trained model has a problem in that it does not classify cicadas and cicadas well, but it is expected that accuracy can be increased through more learning.
- It has the advantage of being able to be used to classify the sounds of insects and animals other than cicadas.
- In addition to ResNet50, there is also a way to increase accuracy by using deeper models, ResNet101 and ResNet152.
