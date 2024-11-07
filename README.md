# Senior Project I: Thai Word Censor Automation

This project aims to create an automated system for censoring Thai profanity in videos using Natural Language Processing (NLP) and audio waveform analysis.

## Project Overview

The system processes video files to detect and censor Thai profanity in audio components.

## First approach

1. Data Collection:
   - Gather a large dataset of Thai audio samples containing both clean and profane speech.
   - Ensure the dataset is diverse and representative of different speakers, accents, and audio qualities.
2. Data Preprocessing:
   - Convert audio files to a consistent format (e.g., WAV) and sample rate.
   - Extract audio features such as Mel-frequency cepstral coefficients (MFCCs) or spectrograms from the waveforms. (Not success doing it)
3. Labeling:
   - Annotate the dataset, marking the timestamps where profanity occurs. (0 non-profanity, 1 profanity)
   - Create a list of Thai profanity words to be censored.
4. Model Architecture:
   - Use a deep learning model suitable for audio processing, such as:
     - Convolutional Neural Networks (CNNs)
     - Recurrent Neural Networks (RNNs) like LSTM or GRU
     - Transformer-based models
   - Or using Existing model: SpeechBrain, Wav2Vec 2.0, Keras Audio Classification Model.
5. Training:
   - Split the dataset into training, validation, and test sets.
   - Train the model to identify profanity in the audio samples.
6. Evaluation and Iteration:
   - Test the model on the test set and real-world samples.
   - Iterate and refine the model based on performance.
7. Post-processing:
   - Implement a method to censor or bleep out the identified profanity.

ผม train A.I. โดยใช้หลักการของอาจารย์ waveform analysis approach ที่จะทำการตรวจจับคำทีละคำในรูปแบบของคลื่นเสียงโดยตรงตามที่อาจารย์ให้คำแนะนำซื้งเท่าที่ผมหายังไม่มี model ภาษาไทย และ model ส่วนใหญ่เป็น text ครับ ผมเลยพยายาม train A.I. โดยใช้ model ที่มีอยู่แล้วอย่าง SpeechBrain ระหว่างนี้ที่รอ A.I. train ผมก็หาวิธีการอื่นเพิ่มเติมเหมือนกันรวมถึงลองวิธี speech to text ด้วยครับ ซึ่ง data ที่ผมใช้นั้นยังมีแค่เสียงของผม เพราะว่าอยากทราบในเบื่องต้นก่อนว่าการที่ train ผ่านคนๆ คนเดียว มันได้ผลลัพธ์ที่ตรงตามที่ต้องการก่อนที่จะขอรวบรวม data จากคนอื่นๆ เพิ่มเติมครับ อันนี้เป็น progression ที่ผมไปศึกษาและลองทำมาทั้งหมดครับ

So, I am doing first one is waveform analysis approach which will detect word by word in audio format directly as you advised. 

## Technologies Used

- OpenCV / FFmpeg for video processing
- PyDub / Librosa for audio processing
- Google Speech Recognition API or CMU Sphinx for speech-to-text
- PyThaiNLP for Thai language processing
- Custom profanity detection algorithm
- Librosa for waveform analysis

## Setup and Installation

(Add instructions for setting up the project environment and installing dependencies)

## Usage

(Add instructions on how to use the program, including input/output formats and any command-line arguments)

## Future Improvements

- Enhance speech recognition accuracy for Thai language
- Improve profanity detection using machine learning techniques
- Optimize performance for processing large video files
- Implement real-time processing capabilities

## Contributors

(Add your name and any other contributors)

## License

(Add appropriate license information)
