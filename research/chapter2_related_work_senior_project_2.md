# Chapter 2: Related Work

## 2.1 Speech-Based Profanity Detection Systems

Profanity detection in speech has emerged as a significant research area in natural language processing (NLP) and speech processing, driven by the increasing need for automated content moderation across digital platforms. Existing methods generally fall into three main categories: Automatic Speech Recognition (ASR)-based text processing, hybrid approaches, and direct speech-based profanity detection using deep learning (Baevski et al., 2020; Wazir et al., 2022).

### 2.1.1 ASR-Based Approaches

Traditional approaches to speech-based profanity detection typically involve a two-step process: first converting speech to text using ASR systems, then applying text-based profanity detection methods. While computationally efficient, these approaches suffer from error propagation, where ASR transcription errors directly impact detection accuracy (Baevski et al., 2020). This limitation is particularly pronounced for low-resource languages like Thai, where ASR performance is generally lower due to limited training data and linguistic complexity.

### 2.1.2 Direct Speech-Based Detection

Recent advances have focused on direct analysis of speech signals without intermediate text conversion. Wazir et al. (2022) demonstrated the effectiveness of deep learning models for inappropriate speech content detection in film censorship applications, showing that direct speech analysis can outperform ASR-based approaches. Their work highlighted the importance of temporal feature extraction and the impact of model architecture choices on detection performance, establishing the foundation for end-to-end speech-based profanity detection systems.

Wav2Vec2 architecture offers significant advantages over ASR-based methods for profanity detection through its direct waveform analysis capabilities. Unlike ASR systems that convert speech to discrete text tokens, Wav2Vec2 processes raw audio waveforms directly, enabling precise temporal localization of acoustic features (Baevski et al., 2020). This direct analysis approach allows for fine-grained windowing strategies that can capture the exact temporal boundaries of profane words, which is crucial for accurate detection and subsequent censoring applications. The model's ability to analyze speech at various temporal resolutions through configurable window sizes makes it particularly suitable for profanity detection tasks where precise temporal localization is essential for practical deployment.

## 2.2 Evaluation Methodologies in Speech Processing

### 2.2.1 Traditional Evaluation Approaches

Conventional speech processing systems typically employ fixed evaluation parameters without systematic exploration of their impact on performance. These approaches often use standard configurations borrowed from speech recognition tasks, which may not be optimal for specialized applications like profanity detection (Park & Glass, 2008).

### 2.2.2 Word-Level and Temporal Evaluation

More sophisticated evaluation approaches focus on word-level accuracy and temporal precision. These methods, borrowed from object detection and speech recognition domains, use metrics such as Intersection over Union (IoU) to assess temporal alignment between predictions and ground truth labels (Everingham et al., 2010). The adaptation of these techniques to speech-based profanity detection requires careful consideration of linguistic boundaries and semantic units, particularly for languages with complex phonetic structures like Thai.

### 2.2.3 Multi-Method Evaluation Frameworks

Recent research has emphasized the importance of using multiple evaluation methods to comprehensively assess model performance. Different metrics may capture different aspects of system behavior, and optimal parameters for one evaluation method may not be optimal for others. This necessitates systematic analysis across multiple evaluation approaches to identify robust parameter configurations.

## 2.3 Parameter Optimization in Speech Processing

### 2.3.1 Window Size and Stride Configuration

The choice of temporal parameters—particularly window size and stride—significantly impacts the performance of speech processing systems. In speech recognition, optimal window sizes typically range from 20-40ms for phoneme-level analysis, while larger windows (hundreds of milliseconds to seconds) are used for word or utterance-level tasks (Povey et al., 2011). However, limited research has systematically explored these parameters specifically for profanity detection tasks, creating a gap in understanding optimal configurations for this domain.

### 2.3.2 Correlation Analysis in Parameter Selection

Statistical correlation analysis has been used in various speech processing applications to understand the relationship between system parameters and performance metrics. These approaches help identify optimal parameter ranges and provide insights into system behavior across different configurations (Ruder et al., 2019). The application of correlation analysis to temporal parameter optimization in profanity detection represents an underexplored area with significant practical implications.

### 2.3.3 Systematic Parameter Exploration

While individual studies have reported performance results for specific parameter configurations, comprehensive systematic exploration of parameter spaces remains limited. Most existing work uses fixed parameter settings without exploring the impact of different configurations on performance, limiting the generalizability of findings and practical guidance for practitioners.

## 2.4 Thai Language Processing Challenges

### 2.4.1 Linguistic Characteristics

Thai presents unique computational challenges due to its tonal nature, complex phonetic structure, and lack of explicit word boundaries in written text (Wutiwiwatchai & Furui, 2007). These characteristics directly impact speech processing systems, requiring specialized approaches for effective analysis. The tonal nature of Thai means that subtle variations in pitch can change word meanings, adding complexity to automatic processing systems and potentially affecting optimal parameter choices.

### 2.4.2 Phonetic Complexity and Parameter Impact

The complex phonetic structure of Thai, including its tonal characteristics and phoneme variations, suggests that optimal temporal parameters may differ significantly from those established for non-tonal languages (Theera-Umpon et al., 2011). This linguistic specificity necessitates dedicated parameter optimization studies for Thai language applications rather than adopting configurations optimized for other languages.

### 2.4.3 Limited Dataset Availability

Thai language processing suffers from limited availability of large-scale labeled datasets, particularly for specialized tasks like profanity detection. This constraint necessitates efficient approaches to model development and evaluation that can provide maximum insight with minimal data requirements, making parameter optimization studies particularly valuable for maximizing the utility of available data.

## 2.5 Transfer Learning and Pre-trained Models

### 2.5.1 Wav2Vec2 Architecture

The Wav2Vec2 framework has demonstrated significant success in speech processing tasks through self-supervised pre-training followed by task-specific fine-tuning (Baevski et al., 2020). The availability of Thai-specific pre-trained models, such as airesearch/wav2vec2-large-xlsr-53-th, provides a foundation for developing Thai speech processing applications despite data limitations. However, optimal fine-tuning parameters for specific tasks like profanity detection require dedicated investigation.

### 2.5.2 Fine-tuning Strategies and Parameter Sensitivity

Effective fine-tuning of pre-trained models requires careful consideration of various parameters, including temporal segmentation settings. Recent work has shown that proper parameter configuration can achieve competitive performance even with limited labeled data (Kenton & Toutanova, 2019), making systematic parameter optimization particularly crucial for low-resource language applications.

## 2.6 Evaluation Systems and Frameworks

### 2.6.1 Standardization Challenges

The lack of standardized evaluation frameworks for speech-based profanity detection makes it difficult to compare approaches and establish best practices. This gap is particularly pronounced for low-resource languages where benchmark datasets are scarce, highlighting the need for comprehensive evaluation systems that can be adopted by the research community.

### 2.6.2 Practical Deployment Considerations

While academic research often focuses on novel algorithms, practical deployment of profanity detection systems requires consideration of computational efficiency, real-time processing requirements, and parameter sensitivity. Evaluation frameworks that consider these practical aspects are essential for bridging the gap between research and application.

## 2.7 Gaps in Current Research

### 2.7.1 Systematic Parameter Analysis for Thai

Limited research has focused specifically on systematic parameter optimization for Thai language profanity detection. The unique linguistic characteristics of Thai suggest that optimal parameters may differ from those established for other languages, necessitating language-specific analysis that has not been comprehensively addressed in existing literature.

### 2.7.2 Multi-Method Evaluation Comparison

While individual evaluation methods have been studied in isolation, comprehensive comparison of different evaluation approaches with systematic parameter exploration remains limited. Understanding how different evaluation methods respond to parameter changes is crucial for developing robust evaluation frameworks.

### 2.7.3 Practical Parameter Guidance

Most existing research provides limited practical guidance for parameter selection in real-world applications. The gap between academic research and practical implementation necessitates studies that provide concrete, evidence-based recommendations for system configuration.

## 2.8 Research Positioning

This study addresses the identified gaps by providing a systematic analysis of temporal parameters (window size and stride configurations) specifically for Thai profanity detection across multiple evaluation methods. By focusing on evaluation methodology and parameter optimization rather than novel model architectures, this work provides practical guidance that can be immediately applied by future researchers and practitioners.

The comprehensive correlation analysis across multiple evaluation methods fills a critical gap in understanding the relationship between system parameters and detection performance in the Thai language context. This approach provides evidence-based recommendations for optimal parameter selection, addressing the practical needs of developers working with limited computational resources and datasets.

Furthermore, the development of a standardized evaluation framework with systematic parameter exploration contributes to the establishment of best practices for Thai speech processing applications, potentially accelerating future research and development in this domain.
