# Turn Detection & Audio Model Fine-Tuning

Data Source -> [Multi turn Multi Speaker Dataset](https://www.kaggle.com/datasets/washingtongold/voxconverse-dataset)

[Github Link](https://github.com/erishan6/apna_voice_ds_assignment/) for the code

Folder Structure
```
apna_voice_ds_assignment/
├── data/  # audio + annotations
├── turn_evaluation/
│   ├── PipecatCoreMLSmartTurnVADWrapper.py  # Wrapper class for using VAD
│   ├── task1_smartturn.py  # Task 1 smartturn v2
│   ├── task1_VAD.py        # Task 1 baseline model
│   └── task3.py            # Task 3 code
├── voice_streaming/
│   └──  task2a.py          # Task 2 Option A
└── README.md  # <200-word summary
```

This project explores turn detection and real-time voice synthesis using open datasets and 
open-source tools. 

For voice activity detection, I developed and compared two approaches: the traditional 
Silero VAD and an advanced SmartTurn system with end-of-turn detection capabilities. 
I created a custom wrapper class to handle real-time audio chunk processing for SmartTurn, 
configuring parameters like speech duration thresholds and implementing evaluation metrics 
including precision, recall, and latency against ground truth annotations. I specifically 
tested how different chunk sizes (256, 512, and 1024 samples) impacted SmartTurn's segment
detection accuracy.

For text-to-speech evaluation, I benchmarked two models - Glow-TTS and Tacotron2-DCA - 
analyzing both full-text synthesis and chunked processing in 40-character segments. 
I measured critical performance indicators including generation latency and audio quality 
using PESQ scores, with all results systematically exported to CSV files for analysis. 
Throughout both VAD and TTS evaluations, I implemented standardized assessment functions 
to ensure consistent metric calculations and created visual comparisons of the results 
through bar chart visualizations. This work demonstrates my approach to evaluating voice 
processing systems holistically, addressing both speech detection and synthesis components 
with quantifiable metrics relevant to real-time streaming applications.