#############################
Prepare the  Base-BERT Model

1.Click https://huggingface.co/bert-base-chinese#
2.Click the download icon in "Files and Versions" to download "pytorch_model.bin"
3.move "pytorch_model.bin" to the path "model/bert-base-chinese"


#############################
How to run?

Requirements:
1.Install Anaconda
2.RUN:
conda create -n spelLM python=3.6
conda activate spelLM
conda install pytorch-gpu==1.2.0
conda install scikit-learn==0.23.1
conda install tqdm
pip install transformers==3.0.0
-----------------------------------------------------------------------
The training are divided into three steps：

STEP1: Fine-tune the classifier model
RUN:
cd train
nohup python -u fine_tune_classifier.py>log/log_tune_classifier.txt 2>&1 &
-----------------------------------------------------------------------
STEP2: Train Our Model
RUN:
nohup python -u train.py>log/log_train.txt 2>&1 &  
(Model will be saved in "model/bert-base-chinese-q-layer")
-----------------------------------------------------------------------
STEP3: Evaluate Our Model
--Character-level-metric
RUN:
nohup python -u test.py -with_error==True >log/log_test_Character.txt 2>&1 &
(-with_error==True means character-level. The result will be saved in "test_out/".The evaluation-metric will be saved in "train/metric_results_**.txt" and printed in "log/log_test_Character.txt")

--Sentence-level-metric
RUN:
nohup python -u test.py -with_error==False >log/log_test_Sentence.txt 2>&1 &
(-with_error==False means sentence-level. the result will be saved in "test_out/".the evaluation-metric will be saved in "train/metric_results_**.txt" and printed in "log/log_test_Sentence.txt")

