<h2>VLAR Challenge: Submission Instructions and Demo</h2>
   
This repo provides the submission instructions and a starter code package (modified from the SMART-101 code here: https://github.com/merlresearch/SMART) demonstrating how to format your pre-trained model for evaluation in the VLAR-Challege on eval.ai. The code shows 
where to expect to read the test/val puzzles from on the eval.ai server and where to produce your prediction results (output of your model) so that it can be evaluated using our VLAR challenge evaluation code. See solve_VLAR.py for implementation details. 
    
For this starter code demo, we provide a ResNet-50 + BERT pre-trained model that is trained
on the SMART-101 dataset in the puzzle_split mode (see the paper for details on what this split mode is). After cloning the starter code, you will need to download the pretrained model bundle from the below link: https://www.dropbox.com/s/8uapjgh90eb4tus/vlar_checkpoints.zip?dl=0

To ensure the code can be run on eval.ai correctly, we are providing a sample VLAR-val.json in ./dataset/ folder. 
This file shows what the format of a `dev` phase (or a test phase) input to your algorithm is. If your code runs on 
VLAR-val.json locally, it should run on eval.ai as well. See the instructions below on how to run your code on eval.ai. 

## Participation Guidelines

Participate in the contest by registering on the [EvalAI challenge page](https://eval.ai/web/challenges/challenge-page/2088/overview) and creating a team. Participants will upload docker containers with their agents that are evaluated on an AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to ensure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Local Evaluation

1. Clone the challenge repository:
   You may use a Python virtual environment and install your packages into it.
   ```
   # create a virtual env
   $ conda create -n VLAR python=3.9
   $ conda activate VLAR
   $ git clone https://github.com/smartdataset/vlar23_starter_code.git
   $ cd vlar23_starter_code
   # follow the instructions in SMART101-Code-README.md to install the dependencies in requirements.txt
   ```
1. Implement your own model or try ours. We provide a simple baseline in `solve_VLAR.py` that uses pretrained ResNet-50 + BERT pre-trained model trained on SMART-101 dataset. To use this pre-trained model, run the following:
   ```
   $ wget -O vlar_checkpoints.zip  https://www.dropbox.com/s/8uapjgh90eb4tus/vlar_checkpoints.zip?dl=0
   $ unzip vlar_checkpoints.zip
   $ rm vlar_checkpoints.zip
   ```
1. Install [docker](https://docs.docker.com/engine/install/).
1. Modify the provided Dockerfile (`Dockerfile`) if you need custom modifications. Let’s say your code needs a custom checkpoint of a model you trained and needs `transformers` package, the dependencies should be pip installed and additional files should be explicitly added (see our Dockerfile in the repo for a sample):
    ```dockerfile
    # install dependencies using pip
    RUN pip install transformers
    ADD custom_ckpt.pth /ckpt.pth
    ```
    Build your docker container using: `docker build . --file Dockerfile -t smart_101_submission`.
    Note #1: you may need `sudo` privileges to run this command.
1. Evaluate your docker container locally (you may use the provided ./dataset/VLAR-val.json for the local run):
    ```bash
    # Testing on val split
    docker run -v /path/to/local/copy/of/dataset/:/dataset/ smart_101_submission
    ```
    If the above command runs successfully you will get an output similar to:
    ```
    2023-03-01 16:35:02,244 loading model ...
    2023-03-01 16:35:02,244 making predictions using the model
    2023-03-01 16:35:02,244 writing the model responses to file
    2023-03-01 16:35:02,244 done!
    ```
   If the docker runs correctly, the above step should have produced a file: $(pwd)/dataset/submission.json. Note that, we use a sleep of 5 minutes at the end of the evaluation for internal purposes. You do not need to wait for 5 minutes and and kill the program once 'done' is printed (as above). 
    Note: a similar command will be run to evaluate your submission for the leaderboard. **Please submit your docker for remote evaluation (below) only if it runs successfully on your local setup.** 
### Online submission
We have two separate eval.ai challenges: (i) SMART-101 CVPR 2024 Challenge for the test phase and (ii) another one for the dev phase. These challenges are entirely different and serve different purposes. A participant must create an eval.ai participant profile in (i) for participating in the challenge.

**Test Challenge phase**: This phase/split will be used to decide challenge winners. Each team is allowed a total of 10 submissions until the end of challenge submission phase. The highest performing of these 10 will be automatically chosen. Results on this split will not be made public until the announcement of final results at the [MAR workshop at CVPR 2024](https://marworkshop.github.io/cvpr24/index.html). **Please read the instructions below carefully for this test phase submission**. 
<br>Your submission will be evaluated on 100 puzzles and will have a total available time of 5 mins to finish.<br>
a. Create a participant profile on eval.ai at the SMART-101-Challenge-test website [here](https://eval.ai/web/challenges/challenge-page/2247/phases). You will receive a <participant id> after this step. <br>
b. Follow the steps to create a local docker image on your computer.<br>
c. Upload your docker image to a docker-sharing website (e.g., docker-hub, or make a tar ball and share via dropbox, for example). This step will produce a <docker-share-link>. Some useful instructions for sharing the docker are provided below.<br>
```
# using tar based sharing
docker save <docker-image-name>:<tag> > <my_submission>.tar
```
or 
```
# using docker-hub based sharing. You will need to create a docker-hub login and then run the following on your machine where the docker image exists.
docker login
docker tag <docker-image-name>:<tag> <docker-hub-username>/<docker-hub-repository-name>:<tag>
docker push <docker-hub-username>/<docker-hub-repository>:<tag>
```<br>
d. Fill this [form](https://docs.google.com/forms/d/e/1FAIpQLSd3cZMkPpQpxg1_WN6w5mb8WeWD15AQQnq4gsUo1Udk40MPrg/viewform?usp=sharing) with your eval.ai <participant id>, the <docker-share-link>, and the details regarding the compute needed for evaluating your submission. You should also provide an email address for correspondence.<br>
e. We will run your submission using the "docker run" commands described above against our test puzzles. We will send you an email with a "submission.json" attachment that contains the responses of your submission on our private test puzzles.<br>
f. The participant then needs to login to eval.ai SMART-101-Challenge-test website (at the link in step (a)), and submit this submission.json file at the test phase (tab). This step will evaluate the submission against the ground truth annotations. The score from the evaluation will be displayed on the leaderboard. <br> 
<br>
Please check SMART-101-Challenge-test website for the number of submissions allowed per day in the test phase. If you face any issues or have questions you can ask them by opening an issue on this repository or emailing us.

**Dev phase**: 
The purpose of this phase/split is sanity checking -- to confirm that our remote evaluation reports the same result as the one you’re seeing locally. Each team is allowed maximum of 100 submissions per day for this phase, but please use them judiciously. We will block and disqualify teams that spam our servers. This validation / dev phase will use a separate eval.ai dev challenge, using a smaller compute infrastructure in order to ensure we are able to run your test docker submissions (see below). Please follow the above steps for this dev phase to create a docker image on your local machine. Next, follow instructions in the `submit` tab of the EvalAI challenge page to submit your docker image. Note that you will need a version of EvalAI `>= 1.3.15`. Pasting those instructions here for convenience:
```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.3.15"
# Set EvalAI account token
evalai set_token <your EvalAI participant token>
# Push docker image to EvalAI docker registry
# Val phase
evalai push <image>:<tag> --phase smart-101-vlar-dev2023-2088
```
If you face any issues or have questions you can ask them by opening an issue on this repository or emailing us.
