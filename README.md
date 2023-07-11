<h2>VLAR Challenge: Submission Code Demo.</h2>
   
This is the starter code (modified from the SMART-101 code here: https://github.com/merlresearch/SMART) for demonstrating how to format your pre-trained model for evaluation in the VLAR-Challege on eval.ai. The code shows 
where to expect to read the test/val puzzles from on the eval.ai server and where to produce your prediction results (output of your model) so that it can be evaluated using our VLAR challenge evaluation code. See solve_VLAR.py for implementation details. 
    
For this starter code demo, we provide a ResNet-50 + BERT pre-trained model that is trained
on the SMART-101 dataset in the puzzle_split mode (see the paper for details on what this split mode is). After cloning the starter code, you will need to download the pretrained model from the link below and 
copy it to ./checkpoints/ckpt_resnet50_bert_212.pth  
Checkpoint download link: https://www.dropbox.com/s/69ocrjwwccb6fqv/ckpt_resnet50_bert_212.pth?dl=0

To ensure the code can be run on eval.ai correctly, we are providing a sample VLAR-val.json in ./dataset/ folder. 
This file shows what the format of a `dev` phase (or a test phase) input to your algorithm is. If your code runs on 
VLAR-val.json locally, it should run on eval.ai as well. See the instructions below on how to run your code in eval.ai. 
See solve_VLAR.py for the exact details.

Docker Image Preparation for Eval.AI submission 
------------------------------------------------
This document guides you to prepare your docker image using the starter code and the pre-trained model for submission to the VLAR challenge. These steps are inspired by the tutorials described here: https://docs.docker.com/language/python/build-images/
```
# create a virtual env
$ python3 -m venv .vlar
$ source .vlar/bin/activate
# follow the instructions in SMART101-Code-README.md to install the dependencies in requirements.txt
```
We assume the val/test files are in /dataset (which is available only when the code is run on eval.ai, although we provide a sample VLAR-val.json in the ./dataset folder in this github repo -- this VLAR-val.json has puzzles that are a subset of the puzzles used in VLAR-val.json used in the dev phase on eval.ai). 
The val and test consists of `/dataset/VLAR-val.json` and `/dataset/VLAR-test.json` files, respectively, and the associated puzzle images are in `/dataset/test-images/`. 
```
# Use the Dockerfile in the github starter code and run the following commands for creating the docker image.
$ docker build -t submission .
$ docker tag submission:latest submission:1001 
```
This above command will create a docker image for the starter illustrative example. It took me nearly 5 min to build this docker image and was about 6.5 GB in size. It includes the torch packages, numpy, etc.
```
# The code when run at eval.ai will use following command using the docker image.
$ docker run -v <path/to/val_or_test_data/on/eval.ai>/dataset:/dataset -v <path/to/val_or_test_data/on/eval.ai>/submission:/submission submission
```

The above command will map the local path to /dataset and run the docker image named submission. 
The output will be a file submission.json that should appear in `<path/to/val_or_test_data/on/eval.ai>/submission/submission.json` if everything goes well. You may test this command locally via using the provided VLAR-val.json to make sure everything works fine.

Note: For dev phase, you need to read the puzzles from VLAR-val.json and for the test phase you need to read them from VLAR-test.json. 

Eval AI submission:
Follow the instructions here: https://eval.ai/web/challenges/challenge-page/2088/submission
```
$ pip install evalai
$ evalai set_token <your token>
$ evalai push <image>:<tag> --phase smart-101-vlar-dev2023-2088 –private  # for dev

OR

$ evalai push <image>:<tag> --phase smart-101-vlar-test2023-2088 –private  # for test
```

Here, `<image>` is the name of the docker image you have (“submission” in our example) and tag is the docker image tag (“1001” in our example), `<phase_name>` is the unique phase id you have. It took about 10 min for the model to be pushed to eval.ai for me. To check the status of your run on eval.ai, either look in 'My Submissions' tab on eval.ai or use the command line:
```
$ evalai submission <submission id>
```
to check the status of the submission. Here, submission id is the id provided by evalai when making the docker image submission. 

* Useful commands:
To remove a docker image, use `$ docker rmi -f <docker image id>`
