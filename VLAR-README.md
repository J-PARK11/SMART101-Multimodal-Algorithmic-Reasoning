    VLAR Challenge: Submission Code Demo.
    ************************************
   
    This code is modified from the SMART-101 code here: https://github.com/merlresearch/SMART
 
    This is a starter code demonstrating how to format your pre-trained model for evaluation. See solve_VLAR.py for details. 
    This code shows where to expect to read the test/val puzzles from, and how to produce 
    the output, which can be evaluated using our VLAR challenge evaluation code. 
    

    Please see the predict_on_challenge_data() function below for details. Formally, the code shows four steps:
        1) To read the puzzles (see data_loader.py, SMART_Challenge_Data class)
        2) Get/load the prediction/solution model: (see solve_VLAR.py: get_SMART_solver_model())
        3) Run the prediction model on the test puzzles and collect responses: (see solve_VLAR.py : make_predictions()
        4) Collect the responses in a json file for evaluation: (see solve_VLAR.py: make_response_json()
    
    For this demo, we provide a pretrained ResNet-50 + BERT model trained
    on the SMART-101 dataset in the puzzle_split mode using the code in the above repo.
    This model needs to be downloaded from the below link and copied into ./checkpoints/ckpt_resnet50_bert_212.pth  
    Checkpoint download link: https://www.dropbox.com/s/69ocrjwwccb6fqv/ckpt_resnet50_bert_212.pth?dl=0
    
    See scripts.sh file for the command lines to train the model on SMART-101 dataset and how to run the trained model on the VLAR challenge
    val and test datasets. 
    
    Specifically, note that the VLAR-val.json and VLAR-test.json files containing the VLAR challenge puzzles
    are assumed to be kept in /dataset/ folder, and a method should write the responses to /submission/submission.json
    as described in make_predictions() in solve_VLAR.py. 
    
    Note
    ----
    In this demo, we do not use the answer candidate options within the model. However, 
    a user may chose to have additional inputs to the model for taking in the options.


    Useful Docker commands for preparing the docker images for upload to eval.ai
    ----------------------------------------------------------------------------
    This document guides you on how to prepare the docker image for submission to the VLAR challenge. 

    We assume you have the solver_VLAR.py file that is available in our starter code. 
    This file implements a SMART-101 CVPR 2023 paper and is modified for the VLAR challenge.
    The steps for preparing the docker image are as follows and is inspired from the tutorials described here: https://docs.docker.com/language/python/build-images/

    # create a virtual env
    $ python3 -m venv .vlar
    $ python3 -m pip freeze > requirements.txt

    We assume the val/test files are in /dataset (which is available only when the code is run on eval.ai). 
    The val and test consists of /dataset/VLAR-val.json and /dataset/VLAR-test.json files, respectively, and the associated puzzle images are in /dataset/test-images/. 

    # Prepare the Dockerfile. This is how my Dockerfile looks like
    # Use the Dockerfile in the github starter code and run the following commands for creating the docker image.

    # build the docker image.
    $ docker build -t submission .
    $ docker tag submission:latest submission:1001 

    This above command will create a docker image for the starter illustrative example. It took me nearly 5 min to build this docker image and was about 6.5GB in size. 
    It includes the torch packages, numpy, etc.

    # The code when run at eval.ai will use following command using the docker image.
    $ docker run -v <path/to/val_or_test_data/on/eval.ai>/dataset:/dataset -v <path/to/val_or_test_data/on/eval.ai>/submission:/submission submission


    The above command will map the local path to /dataset and run the docker image named submission. 
    The output will be a file submission.json that should appear in <path/to/val_or_test_data/on/eval.ai>/submission/submission.json if everything goes well.

    Note: For dev phase, you need to read the puzzles from VLAR-val.json and for the test phase you need to read them from VLAR-test.json. 

    Eval AI submission:
    Follow the instructions here: https://eval.ai/web/challenges/challenge-page/2088/submission

    $ pip install evalai
    $ evalai set_token <your token>
    $ evalai push <image>:<tag> --phase smart-101-vlar-dev2023-2088 –private  # for dev

    OR

    $ evalai push <image>:<tag> --phase smart-101-vlar-test2023-2088 –private  # for test


    Here, <image> is the name of the docker image you have (“submission” in our example) and tag is the docker image tag (“1001” in our example), <phase_name> is the unique phase id you have, e.g.,.

    It took about 10 min for the model to be pushed to eval.ai for me. 

    Use 
    $ evalai submission <submission id> to check the status of the submission. Here, submission id is the id provided by evalai when making the docker image submission. 


    Other useful commands:
    To remove a docker image, use $ docker rmi -f <docker image id>
