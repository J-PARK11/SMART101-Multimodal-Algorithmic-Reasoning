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
    
    For this demo, we provide a pretrained ResNet-50 + BERT pre-trained model trained
    on the SMART-101 dataset in the puzzle_split mode using the code in the above repo.
    This model is provided in ./checkpoints/ckpt_resnet50_bert_212.pth
    
    See scripts.sh file for the command lines to train the model on SMART-101 dataset and how to run the trained model on the VLAR challenge
    val and test datasets. 
    
    Specifically, note that the VLAR-val.json and VLAR-test.json files containing the VLAR challenge puzzles
    are assumed to be kept in /dataset/ folder, and a method should write the responses to /submission/submission.json
    as described in make_predictions() below. 
    
    Note
    ----
    In this demo, we do not use the answer candidate options within the model. However, 
    a user may chose to have additional inputs to the model for taking in the options.
