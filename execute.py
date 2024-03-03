# -*- coding: utf-8 -*-
import subprocess
import os, sys, time
import signal

# import from local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from opt import opt
os.environ['CUDA_VISIBLE_DEVICES'] = opt['CUDA_VISIBLE_DEVICES']  #'0,1'



# Essentional Command we want to input as automatic training
command_line = 'python train_code/train.py' # --auto_resume_closest


idx = 0
while idx < 10: 
    # if fails for ten times, it is not necessary to do any more
    """However, you should be careful with the '.wait()'"""
    
    # check if "--auto_resume_closest" exists
    if idx > 0:
        if command_line.find("--auto_resume_closest") == -1:
            command_line += ' --auto_resume_closest'

    # Subprocess Handling
    start = time.time()
    p = subprocess.Popen(command_line, shell=True).wait()       # Subprocess
    print("The return status is ", p)
    total_time = time.time() - start
    if total_time < 200:
        print("There must have some simple bug exists that lead the program end so soon. Better to check now!")
        os._exit(0)



    """#if your there is an error from running 'my_python_code_A.py', 
    the while loop will be repeated, 
    otherwise the program will break from the loop"""
    if p != 0:
        print("The Program End Abnormally, and we will restart it again")
        idx += 1
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        os.sleep(7)
        continue
    else:
        break

# Try to save the weight and run mass_metrics_test automatically
description = opt["description"]
os.system("mkdir " + description)
os.system("cp saved_models/*.pth " + description)
os.system("cp -r runs " + description)
os.system("cp -r " + description + " ../saved_training_anime/")
os.system("python test_code/mass_metrics_test.py ")