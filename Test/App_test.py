#Importing the required packages
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, 'Main')

from Main import model_call


class test_App:
    
    #Calling Picke Files
    #Main.intro()

    # Main.Naive_Bayes1()

    # Main.Naive_Bayes2()

    # Main.KNN1()

    # Main.KNN2()

    # Main.SVM1()

    # Main.SVM2()

    # Main.Decision_Tree1()

    # Main.Decision_Tree2()

    # Main.Decision_Tree_Hyp1()

    # Main.Decision_Tree_Hyp2()

    # Main.Random_Forest1()

    # Main.Random_Forest2()

    def test_Random_Frest_Hyp1():
        with patch('builtins.input', side_effect=[1.5,1.2,80,82]):
            result = model_call.Random_Forest_Hyp1()
            assert result == 'N'
        #result=model_call.Random_Forest_Hyp1()

    # Main.Random_Forest_Hyp2()
