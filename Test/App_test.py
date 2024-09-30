#Importing the required packages
import unittest
from unittest.mock import patch
import sys

sys.path.insert(0, 'Main')

from Main import Random_Forest_Hyp1


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

    @patch('builtins.input', side_effect=['Y' or 'N'])
    def test_Random_Forest_Hyp1(self, mock_input):
            Random_Forest_Hyp1()
        #result=model_call.Random_Forest_Hyp1()

    # Main.Random_Forest_Hyp2()
