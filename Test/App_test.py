#Importing the required packages


import sys

sys.path.insert(0, 'Main')

import Main


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

    @pytest.mark.parameterize("a,b,c,d,o",[('15,1.2,80,82','P')])
    def test_function(self,a,b,c,d,o):
        assert Main.Random_Forest_Hyp1(a,b,c,d) == o

    # Main.Random_Forest_Hyp2()

