# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:39:33 2019

@author: bill-
"""
import random as rd
import string

#A short randomizer for a fake transaction ID and other unique ID values
#choice: draws with repeating values from a list
#sample only draws without repetitions
#length refers to numerical draw and to alphabetical draw alike

def random_string(length):
    letters = string.ascii_letters
    numbers = string.digits
    result_str = ''.join(rd.choice(letters) for n in range(length))
    result_num = ''.join(rd.choice(numbers) for n in range(length))
    trans_id = ''.join(result_str) + ''.join(result_num)
    return trans_id

print("The randomly generated string is", random_string(length = 5))
