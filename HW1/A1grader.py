import os
import copy
import signal

# Code to limit running time of specific parts of code.
#  To use, do this for example...
#
#  signal.alarm(seconds)
#  try:
#    ... run this ...
#  except TimeoutException:
#     print(' 0/8 points. Your depthFirstSearch did not terminate in', seconds/60, 'minutes.')
# Exception to signal exceeding time limit.


# class TimeoutException(Exception):
#     def __init__(self, *args, **kwargs):
#         Exception.__init__(self, *args, **kwargs)


# def timeout(signum, frame):
#     raise TimeoutException

# seconds = 60 * 5

# Comment next line for Python2
# signal.signal(signal.SIGALRM, timeout)

import os
import numpy as np

print('\n======================= Code Execution =======================\n')

assignmentNumber = '1'

import subprocess, glob, pathlib
nb_name = '*-A{}.ipynb'
# nb_name = '*.ipynb'
filename = next(glob.iglob(nb_name.format(assignmentNumber)), None)
print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
if not filename:
    raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
with open('notebookcode.py', 'w') as outputFile:
    subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                     nb_name.format(assignmentNumber), '--stdout'], stdout=outputFile)
# from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
import sys
import ast
import types
with open('notebookcode.py') as fp:
    tree = ast.parse(fp.read(), 'eval')
print('Removing all statements that are not function or class defs or import statements.')
for node in tree.body[:]:
    if (not isinstance(node, ast.FunctionDef) and
        not isinstance(node, ast.Import)):  #  and
        # not isinstance(node, ast.ClassDef) and
        # not isinstance(node, ast.ImportFrom)):
        tree.body.remove(node)
# Now write remaining code to py file and import it
module = types.ModuleType('notebookcodeStripped')
code = compile(tree, 'notebookcodeStripped.py', 'exec')
sys.modules['notebookcodeStripped'] = module
exec(code, module.__dict__)
# import notebookcodeStripped as useThisCode
from notebookcodeStripped import *


if False:
    from A1mysolution import *
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")

g = 0

for func in ['rmse', 'gradient_descent_adam',
             'linear_model', 'linear_model_gradient',
             'quadratic_model', 'quadratic_model_gradient',
             'cubic_model', 'cubic_model_gradient',
             'quartic_model', 'quartic_model_gradient']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')


            
print('''\nTesting
  X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
  W = np.array([1, 2]).reshape(-1, 1)
  Y = linear_model(X, W)''')

X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
W = np.array([1, 2]).reshape(-1, 1)

try:
    pts = 10
    Y = linear_model(X, W)
    correct_Y = np.array([[ 3], [ 5], [7], [ 9], [11], [17], [19], [23]])

    if np.allclose(Y, correct_Y, 1e-2):
        g += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Y should be')
        print(correct_Y)
except Exception as ex:
    print(f'\n--- 0/{pts} points. linear_model raised the exception\n')
    print(ex)

######################################################################

print('''\nTesting
  X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
  W = np.array([1, 2, -3]).reshape(-1, 1)
  Y = quadratic_model(X, W)''')

X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
W = np.array([1, 2, -3]).reshape(-1, 1)

try:
    pts = 10
    Y = quadratic_model(X, W)
    correct_Y = np.array([[   0], [  -7], [ -20], [ -39], [ -64], [-175], [-224], [-340]])

    if np.allclose(Y, correct_Y, 1e-2):
        g += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Y should be')
        print(correct_Y)
except Exception as ex:
    print(f'\n--- 0/{pts} points. quadratic_model raised the exception\n')
    print(ex)

######################################################################

print('''\nTesting
  X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
  W = np.array([1, 2, -3, 1.5).reshape(-1, 1)
  Y = cubic_model(X, W)''')

X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
W = np.array([1, 2, -3, 1.5]).reshape(-1, 1)

try:
    pts = 10
    Y = cubic_model(X, W)
    correct_Y = np.array([[1.5000e+00], [5.0000e+00], [2.0500e+01], [5.7000e+01], [1.2350e+02], [5.9300e+02], [8.6950e+02], [1.6565e+03]])

    if np.allclose(Y, correct_Y, 1e-2):
        g += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Y should be')
        print(correct_Y)
except Exception as ex:
    print(f'\n--- 0/{pts} points. cubic_model raised the exception\n')
    print(ex)


######################################################################

print('''\nTesting
  X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
  W = np.array([1, 2, -3, 1.5, 0.3]).reshape(-1, 1)
  Y = quartic_model(X, W)''')

X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
W = np.array([1, 2, -3, 1.5, 0.3]).reshape(-1, 1)

try:
    pts = 10
    Y = quartic_model(X, W)
    correct_Y = np.array([[1.8000e+00], [9.8000e+00], [4.4800e+01], [1.3380e+02], [3.1100e+02], [1.8218e+03], [2.8378e+03], [6.0488e+03]])

    if np.allclose(Y, correct_Y, 1e-2):
        g += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Y should be')
        print(correct_Y)
except Exception as ex:
    print(f'\n--- 0/{pts} points. quartic_model raised the exception\n')
    print(ex)

######################################################################


print('''\nTesting
  X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
  T = (((X - 5) * 0.05 +  (X * 0.2) **5) / 5.0 - 5.5) / 6
  W = np.array([1, 2, -3, 1.5, 0.3]).reshape(-1, 1)
  grad = quartic_model_gradient(X, T, W)''')

X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
T = (((X - 5) * 0.05 +  (X * 0.2) **5) / 5.0 - 5.5) / 6
W = np.array([1, 2, -3, 1.5, 0.3]).reshape(-1, 1)

try:
    pts = 10
    grad = quartic_model_gradient(X, T, W)
    correct_grad = np.array([[2.80354597e+03], [2.72272649e+04], [2.72182945e+05], [2.77520312e+06], [2.87167331e+07]])
    if np.allclose(grad, correct_grad, 1e-2):
        g += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. grad should be')
        print(correct_grad)
        print('Your function returned')
        print(grad)
        
except Exception as ex:
    print(f'\n--- 0/{pts} points. quartic_model_gradient raised the exception\n')
    print(ex)

######################################################################


print('''\nTesting
   X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
   T = (((X - 5) * 0.05 +  (X * 0.2) **5) / 5.0 - 5.5) / 6
   W = np.zeros((3, 1))
   W, error_sequence, W_sequence = gradient_descent_adam(
       quadratic_model, quadratic_model_gradient, rmse, X, T, W, rho=1e-3, n_steps=1000)''')

X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
T = (((X - 5) * 0.05 +  (X * 0.2) **5) / 5.0 - 5.5) / 6
W = np.zeros((3, 1))

try:
    pts = 15
    W, error_sequence, W_sequence = gradient_descent_adam(
        quadratic_model, quadratic_model_gradient, rmse, X, T, W, rho=1e-3, n_steps=10000)
    err = error_sequence[-1]
    correct_err = 0.1137
    if np.allclose(err, correct_err, 1e-2):
        g += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Final error should be')
        print(correct_err)
        print('Your function returned')
        print(err)
        
except Exception as ex:
    print(f'\n--- 0/{pts} points. gradient_descent_adam raised the exception\n')
    print(ex)

######################################################################    

    
print('''\nTesting
   X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
   T = (((X - 5) * 0.05 +  (X * 0.2) **5) / 5.0 - 5.5) / 6
   W = np.zeros((4, 1))
   W, error_sequence, W_sequence = gradient_descent_adam(
       cubic_model, cubic_model_gradient, rmse, X, T, W, rho=1e-3, n_steps=10000)''')

X = np.array([1, 2, 3, 4, 5, 8, 9, 11]).reshape((-1, 1))
T = (((X - 5) * 0.05 +  (X * 0.2) **5) / 5.0 - 5.5) / 6
W = np.zeros((4, 1))

try:
    pts = 15
    W, error_sequence, W_sequence = gradient_descent_adam(
        cubic_model, cubic_model_gradient, rmse, X, T, W, rho=1e-3, n_steps=10000)
    err = error_sequence[-1]
    correct_err = 0.0184
    if np.allclose(err, correct_err, atol=1e-2):
        g += pts
        print(f'\n--- {pts}/{pts} points. Returned correct values.')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values. Final error should be')
        print(correct_err)
        print('Your function returned')
        print(err)
        
except Exception as ex:
    print(f'\n--- 0/{pts} points. gradient_descent_adam raised the exception\n')
    print(ex)

    


name = os.getcwd().split('/')[-1]

print('\n{} Execution Grade is {} / 80'.format(name, g))

print('''
Your final assignment grade will be based on other tests.  Run additional tests
of your own design to check your functions before checking in this notebook.''')

print('\n{} FINAL GRADE is ___ / 100'.format(name))

# print('\n{} EXTRA CREDIT is   / 1'.format(name))

