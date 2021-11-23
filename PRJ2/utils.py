import sys
import torch

def bool_input(prompt, default=True):
    resp = input(prompt)
    sys.stdout.write(resp + '\n')
    if default:
        return not resp.lower() == 'n'
    else:
        return resp.lower() == 'y'

def str_input(prompt, default=''):
    resp = input(prompt)
    sys.stdout.write(resp+'\n')
    return resp if len(resp) > 0 else default

def float_input(prompt, default=0.0):
    resp = input(prompt)
    sys.stdout.write(resp+'\n')
    return float(resp) if len(resp) > 0 else default

def val_categorical(y_pred, target):
    preds_tensor = torch.max(y_pred, 1)
    return (preds_tensor == target)


class ConsoleLogging(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open('console-logs/' + filename + '.txt', 'a')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
        pass

    def start(filename):
        sys.stdout = ConsoleLogging(filename)
    
    def stop():
        sys.stdout.file.close()
        sys.stdout = sys.stdout.terminal
