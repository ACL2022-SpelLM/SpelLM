import sys
print(sys.path)
from work import trainer

if __name__ == '__main__':

    train = trainer.Trainer()
    train.evaluate()
