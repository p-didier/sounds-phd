from sacred import Experiment
import time

t0 = time.time()
ex = Experiment('hello_world')
print(time.time() - t0)

@ex.config
def my_config():
    recipient = "world"
    message = "Hello %s!" % recipient

@ex.automain
def my_main(message):
    print(message)
