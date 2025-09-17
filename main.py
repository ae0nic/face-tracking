from direct.stdpy import threading2 as threading

import landmark_runner
import panda_runner

data_queue = []


app = panda_runner.MyApp()
panda_thread = threading.Thread(target=app.run)
panda_thread.start()
while True:
    pass
    # if len(data_queue):
    #     print(data_queue.pop(0))


