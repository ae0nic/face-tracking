from direct.stdpy import threading2 as threading

import landmark_runner
import panda_runner

data_queue = []


# app = panda_runner.MyApp()
# panda_thread = threading.Thread(target=app.run)
# panda_thread.start()
landmark_thread = threading.Thread(target=landmark_runner.run, args=[data_queue])
landmark_thread.start()
# landmark_runner.run(data_queue)
while True:
    pass
    # if len(data_queue):
    #     print(data_queue.pop(0))


