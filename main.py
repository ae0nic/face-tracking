from landmarker.landmark_runner import Landmarker
from panda.panda_runner import MyApp

data_queue = []

with Landmarker(queue=data_queue) as landmarker:
    app = MyApp(data_queue)
    while True:
        landmarker.run()
        app.taskMgr.step()

