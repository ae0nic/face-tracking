from landmarker.landmark_runner import Landmarker
from panda.panda_runner import MyApp

with Landmarker() as landmarker:
    app = MyApp(landmarker)
    app.run()
