import math

import direct
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock

from panda3d.core import Shader, DirectionalLight, PointLight, Vec3, LPoint3f, Thread

from panda3d.bullet import BulletWorld, BulletDebugNode
from panda3d.bullet import BulletSoftBodyNode

from VRoidModel import VRMLoader


class MyApp(ShowBase):
    key_map = {"w": False, "s": False, "a": False, "d": False, "e": False, "q": False,
               "arrow_left": False, "arrow_right": False, "arrow_down": False, "arrow_up": False,
               "debug": False, "debug_draw": 0}
    world = None
    def init_physics(self):
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(False)
        debugNP = self.render.attachNewNode(debugNode)
        debugNP.show()

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))
        self.world.setDebugNode(debugNP.node())

    def __init__(self):
        print(Thread.is_threading_supported())
        ShowBase.__init__(self)

        self.init_physics()

        self.x_text = None
        self.y_text = None
        self.z_text = None

        self.setBackgroundColor(0., 1., 0.)

        # Load the environment model.

        vrm_model = VRMLoader("./model.gltf", self)
        self.scene = vrm_model.body
        self.face = vrm_model.face

        body_shader = Shader.load(Shader.SL_GLSL,
                                  vertex="body.vert",
                                  fragment="body.frag")

        hair_shader = Shader.load(Shader.SL_GLSL,
                                  vertex="hair.vert",
                                  fragment="hair.frag")

        self.scene.setShader(body_shader)
        self.scene.setShaderInput("LIGHTS", 2)
        self.scene.setShaderInput("DEBUG_MODE", self.key_map["debug_draw"])

        vrm_model.hairs.setShader(hair_shader)

        joints = vrm_model.body.getJoints(jointName="*")


        d_light_node = DirectionalLight("d_light")
        d_light_node.setColor((1., 1., 1., 1))
        d_light = self.render.attachNewNode(d_light_node)
        d_light.setHpr(-21, 50, 0)
        self.scene.setLight(d_light)

        p_light_node = PointLight("p_light")
        p_light_node.setColor((1., 1., 1., 1))
        p_light = self.render.attachNewNode(p_light_node)
        p_light.setPos(8, -33, 20)
        self.scene.setLight(p_light)

        vrm_model.reparent(self.render)
        self.disableMouse()
        self.mouth_target = None

        vrm_model.rescale(12)
        vrm_model.position(0, -15, 0)
        self.scene.setH(180)
        self.camera.setY(-40)

        # print(vrm_model.body.getChild(0).getChild(1).getChildren())
        (pos, joints) = vrm_model.get_hair("*Head*", "HairJoint-")
        exposed_nodes = []
        rope_nodes = []
        for j, p in zip(joints, pos):
            origin = LPoint3f(p[0][0], p[0][1], p[0][2])
            end = LPoint3f(p[len(p) - 1][0], p[len(p) - 1][1], p[len(p) - 1][2])
            rope = BulletSoftBodyNode.makeRope(self.world.getWorldInfo(), origin, end, len(p) - 1, 1)
            rope.setTotalMass(50.0)
            np = self.render.attachNewNode(rope)
            np.reparentTo(vrm_model.head_joint)
            strand = []
            for joint in j:
                strand.append(vrm_model.control_joint(joint.getName()))
            exposed_nodes.append(strand)
            rope_nodes.append(rope.getNodes())
            self.world.attachSoftBody(rope)


        self.taskMgr.add(self.moveCamera, "MoveCamera")
        self.taskMgr.add(self.controlJoint, "ControlJoint", extraArgs=[vrm_model], appendTask=True)

        self.taskMgr.add(self.run_physics, 'update')
        self.taskMgr.add(self.update_hair, "updateHair", extraArgs=[exposed_nodes, rope_nodes, vrm_model.head_joint], appendTask=True)

        self.accept("g", self.gDown)

        self.accept("1", self.oneDown)
        self.accept("2", self.twoDown)
        self.accept("n", self.nDown)
        self.accept("0", self.zeroDown)

        self.accept("w", self.wDown)
        self.accept("s", self.sDown)
        self.accept("w-up", self.wUp)
        self.accept("s-up", self.sUp)
        self.accept("a", self.aDown)
        self.accept("d", self.dDown)
        self.accept("a-up", self.aUp)
        self.accept("d-up", self.dUp)
        self.accept("e", self.eDown)
        self.accept("q", self.qDown)
        self.accept("e-up", self.eUp)
        self.accept("q-up", self.qUp)
        self.accept("arrow_left", self.arrow_leftDown)
        self.accept("arrow_right", self.arrow_rightDown)
        self.accept("arrow_left-up", self.arrow_leftUp)
        self.accept("arrow_right-up", self.arrow_rightUp)
        self.accept("arrow_down", self.arrow_downDown)
        self.accept("arrow_up", self.arrow_upDown)
        self.accept("arrow_down-up", self.arrow_downUp)
        self.accept("arrow_up-up", self.arrow_upUp)
        self.face_joint = None

    def run_physics(self, task):
        dt = globalClock.getDt()
        self.world.doPhysics(dt)
        return task.cont

    def update_hair(self, hair, nodes, parent, task):
        # print([len(h) for h in hair])
        # print([len(n) for n in nodes])
        for strand in hair:
            for j in strand:
                j.setH((math.sin(task.time * 5) + 1) * 90)

        # for strand, rope in zip(hair, nodes):
        #     for i in range(len(strand)):
        #         if i == 0:
        #             strand[i].lookAt(parent, rope[i + 1].getPos())
        #         else:
        #             strand[i].lookAt(strand[i-1], rope[i + 1].getPos())
        #         # print(strand[i].getHpr())

        return direct.task.Task.cont

    def controlJoint(self, model: VRMLoader, task):
        model.get_morph_target("29").setX((math.sin(task.time * 5) + 1) * 0.5)
        model.control_joint("HairJoint-1906a1ce-1b58-4a73-8500-32a1e759a35c").setX((math.sin(task.time * 5) + 1) * 90)

        return direct.task.Task.cont

    def gDown(self):
        self.key_map["debug"] = not self.key_map["debug"]

    def oneDown(self):
        self.key_map["debug_draw"] = 1
        self.scene.setShaderInput("DEBUG_MODE", self.key_map["debug_draw"])

    def twoDown(self):
        self.key_map["debug_draw"] = 2
        self.scene.setShaderInput("DEBUG_MODE", self.key_map["debug_draw"])

    def nDown(self):
        self.key_map["debug_draw"] = 9
        self.scene.setShaderInput("DEBUG_MODE", self.key_map["debug_draw"])

    def zeroDown(self):
        self.key_map["debug_draw"] = 0
        self.scene.setShaderInput("DEBUG_MODE", self.key_map["debug_draw"])

    def wDown(self):
        self.key_map["w"] = True

    def wUp(self):
        self.key_map["w"] = False

    def sDown(self):
        self.key_map["s"] = True

    def sUp(self):
        self.key_map["s"] = False

    def aDown(self):
        self.key_map["a"] = True

    def aUp(self):
        self.key_map["a"] = False

    def dDown(self):
        self.key_map["d"] = True

    def dUp(self):
        self.key_map["d"] = False

    def eDown(self):
        self.key_map["e"] = True

    def eUp(self):
        self.key_map["e"] = False

    def qDown(self):
        self.key_map["q"] = True

    def qUp(self):
        self.key_map["q"] = False

    def arrow_leftDown(self):
        self.key_map["arrow_left"] = True

    def arrow_leftUp(self):
        self.key_map["arrow_left"] = False

    def arrow_rightDown(self):
        self.key_map["arrow_right"] = True

    def arrow_rightUp(self):
        self.key_map["arrow_right"] = False

    def arrow_upDown(self):
        self.key_map["arrow_up"] = True

    def arrow_upUp(self):
        self.key_map["arrow_up"] = False

    def arrow_downDown(self):
        self.key_map["arrow_down"] = True

    def arrow_downUp(self):
        self.key_map["arrow_down"] = False

    def moveCamera(self, task):
        if self.x_text is None:
            self.x_text = OnscreenText(text="X: " + str(self.camera.getX()), pos=(-1, 0.9), scale=0.07)
            self.y_text = OnscreenText(text="Y: " + str(self.camera.getY()), pos=(-1, 0.8), scale=0.07)
            self.z_text = OnscreenText(text="Z: " + str(self.camera.getZ()), pos=(-1, 0.7), scale=0.07)

        if not self.key_map["debug"]:
            self.x_text.setFg((0, 0, 0, 0))
            self.y_text.setFg((0, 0, 0, 0))
            self.z_text.setFg((0, 0, 0, 0))
        else:
            self.x_text.setFg((0, 0, 0, 1))
            self.y_text.setFg((0, 0, 0, 1))
            self.z_text.setFg((0, 0, 0, 1))

        self.x_text.setText("X: " + str(self.camera.getX()))
        self.y_text.setText("Y: " + str(self.camera.getY()))
        self.z_text.setText("Z: " + str(self.camera.getZ()))

        if self.key_map["w"]:
            self.camera.setY(self.camera.getY() + 1)

        if self.key_map["s"]:
            self.camera.setY(self.camera.getY() - 1)

        if self.key_map["a"]:
            self.camera.setX(self.camera.getX() - 1)

        if self.key_map["d"]:
            self.camera.setX(self.camera.getX() + 1)

        if self.key_map["e"]:
            self.camera.setZ(self.camera.getZ() + 1)

        if self.key_map["q"]:
            self.camera.setZ(self.camera.getZ() - 1)

        if self.key_map["arrow_left"]:
            self.camera.setH(self.camera.getH() + 1)

        if self.key_map["arrow_right"]:
            self.camera.setH(self.camera.getH() - 1)

        if self.key_map["arrow_down"]:
            self.camera.setP(self.camera.getP() - 1)

        if self.key_map["arrow_up"]:
            self.camera.setP(self.camera.getP() + 1)

        return direct.task.Task.cont

if __name__ == "__main__":
    app = MyApp()
    app.run()
