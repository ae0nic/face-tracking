import math

import direct
import numpy
import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock

from panda3d.core import Shader, DirectionalLight, PointLight, Vec3, LPoint3f, NodePath, rad_2_deg

from panda3d.bullet import BulletWorld, BulletDebugNode
from panda3d.bullet import BulletSoftBodyNode
from scipy.constants import degree

from panda.VRoidModel import VRMLoader

from math import pi, e, atan2

from panda.udeas import decoder_machine


class MyApp(ShowBase):
    key_map = {"w": False, "s": False, "a": False, "d": False, "e": False, "q": False,
               "arrow_left": False, "arrow_right": False, "arrow_down": False, "arrow_up": False,
               "debug": False, "debug_draw": 0, "b": False}
    world = None
    def init_physics(self):
        debugNode = BulletDebugNode('Debug')
        debugNode.showWireframe(True)
        debugNode.showConstraints(True)
        debugNode.showBoundingBoxes(False)
        debugNode.showNormals(False)
        debugNP = self.render.attachNewNode(debugNode)

        self.world = BulletWorld()
        self.world.setGravity(Vec3(0, 0, -9.81))

    def __init__(self, landmarker):
        ShowBase.__init__(self)
        self.landmarker = landmarker

        self.init_physics()

        self.x_text = None
        self.y_text = None
        self.z_text = None

        self.setBackgroundColor(0., 1., 0.)

        # Load the character model
        vrm_model = VRMLoader("./panda/Nezure.gltf", self)
        self.scene = vrm_model.body
        self.face = vrm_model.face

        # Create our shaders
        body_shader = Shader.load(Shader.SL_GLSL,
                                  vertex="panda/body.vert",
                                  fragment="panda/body.frag")

        hair_shader = Shader.load(Shader.SL_GLSL,
                                  vertex="panda/hair.vert",
                                  fragment="panda/hair.frag")

        self.scene.setShader(body_shader)
        self.scene.setShaderInput("LIGHTS", 2)
        self.scene.setShaderInput("DEBUG_MODE", self.key_map["debug_draw"])

        vrm_model.hairs.setShader(hair_shader)

        # Set up lighting
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

        # Prepare the model for rendering
        vrm_model.rescale(12)
        vrm_model.position(0, -15, 0)
        self.scene.setH(180)
        self.camera.setPos(0, -16, 19)

        # Add physics to hair (will work when Justin makes new model)
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

        self.controlled_joints = {}
        self.controlled_joints["Mouth"] = vrm_model.get_morph_target("29")
        self.controlled_joints["Eye_L"] = vrm_model.get_morph_target("14")
        self.controlled_joints["Eye_R"] = vrm_model.get_morph_target("13")
        self.controlled_joints["Head"] = vrm_model.control_joint("J_Bip_C_Head")
        self.controlled_joints["Neck"] = vrm_model.control_joint("J_Bip_C_Neck")
        self.controlled_joints["Chest_U"] = vrm_model.control_joint("J_Bip_C_UpperChest")
        self.controlled_joints["Chest"] = vrm_model.control_joint("J_Bip_C_Chest")
        self.controlled_joints["Spine"] = vrm_model.control_joint("J_Bip_C_Spine")
        self.controlled_joints["Shoulder_L"] = vrm_model.control_joint("J_Bip_R_UpperArm")
        self.controlled_joints["Elbow_L"] = vrm_model.control_joint("J_Bip_R_LowerArm")
        self.controlled_joints["Hand_L"] = vrm_model.control_joint("J_Bip_R_Hand")
        self.controlled_joints["Shoulder_R"] = vrm_model.control_joint("J_Bip_L_UpperArm")
        self.controlled_joints["Elbow_R"] = vrm_model.control_joint("J_Bip_L_LowerArm")
        self.controlled_joints["Hand_R"] = vrm_model.control_joint("J_Bip_L_Hand")



        # Run these every frame
        self.taskMgr.add(self.moveCamera, "MoveCamera")
        self.taskMgr.add(self.controlJoint, "ControlJoint", extraArgs=[vrm_model], appendTask=True) # Move the model

        self.taskMgr.add(self.run_physics, 'UpdatePhysics')
        self.taskMgr.add(self.update_hair, "UpdateHair", extraArgs=[exposed_nodes, rope_nodes, vrm_model.head_joint], appendTask=True)

        # Camera and debug controls
        self.accept("g", self.gDown)
        self.accept("1", self.oneDown)
        self.accept("2", self.twoDown)
        self.accept("n", self.nDown)
        self.accept("0", self.zeroDown)

        self.accept("b", self.bDown)

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

    def run_physics(self, task):
        dt = globalClock.getDt()
        self.world.doPhysics(dt)
        return task.cont

    def update_hair(self, hair, nodes, parent, task):
        # This will work when Justin makes the new model
        for strand, rope in zip(hair, nodes):
            for i in range(len(strand)):
                if i == 0:
                    strand[i].lookAt(parent, rope[i + 1].getPos())
                else:
                    strand[i].lookAt(strand[i-1], rope[i + 1].getPos())
                # print(strand[i].getHpr())

        return direct.task.Task.cont

    def controlJoint(self, model: VRMLoader, task):
        data = self.landmarker.run()
        head = self.controlled_joints["Head"]
        neck = self.controlled_joints["Neck"]
        chestUpper = self.controlled_joints["Chest_U"]
        chest = self.controlled_joints["Chest"]
        spine = self.controlled_joints["Spine"]
        mouth = self.controlled_joints["Mouth"]
        eye_left = self.controlled_joints["Eye_L"]
        eye_right = self.controlled_joints["Eye_R"]
        shoulder_left = self.controlled_joints["Shoulder_L"]
        elbow_left = self.controlled_joints["Elbow_L"]
        hand_left = self.controlled_joints["Hand_L"]
        shoulder_right = self.controlled_joints["Shoulder_R"]
        elbow_right = self.controlled_joints["Elbow_R"]
        hand_right = self.controlled_joints["Hand_R"]

        Hmax = .3 * pi
        Hmin = -.3 * pi
        Pmax = pi / 3
        Pmin = -.3 * pi
        Rmax = pi / 4
        Rmin = -pi / 4
        Htp = pi / 7
        Htn = -pi / 7
        Ptp = pi / 5
        Ptn = -.15 * pi
        Rtp = pi / 100
        Rtn = -pi / 100

        def tweenFunc(angle, ln, lp, tn, tp):
            ang = (ln + ((tn - ln) * pow(e, (angle - tn) / (tn - ln)))) if angle <= tn else (
                        lp - ((lp - tp) * pow(e, -(angle - tp) / (lp - tp)))) if angle >= tp else angle
            return ang * (180 / pi)

        def optimize_bone_for_slope(shoulder: NodePath, elbow: NodePath, hand: NodePath,
                                    pos1: tuple[float, float, float], pos2: tuple[float, float, float],
                                    pos3: tuple[float, float, float]):
            (x1, y1, z1) = pos1
            (x2, y2, z2) = pos2
            (x3, y3, z3) = pos3
            delta_x1 = x2 - x1
            delta_y1 = y2 - y1
            delta_z1 = z2 - z1

            delta_x2 = x3 - x2
            delta_y2 = y3 - y2
            delta_z2 = z3 - z2

            pitch1 = rad_2_deg(atan2(delta_y1, delta_x1))
            heading1 = rad_2_deg(atan2(delta_z1, delta_x1))

            pitch2 = rad_2_deg(atan2(delta_y2, delta_x2))
            heading2 = rad_2_deg(atan2(delta_z2, delta_x2))
            # Up - down
            shoulder.setR(pitch1)
            # Forward - backward
            shoulder.setP(heading1)

            # Bend
            elbow.setR(pitch2 - pitch1)
            # elbow.setP(heading2 - heading1)
            # Optional




        if len(data[4]) > 0 and not self.key_map["b"]:
            # TODO: remember that i flipped the horizontal earlier in the landmarker
            # X: Increase to right of hips
            # Y: Increase down below hips
            # Z: Increase towards camera (?)

            right_shoulder = data[4][11]
            right_elbow = data[4][13]
            right_hand = data[4][15]

            left_shoulder = data[4][12]
            left_elbow = data[4][14]
            left_hand = data[4][16]
            optimize_bone_for_slope(shoulder_left, elbow_left, hand_left,
                                    (-left_shoulder.x, left_shoulder.y, -left_shoulder.z),
                                    (-left_elbow.x, left_elbow.y, -left_elbow.z),
                                    (-left_hand.x, left_hand.y, -left_hand.z))

            optimize_bone_for_slope(shoulder_right, elbow_right, hand_right,
                                    (right_shoulder.x, -right_shoulder.y, -right_shoulder.z),
                                    (right_elbow.x, -right_elbow.y, -right_elbow.z),
                                    (right_hand.x, -right_hand.y, -right_hand.z))
        else:
            shoulder_left.setR(60)
            shoulder_left.setP(0)
            shoulder_right.setR(-60)
            shoulder_right.setP(0)

            elbow_left.setR(0)
            elbow_left.setP(0)
            elbow_right.setR(0)
            elbow_right.setP(0)

        # model.control_joint("HairJoint-1906a1ce-1b58-4a73-8500-32a1e759a35c").setX((math.sin(task.time * 5) + 1) * 90)
        if data[0] == True:
            inH = data[1][0]
            inP = -data[1][1]
            inR = data[1][2]
            HPR1 = HPR2 = np.array([tweenFunc(inH, Hmin, Hmax, Htn, Htp),
                                    tweenFunc(inP, Pmin, Pmax, Ptn, Ptp),
                                    tweenFunc(inR, Rmin, Rmax, Rtn, Rtp)])

            HPR2 = np.array(
                [(inH * (180 / pi) - HPR2[0]), (inP * (180 / pi) - HPR2[1]),
                 (inR * (180 / pi) - HPR2[2])])

            h = np.array([1 / 3, .5, 4 / 9])
            n = np.array([2 / 3, .5, 5 / 9])
            cU = np.array([.25, .2, .25])
            c = np.array([.35, .3, .35])
            s = np.array([.4, .5, .45])
            head.setHpr(tuple(HPR1 * h))
            neck.setHpr(tuple(HPR1 * n))
            chestUpper.setHpr(tuple(HPR2 * cU))
            chest.setHpr(tuple(HPR2 * c))
            spine.setHpr(tuple(HPR2 * s))

            self.scene.setX(data[1][3] * 6.)
            self.scene.setY(data[1][5] * 6.)
            self.scene.setZ(-data[1][4] * 6.)
            for shape in data[3]:
                match shape.category_name:
                    case "jawOpen":
                        mouth.setX(numpy.clip(shape.score * 2, 0, 1.2))
                    case "eyeBlinkLeft":
                        left = 0.5
                        right = 0.7
                        k = max(0, min(1, (shape.score - left)/(right - left)))
                        eye_left.setX(k)
                    case "eyeBlinkRight":
                        left = 0.5
                        right = 0.7
                        k = max(0, min(1, (shape.score - left) / (right - left)))
                        eye_right.setX(k)

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

    def bDown(self):
        self.key_map["b"] = not self.key_map["b"]

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
