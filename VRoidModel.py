from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath

from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletSoftBodyNode


class VRMLoader:
    def __init__(self, file: str, app: ShowBase):
        self.joints = {}
        self.scale = 1
        self.pos = (0, 0, 0)

        model = app.loader.loadModel(file)

        body_parts = {}
        body_animations = {}
        hair_parts = {}
        hair_animations = {}
        face_parts = {}
        face_animations = {}

        for np in model.get_children():
            print(np.get_name())
            if np.get_name() == "Face":
                face_parts["modelRoot"] = np
                face_animations["modelRoot"] = {}
            elif np.get_name() == "Hairs":
                body_parts["Hair001"] = np
                body_animations["Hair001"] = {}
            else:
                body_parts[np.get_name() if np.get_name() != "Body" else "modelRoot"] = np
                body_animations[np.get_name() if np.get_name() != "Body" else "modelRoot"] = {}



        self.body = Actor(models=body_parts, anims=body_animations)
        self.face = Actor(models=face_parts, anims=face_animations)
        # self.hair = Actor(models=hair_parts, anims=hair_animations)
        # self.hair.listJoints()
        # self.body.listJoints()

        # self.face.listJoints()
        self.face.setPos(0, 0, -1.4)

        self.head_joint = self.body.exposeJoint(None, "modelRoot", "J_Bip_C_Head")

        self.face.reparentTo(self.head_joint)


        for c in self.body.getChildren():
            if c.getName() == "Hairs":
                self.hairs = c
        #         # hair_parts = {"modelRoot": c}
        #         # hair_anims = {"modelRoot": {}}
        #         # self.hair = Actor(models=hair_parts, anims=hair_anims)
        #         # print(self.hair.getPartNames())
        #         # self.hair.listJoints()
        #         c.setPos(0, 0, -1.4)
        #         c.set_transparency(1)
        #         c.reparentTo(self.head_joint)

    def _recurse_joint(self, joint, array):
        array.append(self.body.exposeJoint(None, "modelRoot", joint.getName()))
        for c in joint.getChildren():
            self._recurse_joint(c, array)

    def rescale(self, amount : float):
        self.scale = amount
        self.body.setScale(amount, amount, amount)

    def position(self, x, y, z):
        self.body.setPos(x, y, z)
        self.pos = (x, y, z)

    def get_hair(self, parent: str, name_mask: str):
        world = BulletWorld()
        info = world.getWorldInfo()
        info.setAirDensity(1.2)
        info.setWaterDensity(0)
        info.setWaterOffset(0)
        info.setWaterNormal((0, 0, 0))

        all_positions = []
        hair_groups = []
        parent_node = self.body.get_joints(jointName=parent)[0]
        print(parent_node.getChildren())
        for c in parent_node.getChildren():
            if c.getName().startswith(name_mask):
                group = []
                self._recurse_joint(c, group)
                positions = [(-j.getPos().x * self.scale + self.pos[0],
                              -j.getPos().y * self.scale + self.pos[1],
                              j.getPos().z * self.scale + self.pos[2])
                             for j in group]
                all_positions.append(positions)
                hair_groups.append(group)
        return all_positions, hair_groups


    def reparent(self, path: NodePath):
        self.body.reparentTo(path)

    def get_morph_target(self, name: str):
        """
        Returns the result of controlJoint() called on the face.
        :param name: The name of the morph target to obtain, which is probably a number.
        :return: A NodePath representing the morph target. Using setX() moves the morph target.
        """
        if self.joints.get(name) is None:
            self.joints[name] = self.face.controlJoint(None, "modelRoot", name)
        return self.joints.get(name)

    def control_joint(self, name: str) -> NodePath:
        """
        Returns the result of controlJoint() called on the body.
        :param name: The name of the joint to obtain.
        :return: A NodePath representing the joint.
        """
        if self.joints.get(name) is None:
            self.joints[name] = self.body.controlJoint(None, "modelRoot", name)
        return self.joints.get(name)

    def expose_joint(self, name: str) -> NodePath:
        """
        Returns the result of exposeJoint() called on the body.
        :param name: The name of the joint to obtain.
        :return: A NodePath representing the joint.
        """
        return self.body.exposeJoint(None, "modelRoot", name)
