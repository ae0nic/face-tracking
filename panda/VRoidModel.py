from direct.actor.Actor import Actor
from direct.showbase.ShowBase import ShowBase
from panda3d.core import NodePath

from panda3d.bullet import BulletWorld
from panda3d.bullet import BulletSoftBodyNode


class VRMLoader:
    def __init__(self, file: str, app: ShowBase):
        self.controlled_joints = {}
        self.exposed_joints = {}
        self.scale = 1
        self.pos = (0, 0, 0)

        model = app.loader.loadModel(file)

        body_parts = {}
        body_animations = {}

        for np in model.get_children():
            body_parts[np.get_name() if np.get_name() != "toConvert2" else "modelRoot"] = np
            body_animations[np.get_name() if np.get_name() != "toConvert2" else "modelRoot"] = {}



        self.body = Actor(models=body_parts, anims=body_animations)
        self.head_joint = self.control_joint("head")


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

    def get_hair(self, parent: str):
        print("---- Getting Hair ----")
        all_positions = []
        hair_groups = []
        parent_node = self.body.get_joints(jointName=parent)[0]
        for c in parent_node.getChildren():
            if "hair" in c.getName() or "ponytail" in c.getName():
                print(c.getName())
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
        if self.controlled_joints.get(name) is None:
            self.controlled_joints[name] = self.body.controlJoint(None, "modelRoot", name)
        return self.controlled_joints.get(name)

    def control_joint(self, name: str) -> NodePath:
        """
        Returns the result of controlJoint() called on the body.
        :param name: The name of the joint to obtain.
        :return: A NodePath representing the joint.
        """
        if self.controlled_joints.get(name) is None:
            self.controlled_joints[name] = self.body.controlJoint(None, "modelRoot", name)
        return self.controlled_joints.get(name)

    def expose_joint(self, name: str) -> NodePath:
        """
        Returns the result of exposeJoint() called on the body.
        :param name: The name of the joint to obtain.
        :return: A NodePath representing the joint.
        """
        if self.exposed_joints.get(name) is None:
            self.exposed_joints[name] = self.body.exposeJoint(None, "modelRoot", name)
        return self.exposed_joints.get(name)
