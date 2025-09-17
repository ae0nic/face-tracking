import direct.directbase.DirectStart
from panda3d.core import Vec3, Point3
from panda3d.bullet import BulletWorld, BulletDebugNode, BulletSoftBodyNode
from panda3d.bullet import BulletPlaneShape
from panda3d.bullet import BulletRigidBodyNode
from panda3d.bullet import BulletBoxShape

base.cam.setPos(0, -10, 0)
base.cam.lookAt(0, 0, 0)

debugNode = BulletDebugNode('Debug')
debugNode.showWireframe(True)
debugNode.showConstraints(True)
debugNode.showBoundingBoxes(False)
debugNode.showNormals(False)
debugNP = render.attachNewNode(debugNode)
debugNP.show()

# World
world = BulletWorld()
world.setGravity(Vec3(0, 0, 0))
world.setDebugNode(debugNP.node())

# Plane
shape = BulletPlaneShape(Vec3(0, 0, 1), 1)
node = BulletRigidBodyNode('Ground')
node.addShape(shape)
np = render.attachNewNode(node)
np.setPos(0, 0, -2)
world.attachRigidBody(node)

# Box
shape = BulletBoxShape(Vec3(0.5, 0.5, 0.5))
node = BulletRigidBodyNode('Box')
node.setMass(1.0)
node.addShape(shape)
np = render.attachNewNode(node)
np.setPos(0, 0, 2)
world.attachRigidBody(node)
model = loader.loadModel('models/box.egg')
model.flattenLight()
model.reparentTo(np)

res = 8
p1 = Point3(0, 0, 4)
p2 = Point3(10, 0, 4)
fixeds = 1

bodyNode = BulletSoftBodyNode.makeRope(world.getWorldInfo(), p1, p2, res, fixeds)
bodyNode.setTotalMass(50.0)
bodyNP = render.attachNewNode(bodyNode)
world.attachSoftBody(bodyNode)


# Update
def update(task):
    dt = globalClock.getDt()
    world.doPhysics(dt)
    return task.cont

taskMgr.add(update, 'update')
base.run()