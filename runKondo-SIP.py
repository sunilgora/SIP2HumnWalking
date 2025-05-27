import mujoco as mj
import numpy as np
import os,time
from SIPlib import myRobot,humn2SIP,q2data,data2q,numik,trnparam,addrobot2scene,mydataplots,DepthvsForce
import pickle
from scipy.interpolate import CubicSpline
import scipy.io

simend = 2 #simulation time
simfreq = 50 # 100 fps
foot_size = np.array([0.050, 0.040]) #50mm,40mm np.array([length, width])
SIPwalk=True

xml_path= 'kondo/scene_Plane.xml' #Scene
robotpath='kondo/kondo_khr3hv.xml' #Robot
#robotpath='example/model/humanoid/humanoid.xml' #Robot <!-- Include sites at hip and both foot -->
xml_str=addrobot2scene(xml_path,robotpath)

# get the full path
# dirname = os.path.dirname(__file__)
# abspath = os.path.join(dirname + "/" + xml_path)
# xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_string(xml_str)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Set timestep
model.opt.timestep=0.00005

# data to humanoid parameters
humn=myRobot(ub_jnts=np.arange(18,model.nv),left_legjnts=np.arange(6,12),right_legjnts=np.arange(12,18),foot_size=foot_size,vel=0.1)

humn.mj2humn(model,data)

# Set pos and orientation
# Initial joint angles and velocity
q0=data2q(data)
q0[[2,9,10,15,16]]=[0.95*q0[2],0.5,-0.5,0.5,-0.5]

print(humn.o_left)
humn.o_left[0]=0.0
humn.o_left[2]=-model.geom_solimp[0][2]/2
print(humn.o_right)
humn.o_right[0]=0.0
humn.o_right[2]=-0.0

# Initial COM position
humn.r_com[0]=-0.0#1#-0.01
humn.r_com[1]=(0+humn.o_left[1])/3
humn.r_com[2]=0.8*humn.r_com[2]

data.qvel[0]=0.1#0.1 #0.1
# data.qvel[1]=0.1#0.1 #0.1


#print(q0)
# Find initial IK solution
q=numik(model,data,q0,1,humn.r_com,humn.o_left,humn.o_right,humn.ub_jnts,0)
data=q2data(data,q)
q0=q.copy()
#glfw.terminate()

# Humanoid parameters
humn.mj2humn(model,data)

# Check leg transition required
humn.xlimft = 0.5 # Max Steplength
humn.ylimft = abs(humn.o_left[1]-humn.o_right[1])   #max(0.1 * l, 2 * abs(qcm[1] - qcp[1]))
humn.spno = 1  # SSP=1, DSP=2
humn.Stlr=np.array([1, 0])
humn.zSw=0.02 #swing foot lift
humn.sspbydsp=2
humn.Tsip = 0  # Cycle Time for footstep control

# Terrain parameters
humn.plno = 0  # 0-default 1-virtual terrain
humn.zpln = 0  # height of terrain plane

# Terrain parameters
zeta1=7.75 # damping ratio=25
nocp=4
trn=trnparam(nocp,zeta1,humn.zpln) #hard terrain parameters for left foot terrain
trn.mjparam(model)

print('stiffness and damping =',trn.solref)
#time.sleep(2)

# Actual parameters of terrain
#model.geom_solimp[1]=[0.0, 0.95, 0.0002, 0.5, 2]
for nocp1 in np.arange(nocp,0,-1/10): #[nocp]:#[0.3]:#for defT #[1.8]:#for hardT 
    trn1=trnparam(nocp1,zeta1,humn.zpln) #hard terrain parameters for left foot terrain
    trn1.mjparam(model)

    #Change terrain solref of Humanoid xml model
    i=0
    while model.geom_bodyid[i] == 0:
        model.geom_solimp[i] = trn1.solimp[i]
        model.geom_solref[i]=trn1.solref[i]
        i=i+1

    if abs(DepthvsForce(model,data,0))<(model.geom_solimp[0][2]/2):
        print('nocp=',nocp1)
        print('stiffness=', model.geom_solref[0][0], 'damping=', model.geom_solref[0][1], ', ...wait for 2 sec')
        print('Des vs Act Deformation is:',(model.geom_solimp[0][2]/2),DepthvsForce(model,data,1))
        break

time.sleep(2)


# SIP motion
# Kondo to SIP model

# Parameters of SIP
sip=humn2SIP(humn,trn,model, data)

if SIPwalk==True:
    # Generate SIP walking pattern *** for one cycle i.e. SSP+DSP
    sipdata,ftplac = sip.siptraj(simend,simfreq,0)
    # Or load sipdata,ftplac:
    with open('siptraj.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        sipdata, ftplac = pickle.load(f)

    print('!!!!!!! SIP motion is generated/saved')

    # SIP to Kondo traj
    humn.sip2humn(sipdata,ftplac)
else:
    #LIPM to Kondo traj
    humn.lipm2humn(1/1000,simend,50,1)

# Generate Gait / Cartesian traj to joint space traj
ttraj=np.arange(0,simend,1/1000)
qtraj=humn.cart2joint(model,data,ttraj,0)
# Or open qtraj
# with open('qtraj.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     ttraj,qtraj = pickle.load(f)

# Generate spline from qtraj
humn.qspl=[]
for i in np.arange(0,model.nv):
    humn.qspl.append(CubicSpline(ttraj,qtraj[:,i],bc_type='clamped'))
    #dqi.append(CubicSpline.derivative(qi[i]))

# Initialize
data=q2data(data,q0)

# contact force
humn.mjfc(model,data)

# initialize the controller
#init_controller(model, data)
Kp=np.zeros(model.nu)
Kv=np.zeros(model.nu)
Kp[0:12]=7 #7
Kv[0:12]=0.5  #0.003
Kp[12:]=1
Kv[12:]=0.05

humn.init_controller(Kp, Kv,0*Kp)

# set the controller
# mj.set_mjcb_control(controller)
humn.COMctrl=0
humn.ZMPctrl=-0#.0001#01
humn.AMctrl=0
humn.FWctrl=0
humn.ftctrl=0

DesData,ActData=humn.sim(model,data,trn,simfreq,simend)

#Save data for MATLAB plot

# Save the list to a .mat file
scipy.io.savemat('myPyData.mat', {'DesData': DesData,'ActData':ActData})

# Plot data
mydataplots(DesData,ActData,humn)
