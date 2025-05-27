# This file contains functions used for footstep planning of Kondo khr3hv using SIP model
# Author : Sunil Gora, IIT Kanpur
import numpy as np
import os
import xml.etree.ElementTree as ET
import mujoco as mj
import mujoco.viewer
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pickle
from scipy.optimize import minimize,differential_evolution
from copy import deepcopy
import time
import keyboard
import copy

# Find Euler angles from COM position and Contact position of SIP model
def findeulr(qcm,qcp,l):
    return np.array([np.arctan2(qcp[1] - qcm[1], qcm[2] - qcp[2]), np.arctan2(qcm[0] - qcp[0], np.linalg.norm(qcm[1:] - qcp[1:]) ), 0.0])

### OpenAI codes for Euler-quat conversion
# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0
def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler

def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))

# Modify sip.xml file to change the parameters of the pendulum (m,I,l,qcm,qcp) and terrain in SSP and DSP (solref, solimp)
def modifysip(m,I,l,qcm,qcp,plno,plnpos,plnsize,nocp,solref,solimp):
    # Load basic xml file of SIP
    xml_path = 'sip.xml'  # xml file (assumes this is in the same folder as this file)
    plno=np.minimum(2-plno,plno)

    # get the full path
    dirname = os.getcwd() #os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + xml_path)
    xml_path = abspath

    xmltree = ET.parse(xml_path)
    root = xmltree.getroot()
    # Change mass,pos, orientation and length of pendulum
    bodyeul=np.zeros([1,3])
    #model.geom_size[2,1]=l/2 # length of cylindrical rod
    for tag1 in root.findall("worldbody"):
        tag2=tag1.findall('geom')
        tag2[plno].attrib['pos']=' '.join(map(str, np.array(plnpos))) #change contact plane pos
        tag2[plno].attrib['size']=' '.join(map(str, np.array(plnsize))) #change contact plane pos
        tag2[plno].attrib['contype']=' '.join(map(str, np.array([1]))) #activate contact type
        tag2[plno].attrib['conaffinity']=' '.join(map(str, np.array([1]))) #activate contact affinity
        tag2[plno].attrib['solref']=' '.join(map(str, solref)) #change solref
        tag2[plno].attrib['solimp']=' '.join(map(str, solimp)) #change solref

        tag2=tag1.findall('body')
        tag2[0].attrib['pos']=' '.join(map(str, qcm)) #change COM pos
        tag3= tag2[0].findall('geom')
        tag3[0].attrib['mass']=' '.join(map(str, np.array([m]))) # change mass
        r=tag3[-1].attrib['size'].split(" ", 1)[0] #radius of cylinder and contact sphere
        body_pos = np.zeros([len(tag3), 3])
        k=3
        for i in np.arange(1, int((len(tag3)-1)/k+1)):
            body_pos[i, 2] = -(l-float(r)) / 2  # pos of bodyfrmae of cylindrical rod
            tag3[i].attrib['pos']=' '.join(map(str, body_pos[i,:])) # change cylinder frame pos
            tag3[i].attrib['size'] = ' '.join(map(str, np.array([r, (l - float(r)) / 2])))  # change length of cylinder to (l-r)/2
        for i in np.arange(int((len(tag3)-1)/k+1), len(tag3)):
            body_pos[i, 2] = -(l-float(r))  # pos of bodyframe of point contact sphere
            tag3[i].attrib['pos']=' '.join(map(str, body_pos[i,:])) # change contact sphere frame pos
            tag3[i].attrib['size'] = ' '.join(map(str, np.array([r])))  # change length of cylinder to (l-r)/2

        for i in np.arange(1, len(tag3)):
            tag3[i].attrib['contype'] = ' '.join(map(str, np.array([0])))  # deactivate contact type
            tag3[i].attrib['conaffinity'] = ' '.join(map(str, np.array([0])))  # deactivate contact affinity

        if plno==0:
            tag3[0].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
            tag3[0].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity
            for i in np.arange(1, 1+nocp): #int((len(tag3) - 1) /k+1)):
                tag3[i].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
                tag3[i].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity
        else:
            tag3[0].attrib['contype'] = ' '.join(map(str, np.array([0])))  # activate contact type
            tag3[0].attrib['conaffinity'] = ' '.join(map(str, np.array([0])))  # activate contact affinity
            for i in np.arange(int((len(tag3) - 1) /k + 1), int((len(tag3) - 1) /k + 1) + nocp ): #len(tag3)):
                tag3[i].attrib['contype'] = ' '.join(map(str, np.array([1])))  # activate contact type
                tag3[i].attrib['conaffinity'] = ' '.join(map(str, np.array([1])))  # activate contact affinity


        #for geotag2 in tag1.findall('geom'):
            #geotag2.attrib['size']
    # Write the xml tree to sipmotion.xml file and return path of the file
    xmltree.write('sipmotion.xml')
    ET.dump(root)
    xml_path = 'sipmotion.xml'
    return xml_path,float(r)

#MuJoCo model to Humanoid Robot data
class myRobot:
    def __init__(self, ub_jnts,left_legjnts,right_legjnts,foot_size,vel):
        self.ub_jnts=ub_jnts
        self.left_legjnts=left_legjnts
        self.right_legjnts=right_legjnts
        self.foot_size = foot_size
        self.vel = vel
        self.Tsip = 0  # Cycle Time for footstep control
        self.WD = 0 # Work Done

    def mj2humn(self,model,data):
        # Forward kinematics for position and velocity terms
        # Forward Position kinematics
        mj.mj_fwdPosition(model, data)
        # mj.mj_kinematics(model, data)
        mj.mj_comVel(model, data)
        mj.mj_subtreeVel(model, data)
        self.ti=data.time
        self.m =mj.mj_getTotalmass(model) #mass
        self.r_com=data.subtree_com[0].copy() #com position
        self.dr_com=data.subtree_linvel[0].copy()  #com velocity
        self.dq_com=data.cvel[1].copy() # com rot:lin velocity
        self.o_left=data.site('left_foot_site').xpos.copy()  # current Left foot position
        self.o_right=data.site('right_foot_site').xpos.copy()  # current Right foot position
        self.data2q(data) #write self.q and self.dq
        #Traj save
        #if data.time==0:
        #print(data.time)
        # self.tCMtraj = np.array([data.time])
        # self.oCMtraj = np.array([self.r_com])
        # self.tLtraj = np.array([data.time])
        # self.oLtraj = np.array([self.o_left])
        # self.tRtraj = np.array([data.time])
        # self.oRtraj = np.array([self.o_right])



    # Copy data.qpos (with quaternion) to q (with euler angles)
    def data2q(self,data):
        self.q = 0 * data.qvel.copy()
        self.dq = 0 * data.qvel.copy()
        qqt = data.qpos[3:7].copy()
        qeulr = quat2euler(qqt)
        for i in np.arange(0, 3):
            self.q[i] = data.qpos[i].copy()
            self.dq[i] = data.qvel[i].copy()
        for i in np.arange(3, 6):
            self.q[i] = qeulr[i - 3].copy()
            self.dq[i] = data.qvel[i].copy()
        for i in np.arange(6, len(data.qvel)):
            self.q[i] = data.qpos[i + 1].copy()
            self.dq[i] = data.qvel[i].copy()

    # Cartesian trajectory of Humanoid Robot from SIP traj.
    def sip2humn(self,sipdata, ftplac):
        i = 0
        ti = sipdata[i][0]
        dt=sipdata[i+1][0]-sipdata[i][0]
        tcyc = []
        ncyc = 0
        oLeft=self.o_left.copy()
        oRight=self.o_right.copy()
        oCM=self.r_com.copy()
        Stlr=self.Stlr.copy()
        spno=self.spno
        self.tCMtraj = np.array([self.ti])
        self.oCMtraj = np.array([oCM])
        self.tLtraj = np.array([ti-dt])
        self.oLtraj = np.array([oLeft])
        self.tRtraj = np.array([ti-dt])
        self.oRtraj = np.array([oRight])
        if Stlr[0] == 1:  # left foot is stance
            self.oCPtraj = np.array([self.o_left]) #COP
        else:
            self.oCPtraj = np.array([self.o_right])

        for item in ftplac:
            tf = item[0]
            tcyc.append(tf)
            tcm = np.empty((0))
            ocm = np.empty((0, 3))
            oct = np.empty((0, 3))
            while ti < tf:
                ti = sipdata[i][0]
                tcm = np.append(tcm, np.array([ti]), axis=0)
                ocm = np.append(ocm, np.array([sipdata[i][1][0:3]]), axis=0)
                oct = np.append(oct, np.array([sipdata[i][3]]), axis=0)
                i = i + 1
            self.tCMtraj = np.append(self.tCMtraj, tcm, axis=0)
            self.oCMtraj = np.append(self.oCMtraj, ocm, axis=0)
            self.oCPtraj = np.append(self.oCPtraj, oct, axis=0)
            if spno == 1:  # SSP
                if Stlr[0] == 1:  # left foot is stance
                    self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
                    self.oLtraj = np.append(self.oLtraj, oct, axis=0)
                    self.tRtraj = np.append(self.tRtraj, np.array([0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                    self.oRtraj = np.append(
                        np.append(self.oRtraj, np.array([0.5 * (oRight + ftplac[ncyc][3]) + [0, 0, self.zSw]]), axis=0),
                        np.array([ftplac[ncyc][3]]), axis=0)
                else:
                    self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
                    self.oRtraj = np.append(self.oRtraj, oct, axis=0)
                    self.tLtraj = np.append(self.tLtraj, np.array([0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                    self.oLtraj = np.append(
                        np.append(self.oLtraj, np.array([0.5 * (oLeft + ftplac[ncyc][3]) + [0, 0, self.zSw]]), axis=0),
                        np.array([ftplac[ncyc][3]]), axis=0)
            else:  # DSP
                Stlr = np.array([1, 1]) - Stlr
                self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
                self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
                if Stlr[0] == 1:  # left foot is stance
                    csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                        np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]),
                                        bc_type='clamped')
                    csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                        np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]),
                                        bc_type='clamped')
                    csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                        np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]),
                                        bc_type='clamped')
                    self.oLtraj = np.append(self.oLtraj, np.transpose(
                        np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                  np.array([csplz(tcm)]), axis=0)), axis=0)
                    csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                        [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]), bc_type='clamped')
                    csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                        [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]), bc_type='clamped')
                    csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                        [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]), bc_type='clamped')
                    self.oRtraj = np.append(self.oRtraj, np.transpose(
                        np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                  np.array([csplz(tcm)]), axis=0)), axis=0)
                else:
                    csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                        [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]), bc_type='clamped')
                    csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                        [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]), bc_type='clamped')
                    csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                        [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]), bc_type='clamped')
                    self.oRtraj = np.append(self.oRtraj, np.transpose(
                        np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                  np.array([csplz(tcm)]), axis=0)), axis=0)
                    csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                        np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]),
                                        bc_type='clamped')
                    csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                        np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]),
                                        bc_type='clamped')
                    csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                        np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]),
                                        bc_type='clamped')
                    self.oLtraj = np.append(self.oLtraj, np.transpose(
                        np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                  np.array([csplz(tcm)]), axis=0)), axis=0)

            ncyc = ncyc + 1
            spno = 3 - spno
            oLeft = self.oLtraj[-1]
            oRight = self.oRtraj[-1]

        self.oCMx = CubicSpline(self.tCMtraj, self.oCMtraj[:, 0])
        self.oCMy = CubicSpline(self.tCMtraj, self.oCMtraj[:, 1])
        self.oCMz = CubicSpline(self.tCMtraj, self.oCMtraj[:, 2])
        # oCMi=np.array([oCMx,oCMy,oCMz])
        self.oLx = CubicSpline(self.tLtraj, self.oLtraj[:, 0], bc_type='clamped')
        self.oLy = CubicSpline(self.tLtraj, self.oLtraj[:, 1], bc_type='clamped')
        self.oLz = CubicSpline(self.tLtraj, self.oLtraj[:, 2], bc_type='clamped')
        # oLi=np.array([oLx,oLy,oLz])
        self.oRx = CubicSpline(self.tRtraj, self.oRtraj[:, 0], bc_type='clamped')
        self.oRy = CubicSpline(self.tRtraj, self.oRtraj[:, 1], bc_type='clamped')
        self.oRz = CubicSpline(self.tRtraj, self.oRtraj[:, 2], bc_type='clamped')
        # oRi=np.array([oRx,oRy,oRz])
        self.oCPx = CubicSpline(self.tCMtraj, self.oCPtraj[:, 0])
        self.oCPy = CubicSpline(self.tCMtraj, self.oCPtraj[:, 1])
        self.oCPz = CubicSpline(self.tCMtraj, self.oCPtraj[:, 2])
        # return oCMx, oCMy, oCMz, oLx, oLy, oLz, oRx, oRy, oRz

    def lipm2humn(self,dt,Tf,sspbydsp,vis):
        i = 0
        ti = dt#sipdata[i][0]
        t0=0
        # dt=sipdata[i+1][0]-sipdata[i][0]
        tcyc = []
        ncyc = 0
        oLeft=self.o_left.copy()
        oRight=self.o_right.copy()
        oCM=self.r_com.copy()
        Stlr=self.Stlr.copy()
        spno=self.spno
        self.tCMtraj = np.array([self.ti])
        self.oCMtraj = np.array([oCM])
        self.tLtraj = np.array([self.ti])
        self.oLtraj = np.array([oLeft])
        self.tRtraj = np.array([self.ti])
        self.oRtraj = np.array([oRight])
        if Stlr[0] == 1:  # left foot is stance
            self.oCPtraj = np.array([self.o_left]) #COP
            rct = self.o_left
        else:
            self.oCPtraj = np.array([self.o_right])
            rct = self.o_right

        # LIPM motion
        r1=rct.copy()
        rcm0=self.r_com.copy()
        drcm0=self.dr_com.copy()
        tcm = np.empty((0))
        ocm = np.empty((0, 3))
        oct = np.empty((0, 3))
        ftplac=[]
        for ti in np.arange(dt,2*Tf,dt):
            if spno==1: #SSP
                Ts = np.sqrt(abs(rcm0[2] - rct[2]) / 9.81)
                # x(t)=1/2*(x(0)+Ts xdot(0)) * np.exp(t/Ts) + 1/2*(x(0)-Ts xdot(0)) * np.exp(-t/Ts)
                rcm=rct+1/2*(rcm0-rct + Ts * drcm0) * np.exp((ti-t0)/Ts) + 1/2*(rcm0-rct - Ts * drcm0) * np.exp(-(ti-t0)/Ts)
                drcm=1/2*(rcm0-rct + Ts * drcm0) * (1/Ts) * np.exp((ti-t0)/Ts) - 1/2*(rcm0-rct - Ts * drcm0) *(1/Ts)* np.exp(-(ti-t0)/Ts)
            else: #DSP
                # x(t) = x(0)*cos(t/Td) + xdot(0)* Td * sin(t/Td)
                Td=np.sqrt(abs(rcm0[2] - rct[2]) / 9.81)
                rcm = rct + (rcm0-rct) * np.cos((ti-t0)/Td) + drcm0 * Td * np.sin((ti-t0) / Td)
                drcm= -(rcm0-rct) *(1/Td)* np.sin((ti-t0)/Td) + drcm0 * 1 * np.cos((ti-t0) / Td)
            rcm[2]=rcm0[2]
            drcm[2]=0
            tcm = np.append(tcm, np.array([ti]), axis=0)
            ocm = np.append(ocm, np.array([rcm]), axis=0)
            oct = np.append(oct, np.array([rct]), axis=0)
            if spno==1:
                rcp=drcm*Ts
                r2=rcm+(1/sspbydsp)*np.linalg.norm(rcm-rct)*drcm/np.linalg.norm(drcm)
                r2[2]=rcm[2]/sspbydsp
                r3=r1+2*(r2-r1)
                r3[2]=r1[2]

            # print(rcm,r1,r2,r3)
            if (spno==1)*(rcm[0]>rct[0])*( (abs(r3[0]-r1[0]) > self.xlimft) + (abs(r3[1]-r1[1]) > self.ylimft) ) or (spno==2)*(rcm[0]>rct[0])*(abs(rcm[0]-rct[0])>abs(rcm0[0]-rct[0]))*(abs(rcm[1]-rct[1])>abs(rcm0[1]-rct[1])):
                print(ti)
                # Foot placement data
                ftplac.append([ti, r1, r2, r3])

                self.tCMtraj = np.append(self.tCMtraj, tcm, axis=0)
                self.oCMtraj = np.append(self.oCMtraj, ocm, axis=0)
                self.oCPtraj = np.append(self.oCPtraj, oct, axis=0)

                if spno == 1:  # SSP
                    if Stlr[0] == 1:  # left foot is stance
                        self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
                        self.oLtraj = np.append(self.oLtraj, oct, axis=0)
                        self.tRtraj = np.append(self.tRtraj, np.array([0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                        self.oRtraj = np.append(
                            np.append(self.oRtraj, np.array([0.5 * (oRight + ftplac[ncyc][3]) + [0, 0, self.zSw]]), axis=0),
                            np.array([ftplac[ncyc][3]]), axis=0)
                    else:
                        self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
                        self.oRtraj = np.append(self.oRtraj, oct, axis=0)
                        self.tLtraj = np.append(self.tLtraj, np.array([0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), axis=0)
                        self.oLtraj = np.append(
                            np.append(self.oLtraj, np.array([0.5 * (oLeft + ftplac[ncyc][3]) + [0, 0, self.zSw]]), axis=0),
                            np.array([ftplac[ncyc][3]]), axis=0)
                else:  # DSP
                    Stlr = np.array([1, 1]) - Stlr
                    self.tLtraj = np.append(self.tLtraj, tcm, axis=0)
                    self.tRtraj = np.append(self.tRtraj, tcm, axis=0)
                    if Stlr[0] == 1:  # left foot is stance
                        csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                            np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]),
                                            bc_type='clamped')
                        csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                            np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]),
                                            bc_type='clamped')
                        csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                            np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]),
                                            bc_type='clamped')
                        self.oLtraj = np.append(self.oLtraj, np.transpose(
                            np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                      np.array([csplz(tcm)]), axis=0)), axis=0)
                        csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                            [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]), bc_type='clamped')
                        csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                            [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]), bc_type='clamped')
                        csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                            [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]), bc_type='clamped')
                        self.oRtraj = np.append(self.oRtraj, np.transpose(
                            np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                      np.array([csplz(tcm)]), axis=0)), axis=0)
                    else:
                        csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                            [oRight[0], 0.5 * (oRight[0] + ftplac[ncyc][3][0]), ftplac[ncyc][3][0]]), bc_type='clamped')
                        csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                            [oRight[1], 0.5 * (oRight[1] + ftplac[ncyc][3][1]), ftplac[ncyc][3][1]]), bc_type='clamped')
                        csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]), np.array(
                            [oRight[2], 0.5 * (oRight[2] + ftplac[ncyc][3][2]), ftplac[ncyc][3][2]]), bc_type='clamped')
                        self.oRtraj = np.append(self.oRtraj, np.transpose(
                            np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                      np.array([csplz(tcm)]), axis=0)), axis=0)
                        csplx = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                            np.array([oLeft[0], 0.5 * (oLeft[0] + ftplac[ncyc][1][0]), ftplac[ncyc][1][0]]),
                                            bc_type='clamped')
                        csply = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                            np.array([oLeft[1], 0.5 * (oLeft[1] + ftplac[ncyc][1][1]), ftplac[ncyc][1][1]]),
                                            bc_type='clamped')
                        csplz = CubicSpline(np.array([tcm[0], 0.5 * (tcm[0] + tcm[-1]), tcm[-1]]),
                                            np.array([oLeft[2], 0.5 * (oLeft[2] + ftplac[ncyc][1][2]), ftplac[ncyc][1][2]]),
                                            bc_type='clamped')
                        self.oLtraj = np.append(self.oLtraj, np.transpose(
                            np.append(np.append(np.array([csplx(tcm)]), np.array([csply(tcm)]), axis=0),
                                      np.array([csplz(tcm)]), axis=0)), axis=0)

                t0 = ti
                tcyc.append(t0)
                ncyc = ncyc + 1
                spno = 3 - spno
                if spno==2:
                    rct=r2.copy()
                else:
                    rct = r3.copy()
                    r1 = r3.copy()
                rcm0 = rcm.copy()
                drcm0 = drcm.copy() #(ocm[-1,:] - ocm[-2]) / dt

                # plt.figure(18)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm, (np.array([sublist[0] for sublist in ocm])))
                # print(oct[-1])
                # plt.plot(tcm[-1],oct[-1][0],'*r')
                # plt.plot(tcm, (np.array([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[-1],oct[-1][1],'*b')
                # plt.figure(19)
                # # print(len(tcm[0:-1]),(np.diff([sublist[1] for sublist in ocm])))
                # plt.plot(tcm[0:-1],(np.diff([sublist[0] for sublist in ocm])/dt))
                # plt.plot(tcm[0:-1],(np.diff([sublist[1] for sublist in ocm])/dt))


                oLeft = self.oLtraj[-1]
                oRight = self.oRtraj[-1]
                tcm = np.empty((0))
                ocm = np.empty((0, 3))
                oct = np.empty((0, 3))
                if ti >= Tf:
                    break
            if vis==1 and (ti%0.01)<dt:
                print(ti) # ,drcm,rcm,r2)
                plt.figure(10)
                # plt.plot(ctime+data.time,qcp[2],'.r')
                plt.plot(ti,drcm[0],'*r',ti,drcm[1],'*g',ti,drcm[2],'*b')
                plt.xlabel('Time')
                plt.ylabel('dq/dt')
                plt.pause(0.001)
        # plt.show()
        # print(self.tCMtraj)
        self.oCMx = CubicSpline(self.tCMtraj, self.oCMtraj[:, 0])
        self.oCMy = CubicSpline(self.tCMtraj, self.oCMtraj[:, 1])
        self.oCMz = CubicSpline(self.tCMtraj, self.oCMtraj[:, 2])
        # oCMi=np.array([oCMx,oCMy,oCMz])
        self.oLx = CubicSpline(self.tLtraj, self.oLtraj[:, 0], bc_type='clamped')
        self.oLy = CubicSpline(self.tLtraj, self.oLtraj[:, 1], bc_type='clamped')
        self.oLz = CubicSpline(self.tLtraj, self.oLtraj[:, 2], bc_type='clamped')
        # oLi=np.array([oLx,oLy,oLz])
        self.oRx = CubicSpline(self.tRtraj, self.oRtraj[:, 0], bc_type='clamped')
        self.oRy = CubicSpline(self.tRtraj, self.oRtraj[:, 1], bc_type='clamped')
        self.oRz = CubicSpline(self.tRtraj, self.oRtraj[:, 2], bc_type='clamped')
        # oRi=np.array([oRx,oRy,oRz])
        self.oCPx = CubicSpline(self.tCMtraj, self.oCPtraj[:, 0])
        self.oCPy = CubicSpline(self.tCMtraj, self.oCPtraj[:, 1])
        self.oCPz = CubicSpline(self.tCMtraj, self.oCPtraj[:, 2])
        # return oCMx, oCMy, oCMz, oLx, oLy, oLz, oRx, oRy, oRz


    # Joint angles from cartesian trajectories of Kondo humanoid robot
    def cart2joint(self,model, data0, ttraj, zeroAM):

        data = deepcopy(data0)
        qtraj = []
        delt = ttraj[1] - ttraj[0]
        for ti in ttraj:
            mj.mj_fwdPosition(model, data);
            # Current joint angles
            q0 = data2q(data)
            # Desired traj
            ocm = np.array([self.oCMx(ti), self.oCMy(ti), self.oCMz(ti)])
            oLeft = np.array([self.oLx(ti), self.oLy(ti), self.oLz(ti)])
            oRight = np.array([self.oRx(ti), self.oRy(ti), self.oRz(ti)])
            qdes = numik(model, data, q0, delt, ocm, oLeft, oRight, self.ub_jnts,zeroAM)  # 0 - Upperbody locked and Nonzero ang. momentum about COM, 1 - ZAM abt COM using Arms swing
            data = q2data(data, qdes)
            qtraj.append(qdes)
            print(ti)
        qi = []
        # dqi=[]
        qtraj = np.array(qtraj)
        if len(ttraj) > 2:
            fig1 = plt.figure(1)
            fig2 = plt.figure(2)
            for i in np.arange(6, model.nv):
                plt.figure(1)
                plt.plot(ttraj, qtraj[:, i],label=f'th_{i}')
                plt.figure(2)
                plt.plot(ttraj, np.append(0, 1 / delt * np.diff(qtraj[:, i])),label=f'dth_{i}')
            plt.figure(1)
            plt.legend()
            #fig1.savefig('qtraj.jpeg')
            plt.close(fig1)
            plt.figure(2)
            plt.legend()
            #fig2.savefig('dqtraj.jpeg')
            plt.close(fig2)
            #Saving the data:
            # with open('qtraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
            #     pickle.dump([ttraj, qtraj], f)

        return qtraj

    def mjfc(self,model,data):
        # Normal Contact Force
        self.fcn = 0
        self.fcl = np.zeros([3])
        self.fcr = np.zeros([3])
        self.rf = np.zeros([3])
        self.rfl = np.zeros([3])
        self.rfr = np.zeros([3])
        for i in np.arange(0, data.ncon):
            # conid = min(data.contact[i].geom1,data.contact[i].geom2)
            fci = np.zeros([6])
            try:
                mj.mj_contactForce(model, data, i, fci)
                self.fcn = self.fcn + abs(fci[0])
                self.rf = self.rf + np.array(data.contact[i].pos) * abs(fci[0])  # pos*normal force
                #fc0 = np.matmul(np.array(data.contact[i].frame).reshape((3, 3)), -fci[0:3])  # force vector in world frame

                if model.geom_bodyid[data.contact[i].geom2] == model.site_bodyid[data.site("left_foot_site").id] or model.geom_bodyid[data.contact[i].geom1] == model.site_bodyid[data.site("left_foot_site").id]:  # Left foot body
                    self.fcl = self.fcl + fci[0:3]
                    self.rfl = self.rfl + np.array(data.contact[i].pos) * abs(fci[0])
                elif model.geom_bodyid[data.contact[i].geom2] == model.site_bodyid[data.site("right_foot_site").id] or model.geom_bodyid[data.contact[i].geom1] == model.site_bodyid[data.site("right_foot_site").id]:  # Right foot body
                    self.fcr = self.fcr + fci[0:3]
            except:
                print('no contact')

    #def mjaref(self,cnt):


    def init_controller(self,Kp,Kv,Ki):
        self.Kp=Kp
        self.Kv=Kv
        self.Ki=Ki

    def controller(self,model,data):
        # Desired traj
        # thdes=np.zeros([model.nu])
        # dthdes = np.zeros([model.nu])
        self.qdes = np.zeros([model.nv])
        self.dqdes = np.zeros([model.nv])
        self.ddqdes = np.zeros([model.nv])

        self.tau=0*data.ctrl

        # PD control
        self.tau += self.PDcontrol(model, data)

        # Ankle Torque Control
        # tau_ankle = 0

        # COM control # Choi et al. without ZMP control
        self.tau += self.COMcontrol(model,data)

        # ZMP control # Choi et al.
        self.tau += self.ZMPcontrol(model,data)

        # Angular momentum control
        self.tau += self.AMcontrol(model,data)

        # FW control # FW inverted pendulum to control COM
        self.tau +=self.FWcontrol(model,data)

        # Inverse dynamics for model-based control
        tauid = tauinvd(model, data, self.ddqdes)

        # Controller data.ctrl
        #data.ctrl = tau_PD + COMctrl * tau_COM + AMctrl * tau_AM + FWctrl * tau_FW
        # Gear Ratio
        for i in range(model.nu):
            self.tau[i] = self.tau[i] / model.actuator_gear[i][0]

        return self.tau

    def PDcontrol(self, model, data):
        # Current joint angles
        q = data2q(data)

        for i in np.arange(0, model.nv):
            self.qdes[i] = self.qspl[i](1 * data.time)
            self.dqdes[i] = self.qspl[i](1 * data.time, 1)
            self.ddqdes[i] = self.qspl[i](1 * data.time, 2)
            # qdes[i+6]=thdes[i].copy()
            # data.qacc[i]=ddqdes[i].copy()
        # PD Controller
        self.tau_PD = 0 * data.ctrl
        for i in np.arange(0, model.nu):
            self.tau_PD[i] = (self.Kp[i]) * (self.qdes[i + 6] - q[i + 6]) + self.Kv[i] * (
                    (self.qdes[i + 6] - q[i + 6]) / model.opt.timestep - data.qvel[i + 6])  # (dqdes[i+6]-dq[i+6])

        return self.tau_PD

    def COMcontrol(self,model, data):
        self.ocm_des = np.array(
            [self.oCMx(data.time), self.oCMy(data.time), self.oCMz(data.time)])  # Desired COM Pos
        self.rcom = data.subtree_com[0].copy()  # current COM position
        # drcom = data.subtree_linvel[0].copy()
        if self.COMctrl == 0:
            self.tau_COM = 0 * data.ctrl
        else:
            jnts = range(model.nv)
            #tau_COM, q_err, dq_err = self.COMcontrol(model, data, ocm_des, rcom, self.Kp, self.Kv, jnts)

            Jcm = np.zeros((3, model.nv))  # COM position jacobian
            mj.mj_jacSubtreeCom(model, data, Jcm, 0)

            J1 = np.zeros([model.nv - len(jnts), model.nv])
            J1[:, [x for x in range(model.nv) if x not in jnts]] = np.eye(model.nv - len(jnts))
            InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
            J2 = Jcm
            delx2 = 1 * 1e3 * (self.ocm_des - self.rcom) / model.opt.timestep
            Jt2 = np.matmul(J2, InJ1)
            dq_err = 1 * np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, data.qvel))
            # dq_err=1*np.matmul(np.linalg.pinv(Jcm), delx2)
            q_err = dq_err * model.opt.timestep

            self.tau_COM = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_COM[i] = self.Kp[i] * (q_err[i + 6]) + self.Kv[i] * (dq_err[i + 6])  # (dqdes[i+6]-dq[i+6])

            # Modify desired joint traj
            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep


        return self.tau_COM

    def ZMPcontrol(self,model, data):
        self.ocp_des = np.array(
            [self.oCPx(data.time), self.oCPy(data.time), self.oCPz(data.time)])  # Desired COP Pos
        # Normal contact force
        self.mjfc(model, data)
        # COP pos
        if abs(self.fcn) > 0:
            self.rcop = self.rf / abs(self.fcn)
        else: #No contact
            self.rcop = self.ocp_des #np.array([np.nan, np.nan, np.nan])

        if self.ocp_des[2]>self.rcom[2]:
            if abs(self.fcn) == 0:
                self.rcop[2]=0
            self.ocp_des[0]=self.ocm_des[0] + (self.ocm_des[0]-self.ocp_des[0])*abs(self.rcop[2]-self.ocm_des[2])/(self.ocp_des[2]-self.ocm_des[2])
            self.ocp_des[1] = self.ocm_des[1] +  (self.ocm_des[1] - self.ocp_des[1])*abs(self.rcop[2] - self.ocm_des[2])/(self.ocp_des[2] - self.ocm_des[2])
            self.ocp_des[2] = self.rcop[2]
        if abs(self.fcn) == 0:
            self.rcop = self.ocp_des #np.array([np.nan, np.nan, np.nan])

        # self.rcom = data.subtree_com[0].copy()  # current COM position
        # drcom = data.subtree_linvel[0].copy()
        if self.ZMPctrl == 0:
            self.tau_ZMP = 0 * data.ctrl
        else:
            jnts = range(model.nv)
            #tau_COM, q_err, dq_err = self.COMcontrol(model, data, ocm_des, rcom, self.Kp, self.Kv, jnts)

            Jcm = np.zeros((3, model.nv))  # COM position jacobian
            Jct = np.zeros((3, model.nv))  # Stance foot center position jacobian
            mj.mj_jacSubtreeCom(model, data, Jcm, 0)

            if abs(self.fcl[0])>0 and abs(self.fcr[0])==0:
                mj.mj_jacSite(model,data, Jct, None, data.site("left_foot_site").id)
            elif abs(self.fcr[0])>0 and abs(self.fcr[0])==0:
                mj.mj_jacSite(model,data, Jct, None, data.site("right_foot_site").id)
            else:
                if self.o_right[0] > self.o_left[0]:
                    mj.mj_jacSite(model, data, Jct, None, data.site("left_foot_site").id)
                else:
                    mj.mj_jacSite(model, data, Jct, None, data.site("right_foot_site").id)

            J1 = np.zeros([model.nv - len(jnts), model.nv])
            J1[:, [x for x in range(model.nv) if x not in jnts]] = np.eye(model.nv - len(jnts))
            InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
            J2 = Jcm - Jct
            delx2 =  1 * self.ZMPctrl * (self.ocp_des - self.rcop) / model.opt.timestep
            # print(delx2,self.ocp_des,self.rcop)
            Jt2 = np.matmul(J2, InJ1)
            # dq_err = 1 * np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, data.qvel))
            dq_err=1*np.matmul(np.linalg.pinv(Jcm - Jct), delx2)
            self.q_err = self.q_err + dq_err * model.opt.timestep

            self.tau_ZMP = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_ZMP[i] = self.Kp[i] * (dq_err[i + 6]* model.opt.timestep) + self.Kv[i] * (dq_err[i + 6])  # (dqdes[i+6]-dq[i+6])

            # Modify desired joint traj
            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep


        return self.tau_ZMP

    def AMcontrol(self,model, data):
        # Zero angular momentum is feasible with no contact force. In dynamics, The Zero angular momentum contradict with the contact force.
        # Angular momentum
        self.Iwb = np.zeros([3, model.nv])
        Lw = self.Iwb @ (0*data.qvel)
        mj.mj_angmomMat(model, data, self.Iwb, 0)

        self.tau_AM = 0 * data.ctrl
        if self.AMctrl == 1:
            jnts = self.ub_jnts
            #tau_AM, q_err, dq_err = AMcontrol(model, data, Iwb, dqdes, Lw, self.Kp, self.Kv, jnts)

            J1 = np.zeros([model.nv - len(jnts), model.nv])
            J1[:, [x for x in range(model.nv) if x not in jnts]] = np.eye(model.nv - len(jnts))
            InJ1 = np.eye(model.nv) - np.matmul(np.linalg.pinv(J1), J1)
            J2 = self.Iwb
            delx2 = Lw  # np.zeros([3])
            Jt2 = np.matmul(J2, InJ1)
            dq_err = 1 * np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2, self.dqdes))
            q_err = dq_err * model.opt.timestep

            #self.tau_AM = 0 * data.ctrl
            for i in np.arange(0, model.nu):
                self.tau_AM[i] = self.Kp[i] * (q_err[i + 6]) + self.Kv[i] * ((dq_err[i + 6]))  # (dqdes[i+6]-dq[i+6])

            # Modify desired joint traj
            self.dqdes = self.dqdes + dq_err
            self.qdes = self.qdes + dq_err * model.opt.timestep

        return self.tau_AM

    def FWcontrol(self,model,data):
        if self.FWctrl == 0:
            self.tau_FW = 0 * data.ctrl
        else:
            self.tau_FW = 0 * data.ctrl
            jnts = 0
            # th_des=-np.arctan2(ocm_des[2],ocm_des[0])
            # th_FW=-np.arctan2(rcom[2],rcom[0])
            # tau_FW[0]=(data.ncon>0)*(-Kp[0]*(th_des-th_FW)-Kv[0]*(0))
            comX_err = (self.ocm_des[0] - self.rcom[0])
            self.tau_FW[jnts] = (data.ncon > 0) * (-self.Kp[0] * (comX_err) - self.Kv[0] * (0))
            self.tau[jnts] = 0

        return self.tau_FW



    def sim(self,model,data,trn,simfreq,simend):

        # Humanoid parameters
        self.mj2humn(model, data)

        # Parameters of SIP
        sip = humn2SIP(self,trn, model, data)
        sip.trn.cntplane(sip.qcp,sip.spno)

        # Mocap body we will control with our mouse. for COP
        mocap_id_COP = model.body("COP").mocapid[0]
        mocap_id_COM_des = model.body("COM_des").mocapid[0]

        ActData = []
        DesData = []

        with mj.viewer.launch_passive(
                model=model, data=data, show_left_ui=True, show_right_ui=False
        ) as viewer:

            # Initialize the camera view to that of the free camera.
            mj.mjv_defaultFreeCamera(model, viewer.cam)

            # Visualization.
            # viewer.opt.frame = mj.mjtFrame.mjFRAME_SITE #Site frame
            # viewer.opt.flags[2] = 1  # Joints
            # viewer.opt.flags[4] = 1  # Actuators
            # viewer.opt.flags[14] = 1 #Contact Points
            viewer.opt.flags[16] = 1 #Contact Forces
            viewer.opt.flags[18] = 1 #Transparent
            # viewer.opt.flags[20] = 1 #COM


            # Update scene and render
            # viewer.sync()

            # print("Press any key to proceed.")
            # key = keyboard.read_key()
            print("Simulation starting.....")
            time.sleep(5)

            while viewer.is_running() and data.time < simend:
                time_prev = data.time

                clock_start = time.time()
                while (data.time - time_prev < 1.0 / simfreq) and data.time < simend:
                    # Current joint angles
                    #q = data2q(data)
                    #dq = data.qvel.copy()

                    if self.ftctrl==1 and data.time>=self.Tsip:

                        # Humanoid parameters
                        self.mj2humn(model, data)

                        sip.qcm=self.r_com.copy()
                        sip.dqcm=self.dr_com.copy()
                        #sip.dth=self.dq[3:6].copy()
                        if self.spno==1:
                            if self.Stlr[0]==1:
                                sip.qcp=self.o_left
                            else:
                                sip.qcp=self.o_right

                        # Generate SIP walking pattern *** For simTime *** for one cycle i.e. SSP+DSP
                        sipdata,ftplac = sip.sipstep(data.time,simend,simfreq,0)
                        self.Tsip=sipdata[-1][0]

                        # SIP to Kondo traj
                        self.sip2humn(sipdata, ftplac)

                        # Generate Gait / Cartesian traj to joint space traj
                        ttraj = np.arange(data.time, self.Tsip, min(1 / 1000,(self.Tsip-data.time)/2))
                        qtraj = self.cart2joint(model, data, ttraj, 0)
                        # Or open qtraj
                        # with open('qtraj.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
                        #     ttraj, qtraj = pickle.load(f)


                        # Generate spline from qtraj
                        self.qspl = []
                        for i in np.arange(0, model.nv):
                            self.qspl.append(CubicSpline(ttraj, qtraj[:, i]))
                            # dqi.append(CubicSpline.derivative(qi[i]))

                        # Change SSP/DSP
                        if self.spno==2:
                            self.Stlr=np.array([1, 1])-self.Stlr
                        self.spno = 3 - self.spno


                    data.ctrl=self.controller(model,data).copy()
                    # data.ctrl=0*data.ctrl #Zero Torque

                    # mj.mj_inverse(model, data)  # Inverse  dynamics
                    # data=q2data(data,self.qdes) # Joint traj for inv dynamics
                    # data.qvel=self.dqdes
                    # data.qacc=self.ddqdes
                    # data.time = data.time + 1 / simfreq

                    mj.mj_step(model, data)  # Forward dynamics
                    # print('solver_fwdinv[2] flag',data.solver_fwdinv)

                    # Terrain identification - solref,solimp
                    if self.ftctrl==1 and data.time>0.9*self.Tsip:
                        # MuJoCo to Humanoid Parameters
                        self.mj2humn(model, data)

                        # Normal contact force
                        # self.mjfc(model, data)

                        if self.spno==1:
                            if self.Stlr[0]==1:
                                cntpt=self.o_left
                            else:
                                cntpt=self.o_right
                        else:
                            if self.Stlr[0]==1:
                                cntpt=self.o_right
                            else:
                                cntpt=self.o_left

                        # Find the contact geom id
                        trn.cntplane(cntpt,1)
                        # a0vec = np.zeros([data.nefc])
                        # mj.mj_mulJacVec(model, data, a0vec,data.qacc_smooth)  # Unconstrained acceleration in contact space #1/m*(-m*9.81) #J*data.qacc_smooth --Unconstrained acceleration
                        # fmj,fsd,defmj,margmj=mjforce(model,data)
                        # print('efc_force,fmj= ',data.efc_force,fmj)

                        for item in data.contact:
                            if item.geom1 == trn.cntgeomid or item.geom2 == trn.cntgeomid:
                                efc_id = item.efc_address
                                trn.r.append(data.efc_pos[efc_id] - data.efc_margin[efc_id])  # deformation
                                trn.rdot.append(data.efc_vel[efc_id])  # deformation rate
                                trn.aref.append(data.efc_aref[efc_id])
                                # trn.A.append(data.efc_diagApprox[efc_id])
                                # trn.f.append(data.efc_force[efc_id])

                        if self.spno==1 and data.time>=self.Tsip: # len(trn.r)==100: # Find solref and solimp for cntpt
                            trn.paramidentify()
                            print('Actual solref=', np.round(model.geom_solref[trn.cntgeomid], 2))
                            print('Calc solref=', np.round(trn.cntsolref, 2))
                            print('Actual solimp=', model.geom_solimp[trn.cntgeomid])
                            print('Calc solimp=', trn.cntsolimp)
                            print(asdf)  # stop with error
                            #Update sip.trn
                            sip.trn.solref[trn.cntgeomid]=trn.cntsolref.copy()
                            sip.trn.solimp[trn.cntgeomid]=trn.cntsolimp.copy()
                            #Erase saved data
                            trn.r=[]
                            trn.rdot=[]
                            trn.aref=[]

                    # Work Done

                    for i in range(model.nu):
                        self.WD += abs(data.ctrl[i] * model.actuator_gear[i][0] * data.qvel[i+6] * model.opt.timestep)

                if (data.time >= simend):
                    break;

                # MuJoCo to Humanoid Parameters
                self.mj2humn(model, data)

                # Normal contact force
                self.mjfc(model, data)
                # print('ncon,efc_force',data.ncon,data.efc_force)
                # fmj,fsd,fdef,dddef=mjforce(model,data)
                # print('mj_force',fmj)
                # print('nefc,efc_pos,efc_force',data.nefc,data.efc_pos,data.efc_force)
                # print('efc_aref,force',data.efc_aref,data.efc_force)
                #COP COM mocap
                if abs(self.fcn) > 0:
                    rcop = self.rf / abs(self.fcn)
                    # Set the target position of the end-effector site.
                    data.mocap_pos[mocap_id_COP, 0:3] = rcop
                else:
                    rcop = np.array([np.nan, np.nan, np.nan])
                    # Set the target position of the end-effector site.
                    data.mocap_pos[mocap_id_COP, 0:3] = np.array([1000, 1000, 1000])

                sipCnt=np.array([self.oCPx(data.time),self.oCPy(data.time),self.oCPz(data.time)])
                sipCOM=np.array([self.oCMx(data.time),self.oCMy(data.time),self.oCMz(data.time)])
                data.mocap_pos[mocap_id_COM_des, 0:3] = sipCOM
                #Visualize SIP Model
                # iterator for decorative geometry objects
                idx_geom = 0
                for i in range(100):
                    # mj Geometry from vyankatesh's code
                    sipPt=sipCOM+i/100*(sipCnt-sipCOM)
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[idx_geom],
                                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                        size=[0.005, 0, 0],
                                        pos=sipPt,
                                        mat=np.eye(3).flatten(),
                                        rgba=np.array([1, 0, 0, 0.3]))
                    idx_geom += 1
                    viewer.user_scn.ngeom = idx_geom
                    # Reset if the number of geometries hit the limit
                    if idx_geom > (viewer.user_scn.maxgeom - 50):
                        # Reset
                        idx_geom = 1

                # Reproduce MuJoCo Forces
                # fmj,fsd =mjforce(model,data)

                # Save data for plots [t,q,dq,rcom,drcom,oL,oR,rcop,fcl,fcr,tau,I*dq,WD]
                DesData.append([data.time, self.qdes.copy(), self.dqdes.copy(), np.array([self.oCMx(data.time),self.oCMy(data.time),self.oCMz(data.time)]), np.empty(0), np.array([self.oLx(data.time),self.oLy(data.time),self.oLz(data.time)]),
                               np.array([self.oRx(data.time),self.oRy(data.time),self.oRz(data.time)]), self.ocp_des.copy()])
                ActData.append([data.time, self.q.copy(), self.dq.copy(), self.rcom.copy(), self.dr_com.copy(), self.o_left.copy(),
                               self.o_right.copy(), rcop.copy(), self.fcl.copy(), self.fcr.copy(), data.ctrl.copy(),
                               self.Iwb @ data.qvel, self.WD.copy(), self.WD/(self.m*9.81)])

                # Update scene and render
                viewer.sync()
                time_until_next_step = 1 / simfreq - (time.time() - clock_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                print(data.time)

        return DesData, ActData



# Kondo to SIP model
# Parameters of SIP
class humn2SIP:
  def __init__(self, humn,trn,model,data):
    self.m =mj.mj_getTotalmass(model) #1.379
    self.I=1/1000*np.eye(3,3);
    self.qcm=data.subtree_com[0].copy() #np.array([-0.05, -0.025, 0.25])
    self.dqcm=data.subtree_linvel[0].copy() #np.array([0.39, 0.15, 0.0])
    self.Stlr=humn.Stlr.copy()
    self.plno=humn.plno
    self.trn=trn
    self.xlimft=humn.xlimft
    self.ylimft=humn.ylimft
    self.spno=humn.spno
    self.sspbydsp=humn.sspbydsp

    if self.spno==1:
        if self.Stlr[0]==1:
            self.qcp=data.site('left_foot_site').xpos.copy()  # current Left foot position  #np.array([0.0, 0.0, 0.0])
            self.dqcp=0*self.dqcm.copy()
        else:
            self.qcp=data.site('right_foot_site').xpos.copy()  # current Left foot position  #np.array([0.0, 0.0, 0.0])
            self.dqcp=0*self.dqcm.copy()
    else:
        if self.Stlr[0]==1:
            self.qcp=data.site('right_foot_site').xpos.copy()  # current Left foot position  #np.array([0.0, 0.0, 0.0])
            self.dqcp=0*self.dqcm.copy()
        else:
            self.qcp=data.site('left_foot_site').xpos.copy()  # current Left foot position  #np.array([0.0, 0.0, 0.0])
            self.dqcp=0*self.dqcm.copy()

    self.l = np.linalg.norm(self.qcm - self.qcp)
    self.dth=data.qvel[3:6] #np.array([0,0,0])
    self.theulr = findeulr(self.qcm,self.qcp,self.l)
    self.qqt = euler2quat(self.theulr)

    self.rc = self.qcm
    self.r1 = self.qcp
    self.r2 = 0
    self.r3 = 0

  # Generate SIP trajectory given the initial pos and velocity of COM
  def siptraj(self, simend, simfreq, vis):
      # Parameters of SIP
      m=self.m
      I=self.I
      #qcm=self.qcm.copy()
      #dqcm=self.dqcm.copy()
      #qcp=self.qcp.copy()
      l = np.linalg.norm(self.qcm - self.qcp)
      theulr = findeulr(self.qcm, self.qcp, l)
      qqt = euler2quat(theulr)
      # dqcp = np.array([0, 0, 0])
      dth = np.matmul(np.linalg.pinv(l * np.array(
          [[0, np.cos(theulr[1]), 0],
           [-np.cos(theulr[0]) * np.cos(theulr[1]), np.sin(theulr[0]) * np.sin(theulr[1]), 0],
           [-np.sin(theulr[0]) * np.cos(theulr[1]), -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - 0 * self.dqcm)

      # zpln = trn[plno].zpln
      # solref = trn[plno].solref
      # solimp = trn[plno].solimp
      # nocp = trn[plno].nocp

      self.trn.cntplane(self.qcp, self.spno)
      xml_path, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)  # create new xml file from basic sip

      # MuJoCo data structures
      model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
      data = mj.MjData(model)  # MuJoCo data
      cam = mj.MjvCamera()  # Abstract camera
      opt = mj.MjvOption()  # visualization options

      # Example on how to set camera configuration
      # cam.azimuth = 90
      # cam.elevation = -45
      # cam.distance = 2
      # cam.lookat = np.array([0.0, 0.0, 0])

      ctime = 0  # total time of all sip motions
      rc = self.qcm.copy()
      r1 = self.qcp.copy()
      r2 = 0
      r3 = 0
      # Run walking pattern generation
      sipdata = []
      ftplac = []
      while ctime < simend:
          # data.contact.solref
          data, sipdata, tf, self.qcm, self.dqcm, rc, r1, r2, r3 = sipmotion(model, data, simend, simfreq, self.spno, m, l, self.qcm, qqt,
                                                                   self.qcp, self.dqcm, dth, rc, r1, r2, r3, self.sspbydsp, self.xlimft,
                                                                   self.ylimft, ctime,
                                                                   sipdata, opt, cam, vis)
          try:
              # stiffness
              kn = data.contact.solimp[0][0] / ((data.contact.solimp[0][1] ** 2) * (data.contact.solref[0][0] ** 2) * (
                      data.contact.solref[0][1] ** 2))
              print(kn)  # stiffness
              # damping
              print(2 / (data.contact.solimp[0][1] * data.contact.solref[0][0]))
          except:
              print('stiffness/dampting error')

          # Increment time
          ctime = ctime + tf
          # Change SSP/DSP
          self.spno = 3 - self.spno
          if self.spno == 2:
              self.plno = 1  # change contact plane
              print('Angular momentum abt r1', np.cross(self.qcm - self.qcp, m * self.dqcm))
              self.qcp = r2.copy()
              print('foot placement position r3', r3)
              # Contact parameters
              self.Stlr = np.array([1, 1]) - self.Stlr

              # Swing foot pos at the end of DSP
              self.rsw = np.append(r1[0:2], self.trn.pos[self.trn.cntgeomid][2] + self.trn.size[self.trn.cntgeomid][2])
              # Stance foot pos at the end of SSP
              self.trn.cntplane(r3, self.spno)
              self.rst = np.append(r3[0:2], self.trn.pos[self.trn.cntgeomid][2] + self.trn.size[self.trn.cntgeomid][2])
              # Foot placement data
              ftplac.append([ctime, r1, r2, self.rst])
              self.trn.cntplane(self.qcp, self.spno)

              plt.plot(self.qcp[0], self.qcp[1], 'bo')

          else:
              print('Angular momentum abt r2', np.cross(self.qcm - self.qcp, m * self.dqcm))
              # Contact parameters
              if self.Stlr[0] == 1:
                  self.plno = 0  # change contact plane
              else:
                  self.plno = 2  # change contact plane
              self.qcp = r3.copy()  # Change contact point
              #self.qcp[2] = 0 - self.trn.cntsolimp[2]  # Contact point deformation
              # Foot placement data
              self.trn.cntplane(self.qcp, self.spno)
              self.qcp[2] = self.trn.cntpos[2] - self.trn.cntsolimp[2]/2  # Contact point deformation
              ftplac.append([ctime, self.rsw, r2, self.qcp])

              plt.plot(self.qcp[0], self.qcp[1], 'ko')

          print('Angular momentum abt qcp', np.cross(self.qcm - self.qcp, m * self.dqcm))
          l = np.linalg.norm(self.qcm - self.qcp)
          theulr = findeulr(self.qcm, self.qcp, l)
          qqt = euler2quat(theulr)
          dqcp = np.array([0, 0, 0])
          dth = np.matmul(np.linalg.pinv(l * np.array([[0, np.cos(theulr[1]), 0],
                                                       [-np.cos(theulr[0]) * np.cos(theulr[1]),
                                                        np.sin(theulr[0]) * np.sin(theulr[1]), 0],
                                                       [-np.sin(theulr[0]) * np.cos(theulr[1]),
                                                        -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - dqcp)
          xml_path, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp,self.trn.cntsolref, self.trn.cntsolimp)  # create new xml file from basic sip

          # MuJoCo data structures
          model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
          data = mj.MjData(model)  # MuJoCo data
          # cam = mj.MjvCamera()  # Abstract camera
          # opt = mj.MjvOption()  # visualization options
          # plt.plot(qcm[0], qcm[1], 'go')  # Plot COM position at leg transition
          # cam.lookat = np.array([qcp[0], qcp[1], 3.0])

      # print(sipdata)
      # Plot qcm --- SIP COM position
      fig, ax = plt.subplots(nrows=2, ncols=2)
      fig.suptitle('COM position')
      for item in sipdata:
          # plt.plot(item[0],item[1][0],'ro')
          # plt.xlabel('Time (s)')
          ax[0][1].plot(item[1][0], item[1][1], 'r.')
          # ax[0][1].set_xlabel('X (m)')
          ax[0][1].set_ylabel('Y (m))')
          ax[1][0].plot(item[1][1], item[1][2], 'r.')
          ax[1][0].set_xlabel('Y (m)')
          ax[1][0].set_ylabel('Z (m))')
          ax[1][1].plot(item[1][0], item[1][2], 'r.')
          ax[1][1].set_xlabel('X (m)')
          # ax[1][1].set_ylabel('Z (m))')
      plt.savefig('SIPxyz.png')
      # plt.show()
      plt.close()

      # glfw.terminate()
      # sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy()])
      # np.savez('siptraj.npz', sipdata=sipdata, ftplac=ftplac)
      # Saving the data:
      with open('siptraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
          pickle.dump([sipdata, ftplac], f)

      return sipdata, ftplac

  def sipstep(self, simstart, simend, simfreq, vis):
      # Parameters of SIP
      m=self.m
      I=self.I
      #qcm=self.qcm.copy()
      #dqcm=self.dqcm.copy()
      #qcp=self.qcp.copy()
      #l = self.l.copy() #np.linalg.norm(qcm - qcp)

      l = np.linalg.norm(self.qcm - self.qcp)
      theulr = findeulr(self.qcm, self.qcp, l)
      qqt = euler2quat(theulr)
      dqcp = np.array([0, 0, 0])
      self.dth = np.matmul(np.linalg.pinv(l * np.array(
          [[0, np.cos(theulr[1]), 0],
           [-np.cos(theulr[0]) * np.cos(theulr[1]), np.sin(theulr[0]) * np.sin(theulr[1]), 0],
           [-np.sin(theulr[0]) * np.cos(theulr[1]), -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - dqcp)

      self.trn.cntplane(self.qcp,self.spno)
      xml_path, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)

      # MuJoCo data structures
      model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
      data = mj.MjData(model)  # MuJoCo data
      cam = mj.MjvCamera()  # Abstract camera
      opt = mj.MjvOption()  # visualization options

      # Example on how to set camera configuration
      # cam.azimuth = 90
      # cam.elevation = -45
      # cam.distance = 2
      # cam.lookat = np.array([0.0, 0.0, 0])

      ctime = simstart  # total time of all sip motions
      # Run walking pattern generation
      sipdata = []
      ftplac = []
      while ctime < simend and len(ftplac)<1:
          # data.contact.solref
          data, sipdata, tf, self.qcm, self.dqcm, self.rc, self.r1, self.r2, self.r3 = sipmotion(model, data, simend, simfreq, self.spno, m, l, self.qcm, qqt,
                                                                   self.qcp, self.dqcm, self.dth, self.rc, self.r1, self.r2, self.r3, self.sspbydsp, self.xlimft,
                                                                   self.ylimft, ctime,
                                                                   sipdata, opt, cam, vis)
          try:
              # stiffness
              kn = data.contact.solimp[0][0] / ((data.contact.solimp[0][1] ** 2) * (data.contact.solref[0][0] ** 2) * (
                      data.contact.solref[0][1] ** 2))
              print(kn)  # stiffness
              # damping
              print(2 / (data.contact.solimp[0][1] * data.contact.solref[0][0]))
          except:
              print('stiffness/dampting error')

          # Increment time
          ctime = ctime + tf
          # Change SSP/DSP
          self.spno = 3 - self.spno
          if self.spno == 2:
              self.plno = 1  # change contact plane
              print('Angular momentum abt r1', np.cross(self.qcm - self.qcp, m * self.dqcm))
              self.qcp = self.r2.copy()
              print('foot placement position r3', self.r3)
              plt.plot(self.qcp[0], self.qcp[1], 'bo')
              # Contact parameters
              self.Stlr = np.array([1, 1]) - self.Stlr
              # zpln = self.qcp[2] + self.trn[self.plno].solimp[2]  # -r+(0*9.81/kn)
              # solref = self.trn[self.plno].solref
              # solimp = self.trn[self.plno].solimp
              # nocp = self.trn[self.plno].nocp
              # Swing foot pos at the end of DSP
              self.rsw=np.append(self.r1[0:2], self.trn.pos[self.trn.cntgeomid][2]+self.trn.size[self.trn.cntgeomid][2])
              # Foot placement position
              self.trn.cntplane(self.r3, self.spno)
              # Stance foot pos at the start of DSP
              self.rst=np.append(self.r3[0:2], self.trn.pos[self.trn.cntgeomid][2]+self.trn.size[self.trn.cntgeomid][2])

              # Contact point data for DSP
              self.trn.cntplane(self.qcp, self.spno)
              ftplac.append([ctime, self.r1, self.r2, self.rst])
          else:
              print('Angular momentum abt r2', np.cross(self.qcm - self.qcp, m * self.dqcm))
              # Contact parameters
              if self.Stlr[0] == 1:
                  self.plno = 0  # change contact plane
              else:
                  self.plno = 2  # change contact plane
              # zpln = self.trn[self.plno].zpln
              # solref = self.trn[self.plno].solref
              # solimp = self.trn[self.plno].solimp
              # nocp = self.trn[self.plno].nocp
              self.qcp = self.r3.copy()  # Change contact point
              # Foot placement data
              self.trn.cntplane(self.qcp, self.spno)
              self.qcp[2] = self.trn.cntpos[2] - self.trn.cntsolimp[2]  # Contact point deformation

              ftplac.append([ctime, self.rsw, self.r2, self.qcp])
              plt.plot(self.qcp[0], self.qcp[1], 'ko')

          print('Angular momentum abt qcp', np.cross(self.qcm - self.qcp, m * self.dqcm))
          l = np.linalg.norm(self.qcm - self.qcp)
          theulr = findeulr(self.qcm, self.qcp, l)
          qqt = euler2quat(theulr)
          dqcp = np.array([0, 0, 0])
          self.dth = np.matmul(np.linalg.pinv(l * np.array([[0, np.cos(theulr[1]), 0],
                                                       [-np.cos(theulr[0]) * np.cos(theulr[1]),
                                                        np.sin(theulr[0]) * np.sin(theulr[1]), 0],
                                                       [-np.sin(theulr[0]) * np.cos(theulr[1]),
                                                        -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - dqcp)
          xml_path, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)
          # MuJoCo data structures
          model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
          data = mj.MjData(model)  # MuJoCo data
          # cam = mj.MjvCamera()  # Abstract camera
          # opt = mj.MjvOption()  # visualization options
          # plt.plot(qcm[0], qcm[1], 'go')  # Plot COM position at leg transition
          # cam.lookat = np.array([qcp[0], qcp[1], 3.0])

      # print(sipdata)
      # Plot qcm --- SIP COM position
      fig, ax = plt.subplots(nrows=2, ncols=2)
      fig.suptitle('COM position')
      for item in sipdata:
          # plt.plot(item[0],item[1][0],'ro')
          # plt.xlabel('Time (s)')
          ax[0][1].plot(item[1][0], item[1][1], 'r.')
          # ax[0][1].set_xlabel('X (m)')
          ax[0][1].set_ylabel('Y (m))')
          ax[1][0].plot(item[1][1], item[1][2], 'r.')
          ax[1][0].set_xlabel('Y (m)')
          ax[1][0].set_ylabel('Z (m))')
          ax[1][1].plot(item[1][0], item[1][2], 'r.')
          ax[1][1].set_xlabel('X (m)')
          # ax[1][1].set_ylabel('Z (m))')
      #plt.savefig('SIPxyz.png')
      # plt.show()
      plt.close()

      # glfw.terminate()
      # sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy()])
      # np.savez('siptraj.npz', sipdata=sipdata, ftplac=ftplac)
      # Saving the data:
      #with open('siptraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
      #    pickle.dump([sipdata, ftplac], f)

      return sipdata, ftplac


  def NLIPtraj(self, simend, simfreq, vis):
      # Parameters of SIP
      m=self.m
      I=self.I
      #qcm=self.qcm.copy()
      #dqcm=self.dqcm.copy()
      #qcp=self.qcp.copy()
      l = np.linalg.norm(self.qcm - self.qcp)
      theulr = findeulr(self.qcm, self.qcp, l)
      qqt = euler2quat(theulr)
      # dqcp = np.array([0, 0, 0])
      dth = np.matmul(np.linalg.pinv(l * np.array(
          [[0, np.cos(theulr[1]), 0],
           [-np.cos(theulr[0]) * np.cos(theulr[1]), np.sin(theulr[0]) * np.sin(theulr[1]), 0],
           [-np.sin(theulr[0]) * np.cos(theulr[1]), -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - 0 * self.dqcm)

      # zpln = trn[plno].zpln
      # solref = trn[plno].solref
      # solimp = trn[plno].solimp
      # nocp = trn[plno].nocp

      self.trn.cntplane(self.qcp, self.spno)
      xml_path, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp, self.trn.cntsolref, self.trn.cntsolimp)  # create new xml file from basic sip

      # MuJoCo data structures
      model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
      data = mj.MjData(model)  # MuJoCo data
      cam = mj.MjvCamera()  # Abstract camera
      opt = mj.MjvOption()  # visualization options

      # Example on how to set camera configuration
      # cam.azimuth = 90
      # cam.elevation = -45
      # cam.distance = 2
      # cam.lookat = np.array([0.0, 0.0, 0])

      ctime = 0  # total time of all sip motions
      rc = self.qcm.copy()
      r1 = self.qcp.copy()
      r2 = 0
      r3 = 0
      # Run walking pattern generation
      sipdata = []
      ftplac = []
      while ctime < simend:
          # data.contact.solref
          data, sipdata, tf, self.qcm, self.dqcm, rc, r1, r2, r3 = sipmotion(model, data, simend, simfreq, self.spno, m, l, self.qcm, qqt,
                                                                   self.qcp, self.dqcm, dth, rc, r1, r2, r3, self.sspbydsp, self.xlimft,
                                                                   self.ylimft, ctime,
                                                                   sipdata, opt, cam, vis)
          try:
              # stiffness
              kn = data.contact.solimp[0][0] / ((data.contact.solimp[0][1] ** 2) * (data.contact.solref[0][0] ** 2) * (
                      data.contact.solref[0][1] ** 2))
              print(kn)  # stiffness
              # damping
              print(2 / (data.contact.solimp[0][1] * data.contact.solref[0][0]))
          except:
              print('stiffness/dampting error')

          # Increment time
          ctime = ctime + tf
          # Change SSP/DSP
          self.spno = 3 - self.spno
          if self.spno == 2:
              self.plno = 1  # change contact plane
              print('Angular momentum abt r1', np.cross(self.qcm - self.qcp, m * self.dqcm))
              self.qcp = r2.copy()
              print('foot placement position r3', r3)
              # Contact parameters
              self.Stlr = np.array([1, 1]) - self.Stlr

              # Swing foot pos at the end of DSP
              self.rsw = np.append(r1[0:2], self.trn.pos[self.trn.cntgeomid][2] + self.trn.size[self.trn.cntgeomid][2])
              self.trn.cntplane(self.qcp, self.spno)
              # Stance foot pos at the end of SSP
              self.rst = np.append(r3[0:2], self.trn.pos[self.trn.cntgeomid][2] + self.trn.size[self.trn.cntgeomid][2])
              # Foot placement data
              ftplac.append([ctime, r1, r2, self.rst])

              plt.plot(self.qcp[0], self.qcp[1], 'bo')

          else:
              print('Angular momentum abt r2', np.cross(self.qcm - self.qcp, m * self.dqcm))
              # Contact parameters
              if self.Stlr[0] == 1:
                  self.plno = 0  # change contact plane
              else:
                  self.plno = 2  # change contact plane
              self.qcp = r3.copy()  # Change contact point
              #self.qcp[2] = 0 - self.trn.cntsolimp[2]  # Contact point deformation
              # Foot placement data
              self.trn.cntplane(self.qcp, self.spno)
              self.qcp[2] = self.trn.cntpos[2] - self.trn.cntsolimp[2]/2  # Contact point deformation
              ftplac.append([ctime, self.rsw, r2, self.qcp])

              plt.plot(self.qcp[0], self.qcp[1], 'ko')

          print('Angular momentum abt qcp', np.cross(self.qcm - self.qcp, m * self.dqcm))
          l = np.linalg.norm(self.qcm - self.qcp)
          theulr = findeulr(self.qcm, self.qcp, l)
          qqt = euler2quat(theulr)
          dqcp = np.array([0, 0, 0])
          dth = np.matmul(np.linalg.pinv(l * np.array([[0, np.cos(theulr[1]), 0],
                                                       [-np.cos(theulr[0]) * np.cos(theulr[1]),
                                                        np.sin(theulr[0]) * np.sin(theulr[1]), 0],
                                                       [-np.sin(theulr[0]) * np.cos(theulr[1]),
                                                        -np.cos(theulr[0]) * np.sin(theulr[1]), 0]])), self.dqcm - dqcp)
          xml_path, r = modifysip(m, I, l, self.qcm, self.qcp, self.plno, self.trn.cntpos, self.trn.cntsize, self.trn.cntnocp,self.trn.cntsolref, self.trn.cntsolimp)  # create new xml file from basic sip

          # MuJoCo data structures
          model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
          data = mj.MjData(model)  # MuJoCo data
          # cam = mj.MjvCamera()  # Abstract camera
          # opt = mj.MjvOption()  # visualization options
          # plt.plot(qcm[0], qcm[1], 'go')  # Plot COM position at leg transition
          # cam.lookat = np.array([qcp[0], qcp[1], 3.0])

      # print(sipdata)
      # Plot qcm --- SIP COM position
      fig, ax = plt.subplots(nrows=2, ncols=2)
      fig.suptitle('COM position')
      for item in sipdata:
          # plt.plot(item[0],item[1][0],'ro')
          # plt.xlabel('Time (s)')
          ax[0][1].plot(item[1][0], item[1][1], 'r.')
          # ax[0][1].set_xlabel('X (m)')
          ax[0][1].set_ylabel('Y (m))')
          ax[1][0].plot(item[1][1], item[1][2], 'r.')
          ax[1][0].set_xlabel('Y (m)')
          ax[1][0].set_ylabel('Z (m))')
          ax[1][1].plot(item[1][0], item[1][2], 'r.')
          ax[1][1].set_xlabel('X (m)')
          # ax[1][1].set_ylabel('Z (m))')
      plt.savefig('SIPxyz.png')
      # plt.show()
      plt.close()

      # glfw.terminate()
      # sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy()])
      # np.savez('siptraj.npz', sipdata=sipdata, ftplac=ftplac)
      # Saving the data:
      with open('siptraj.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
          pickle.dump([sipdata, ftplac], f)

      return sipdata, ftplac


# Find foot placement position when in SSP
def ftstep(m,rc,drc,qcp,sspbydsp):
    r1=qcp
    l=np.linalg.norm(rc-r1)
    #print(np.cross(rc-r1,m*drc)) # angular momentum abot cnt pt
    Lz=m*((rc[0]-r1[0])*drc[1]-(rc[1]-r1[1])*drc[0]) #angular momentum about Z axis
    try:
        dr=np.array([drc[1]*Lz/(m*np.linalg.norm(drc[0:2])**2), -drc[0]*Lz/(m*np.linalg.norm(drc[0:2])**2), 0]) # diff of COP in leg transition
    except:
        dr=0*rc
    pd=(l/sspbydsp)/np.linalg.norm(r1+dr-rc)
    r2=rc-(r1+dr-rc)*pd
    r3=np.array([r1[0]+2*(r2[0]-r1[0]), r1[1]+2*(r2[1]-r1[1]), r1[2]])
    return rc, r1, r2, r3

# Check condition to transition from SSP to DSP or vice versa
def legtrans(rc,r1,r2,r3,qcm,qcp,dqcm,spno,xlimft,ylimft):
    if spno==1:
        #print(abs(r3[0]-r1[0])>xlimft)
        cond= (np.dot(dqcm[0:2],qcm[0:2]-qcp[0:2])>0) and (abs(r3[0]-r1[0])>xlimft or abs(r3[1]-r1[1])>ylimft)
    else:
        cond= ( ((abs(r2[0]-qcm[0])>abs(r2[0]-rc[0])) and (abs(r2[1]-qcm[1])>abs(r2[1]-rc[1]))) ) and np.linalg.norm(qcm-r3)<np.linalg.norm(qcm-r1)
    return cond

# Generate SIP motion in MuJoCo for one phase (SSP or DSP) and break the loop if leg transition cond is true
def sipmotion(model,data,simend,freq,spno,m,l,qcm,qqt,qcp,dqcm,dth,rc,r1,r2,r3,sspbydsp,xlimft,ylimft,ctime,sipdata,opt,cam,vis):
    #sipdata.qcm=[]
    #sipdata.dqcm=[]
    # Set pos and orientation
    data.qpos[0:3]=qcm # position of com
    data.qpos[3:]=qqt # orientation of com
    data.qvel[0:3]=dqcm  # vel of com
    data.qvel[3:] = dth  # ang vel of com

    #initialize the controller
    #init_controller(model,data)

    #set the controller
    #mj.set_mjcb_control(controller)

    while True: #not glfw.window_should_close(window):
        time_prev = data.time

        while (data.time - time_prev < 1.0/freq):
            mj.mj_step(model, data)
            # Update sipdata
            qcm = data.qpos[0:3]
            dqcm = data.qvel[0:3]
            qqt = data.qpos[3:]
            # Euler angles
            theulr=quat2euler(qqt)
            # position of contact point
            qcp=qcm-l*np.array([np.sin(theulr[1]),-np.cos(theulr[1])*np.sin(theulr[0]),np.cos(theulr[1])*np.cos(theulr[0])])
            # Contact force
            fc = np.zeros([6])
            for i in np.arange(0, data.ncon):
                #conid = data.contact[i].geom1
                fci = np.zeros([6])
                try:
                    mj.mj_contactForce(model, data, i, fci)
                    fc = fc + fci
                except:
                    print('no contact')
            sipdata.append([ctime+data.time, data.qpos.copy(), data.qvel.copy(), qcp.copy(), fc.copy()])
            # Check condition for leg transition
            #print(data.subtree_com[0])
            if spno == 1:
                rc, r1, r2, r3 = ftstep(m,qcm, dqcm, qcp,sspbydsp)
            legtrans_check = legtrans(rc,r1,r2,r3,qcm,qcp,dqcm,spno,xlimft,ylimft)
            if legtrans_check == 1 and len(sipdata)>1:
                break
        if legtrans_check == 1 and len(sipdata)>1:
            break

        if (data.time>=simend):
            break;

    return data,sipdata, data.time, qcm,dqcm, rc,r1,r2,r3

# Copy data.qpos (with quaternion) to q (with euler angles)
def data2q(data):
    q=0*data.qvel.copy()
    qqt = data.qpos[3:7].copy()
    qeulr=quat2euler(qqt)
    for i in np.arange(0,3):
        q[i]=data.qpos[i].copy()
    for i in np.arange(3,6):
        q[i]=qeulr[i-3].copy()
    for i in np.arange(6,len(data.qvel)):
        q[i]=data.qpos[i+1].copy()
    return q

# Copy q (with euler angles) to data.qpos (with quaternion)
def q2data(data,q0):
    qqt=euler2quat(q0[3:6])
    for i in np.arange(0,3):
        data.qpos[i]=copy.copy(q0[i])
    for i in np.arange(3,7):
        data.qpos[i]=copy.copy(qqt[i-3])
    for i in np.arange(7,len(data.qpos)):
        data.qpos[i]=copy.copy(q0[i-1])
    return data


def COT(m,WD,dist):
    COT = WD / (m*9.81*dist)
    return COT

# Numerical inverse kinematics of the robot
def numik(model,data,q0, delt, ocm,oleft,oright,ubjnts,zeroAM):
    data=q2data(data,q0)
    mj.mj_fwdPosition(model, data)
    ocmi=data.subtree_com[0].copy() #current COM position

    olefti=data.site('left_foot_site').xpos.copy() #current Left foot position
    orighti=data.site('right_foot_site').xpos.copy() #current right foot position

    quat_hip=data.qpos[3:7].copy()
    quat_left=np.zeros([4])
    quat_right=np.zeros([4])

    quat_conj=np.zeros([4])
    err_quat=np.zeros([4])
    err_ori_hip = np.zeros([3])
    err_ori_left=np.zeros([3])
    err_ori_right=np.zeros([3])
    # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
    # mujoco.mju_negQuat(quat_conj, quat_hip)
    # mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
    # mujoco.mju_quat2Vel(err_ori_hip, err_quat, 1.0)

    if model.nu-len(ubjnts)>=60:
        mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
        mujoco.mju_negQuat(quat_conj, quat_left)
        mujoco.mju_mulQuat(err_quat, np.array([1,0,0,0]), quat_conj)
        mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)

        # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
        mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
        mujoco.mju_negQuat(quat_conj, quat_right)
        mujoco.mju_mulQuat(err_quat, np.array([1,0,0,0]), quat_conj)
        mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

    # Error to minimize for numerical inverse kinematics
    delE=np.linalg.norm(oleft-olefti)+np.linalg.norm(err_ori_left)+np.linalg.norm(oright-orighti)+np.linalg.norm(err_ori_right)+np.linalg.norm(ocm-ocmi)
    k=1 # Increase k to increase accuracy of q
    while delE>1e-8:
        Jcm = np.zeros((3, model.nv)) # COM position jacobian
        mj.mj_jacSubtreeCom(model, data, Jcm,0)
        Jwb = np.zeros((3, model.nv))  # Base orientation jacobian
        Jwb[0:3,3:6]=np.eye(3)
        Jvleft = np.zeros((3, model.nv)) # Left foot center jacobian
        Jwleft = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, Jvleft, Jwleft, model.site('left_foot_site').id)
        #mj.mju_mat2Quat(quat_left, data.site(model.site('left_foot_site').id).xmat)
        #mj.mju_negQuat(quat_left, quat_left)
        #mj.mju_quat2Vel(err_ori_left, quat_left, 1.0)
        Jvright = np.zeros((3, model.nv)) # right foot center jacobian
        Jwright = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, Jvright, Jwright, model.site('right_foot_site').id)
        #mj.mju_mat2Quat(quat_right, data.site(model.site('right_foot_site').id).xmat)
        #mj.mju_negQuat(quat_right, quat_right)
        #mj.mju_quat2Vel(err_ori_right, quat_right, 1.0)
        # lock upperbody
        if ubjnts.size:
            #ubjnts=np.arange(18,model.nv) #kondo khr3hv
            #ubjnts=np.append([6,7,8],np.arange(21,model.nv)) #MuJoCo humanoid model
            Jub = np.zeros((len(ubjnts), model.nv))  # Base orientation jacobian
            Jub[:,ubjnts]=np.eye(len(ubjnts))
        # Ang momentum
        Iwb=np.zeros([3,model.nv])
        mj.mj_angmomMat(model, data, Iwb, 0)

        Avec=np.zeros([18+len(ubjnts)+3+3,model.nv])
        bvec=np.zeros([18+len(ubjnts)+3+3])
        #COM traj
        Avec[0:3,0:model.nv]=Jcm
        bvec[0:3] = ocm-ocmi
        #Hip orient
        Avec[3:6,0:model.nv]=Jwb
        bvec[3:6]=err_ori_hip #np.zeros([3])
        #Left ankle lin vel
        Avec[6:9,0:model.nv]=Jvleft
        bvec[6:9] = oleft-olefti
        #Left ankle ang vel
        Avec[9:12,0:model.nv]=Jwleft
        bvec[9:12] = err_ori_left #np.zeros([3])
        #Right ankle lin vel
        Avec[12:15,0:model.nv]=Jvright
        bvec[12:15] = oright-orighti
        #Right ankle ang vel
        Avec[15:18,0:model.nv]=Jwright
        bvec[15:18] = err_ori_right #np.zeros([3])
        #Upper body joints
        if ubjnts.size:
            #Lock upper body joints
            Avec[18:18+len(ubjnts),0:model.nv]=Jub
            bvec[18:18+len(ubjnts)] = np.zeros([len(ubjnts)])
            #Zero angular momentum using upper body joints
            Avec[18+len(ubjnts):18+len(ubjnts)+3,0:model.nv]=Iwb
            bvec[18+len(ubjnts):18+len(ubjnts)+3] = np.zeros([3])
            # #Sym. motion of upper body joints
            # Avec[18+len(ubjnts)+3:18+len(ubjnts)+6,ubjnts]=Iwb[:,ubjnts]
            # bvec[18+len(ubjnts)+3:18+len(ubjnts)+6] = np.zeros([3])
        else:
            Avec[18:18+3,0:model.nv]=Iwb
            bvec[18:18+3] = np.zeros([3])


        #J=np.append(np.append(np.append(np.append(Jcm,Jwb,axis=0), np.append(Jvleft,Jwleft,axis=0),axis=0), np.append(Jvright,Jwright,axis=0), axis=0),Jub,axis=0)
        #delx=np.append(np.append(np.append( np.append(ocm-ocmi,np.zeros([3]),axis=0), np.append(oleft-olefti,np.zeros([3]),axis=0),axis=0), np.append(oright-orighti,np.zeros([3]),axis=0), axis=0),np.zeros([model.nv-18]),axis=0)
        if (model.nv-len(ubjnts) )<18: #Planer biped
            eqnJ1=np.append(np.array([0,2]),np.array([6,8,10, 12,14,16])) # remove foot orientation
        else: #Spatial biped
            eqnJ1 = np.append(np.array([0, 1, 2]), np.arange(6, 18))

        J1 = Avec[eqnJ1, :].copy()
        delx1 = bvec[eqnJ1].copy()/delt
        dqN1 = np.matmul(np.linalg.pinv(J1), delx1)
        InJ1=np.eye(model.nv)-np.matmul(np.linalg.pinv(J1),J1)
        if zeroAM==True: #Zero ang momentum
            eqnJ2 = np.append(np.array([3, 4, 5]), np.arange(18 + len(ubjnts) +2, 18 + len(ubjnts) + 3)) #Abt Z-axis
            # eqnJ2=np.append(np.array([3,4,5]),np.arange(18+len(ubjnts),18+len(ubjnts)+3)) #Abt all axes
            #eqnJ2=np.arange(18+len(ubjnts),18+len(ubjnts)+2)
            #eqnJ2=np.array([18,19,21,22,23,25,26,27,model.nv,model.nv+1,model.nv+2])
        else:
            eqnJ2=np.append(np.array([3,4,5]),np.arange(18,18+len(ubjnts)))

        J2 = Avec[eqnJ2, :].copy()
        delx2 = bvec[eqnJ2].copy()/delt
        Jt2=np.matmul(J2,InJ1)
        dqN2=dqN1+np.matmul(np.linalg.pinv(Jt2), delx2 - np.matmul(J2,dqN1))
        InJ2=InJ1-np.matmul(np.linalg.pinv(Jt2),Jt2)
        dq=dqN2.copy() #+np.matmul(InJ2,qref-qi)

        if delt<1:
            q0=q0+dq*delt
            return q0

        while delE<=(np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)) :
            qi=q0+dq*delt/k
            data=q2data(data,qi)
            mj.mj_fwdPosition(model, data)
            ocmi = data.subtree_com[0]
            olefti = data.site('left_foot_site').xpos.copy()  # current Left foot position
            orighti = data.site('right_foot_site').xpos.copy()  # current right foot position
            # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
            # mujoco.mju_negQuat(quat_conj, quat_hip)
            # mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
            # mujoco.mju_quat2Vel(err_ori_hip, err_quat, 1.0)

            if model.nu - len(ubjnts) >= 60:
                mujoco.mju_mat2Quat(quat_left, data.site('left_foot_site').xmat)
                mujoco.mju_negQuat(quat_conj, quat_left)
                mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                mujoco.mju_quat2Vel(err_ori_left, err_quat, 1.0)

                # Orientation error. quat_crt * quat_err = quat_des, --> quat_err=neg(quat_crt)*quat_des
                mujoco.mju_mat2Quat(quat_right, data.site('right_foot_site').xmat)
                mujoco.mju_negQuat(quat_conj, quat_right)
                mujoco.mju_mulQuat(err_quat, np.array([1, 0, 0, 0]), quat_conj)
                mujoco.mju_quat2Vel(err_ori_right, err_quat, 1.0)

            # print(np.linalg.norm(np.matmul(J,dq)-delx))
            k=2*k
            if k>2:
                print('Error (x_des-x_cur) is diverging')


        if delE>(np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)) :
            delE = np.linalg.norm(oleft - olefti) + np.linalg.norm(err_ori_left) + np.linalg.norm(oright - orighti) + np.linalg.norm(err_ori_right) + np.linalg.norm(ocm - ocmi)
            q0 = qi.copy()
            k=1
        """
        else:
            print('Error (x_des-x_cur) is diverging')
            k=2*k
            data = q2data(data, q0)
            mj.mj_fwdPosition(model, data)
            ocmi = data.subtree_com[0]
            olefti = data.site('left_foot_site').xpos.copy()  # current Left foot position
            orighti = data.site('right_foot_site').xpos.copy()  # current right foot position
        """

    return q0

# inv dynamics in MuJoCo
def tauinvd(model,data,ddqdes):
    data.qacc=ddqdes.copy()
    #mj.mj_fwdActuation(model,data)
    mj.mj_inverse(model,data)
    #mj.mj_fwdActuation(model,data)
    tau=data.qfrc_inverse[6:].copy()
    return tau

# Find terrain parameters (solref) given solimp=[d0,dwidth,width,midpt,power] and zeta
class trnparam:
    def __init__(self,nocp,zeta,zpln):
        i=0
        self.zeta=zeta
        self.zpln=zpln
        self.nocp=nocp
        # self.A=[]
        self.r=[]
        self.rdot=[]
        self.aref=[]
        # self.f=[]
    def mjparam(self, model):
        self.solimp=[]
        self.solref=[]
        self.pos=[]
        self.size=[]
        self.xmean=[]
        i=0
        while model.geom_bodyid[i]==0:
            self.pos.append(model.geom_pos[i].copy())
            self.size.append(model.geom_size[i].copy())
            solimp=model.geom_solimp[i].copy()
            solref=model.geom_solref[i].copy()
            self.solimp.append(solimp)
            d0=solimp[0]
            dwidth=solimp[1]
            width=solimp[2]
            midpt=solimp[3]
            power=solimp[4]
            #trn.solref = (-Stiffness, -damping)
            #self.solimp = [d0, dwidth, width, midpt, power]
            dmean=(d0+dwidth)/2
            deln=width*self.nocp
            xmean = deln / 2
            if solref[0]<0:
                # kn=9.81/deln
                dampratio=self.zeta
                stiffness=(9.81*(1-dmean)*dwidth*dwidth)/(xmean*dmean*dmean) #/self.zeta**2
                timeconst=1/(dampratio*np.sqrt(stiffness))
                #dampratio = 1 / (timeconst * np.sqrt(stiffness))
                # k=stiffness*d(r)/dwidth
                #xmax=(1-dwidth)*9.81/stiffness
                # wn=np.sqrt(stiffness)
                #timeconst = 1 / (zeta * wn) #0.02 default
                # damping=2*wn*self.zeta
            else:
                timeconst=solref[0] #np.sqrt(1/(9.81*(1-dwidth)/width)) # 1*solref[0]
                dampratio = solref[1]
                # kn=9.81/deln
                stiffness=1/((timeconst**2)*(dampratio**2))#9.81*(1-dwidth)/deln
                # k=stiffness*d(r)/dwidth
                #xmax=(1-dwidth)*9.81/stiffness
                #wn=np.sqrt(nocp*stiffness)
                #timeconst = 1 / (zeta * wn) #0.02 default
                #damping=2*zeta*wn/nocp #
            damping=2/timeconst #2/(dwidth*timeconst)
            i=i+1

            #stiffness=stiffness/nocp
            #damping=damping/nocp

            self.solref.append([-stiffness,-damping])
            self.xmean.append(xmean)

    def cntplane(self,cntpt,spno):
        i=0
        for pos in self.pos:
            size=self.size[i].copy()
            if cntpt[0]>(pos[0]-size[0]) and cntpt[0]<(pos[0]+size[0]):
                if cntpt[1] > (pos[1] - size[1]) and cntpt[1] < (pos[1] + size[1]):
                    self.cntgeomid=i
                    self.cntpos=self.pos[i].copy()
                    self.cntsize=self.size[i].copy()
                    self.cntsolref=self.solref[i].copy()
                    self.cntsolimp=self.solimp[i].copy()
                    self.cntnocp=self.nocp
                    if spno==1:
                        self.cntpos[2] +=  self.cntsize[2]
                    else:
                        self.cntpos[2] = cntpt[2] + self.cntsolimp[2]/2
                        self.cntnocp=2*self.cntnocp
                    # else:
                    #     self.qcp[2]=self.cntpos[2] - self.cntsolimp[2]
                    break
            i=i+1

    def paramidentify(self):
        def fun(x):
            d0 = x[0]
            dwidth = x[1]
            width = x[2]
            midpt = x[3]
            p = x[4]
            zeta=x[5]
            dampratio=zeta
            dmean=(d0+dwidth)/2
            deln = width * 1/self.cntnocp
            xmean = deln / 2
            stiffness = (9.81*(1-dmean)*dwidth*dwidth)/(xmean*dmean*dmean) #9.81 * (1 - d_width) / width
            #wn = np.sqrt(stiffness)
            timeconst = 1/(dampratio*np.sqrt(stiffness)) #1 / (zeta * wn) #0.02 default
            damping = 2/timeconst #2 * zeta * wn  #
            # kn=x[5]
            # bn=x[6]
            # if delr > 0:
            #     delr = 0
            #     rdot = 0
            fval=0
            i=0
            for delr in self.r:
                rdot=self.rdot[i]
                x = abs(delr) / width
                if x <= midpt:
                    a = (1 / midpt) ** (p - 1)
                    y = a * (x ** p)
                else:
                    b = 1 / (1 - midpt) ** (p - 1)
                    y = 1 - b * max(0, 1 - x) ** p
                y = min(1, y)
                # print('d(r) =',1/(1+data.efc_R[0]/data.efc_diagApprox[0]))
                d = d0 + y * (dwidth - d0)
                # d=max(d,d_0)
                # d=min(d,d_width)
                # k = -model.geom_solref[0][0] * d / (d_width ** 2)
                # b = -model.geom_solref[0][1] / d_width
                # print(k/d,b)
                # zeta = (b * d_width) / (2 * np.sqrt(k * (d_width ** 2) / d))  # b/(2*np.sqrt(k)) #
                k= d*stiffness/(dwidth**2)
                b= damping/dwidth
                aref = (-b * rdot - k * delr)
                # A=self.A[i]
                # R=(1-d)/d*A
                # f=1/(A+R)*(aref-self.a0[i])
                fval +=abs(self.aref[i] - aref)
                i=i+1
            return fval
        x0=np.append([0.9,0.95,0.001,0.5,2],1) #np.append(0.5*self.cntsolimp,-self.cntsolref[1]/(2*np.sqrt(-self.cntsolref[0])))
        bnds = ((0, 0.99), (0.1, 0.99), (0.0001, 0.02), (0.0, 0.99), (0, 5), (1e-5,1000))
        # Gradient-based method
        # x=minimize(fun,x0, bounds=bnds).x
        # Search method
        # Solve minimization using differential evolution
        x = differential_evolution(fun, bounds=bnds)['x']

        print('efc_aref-aref=',fun(x))
        dmean=(x[0]+x[1])/2
        deln = x[2] * 1/self.cntnocp
        xmean=deln/2
        stiffness = (9.81*(1-dmean)*x[1]*x[1])/(xmean*dmean*dmean)#9.81 * (1 - x[1]) / x[2]
        #wn = np.sqrt(stiffness)
        timeconst =1/(x[5]*np.sqrt(stiffness)) #1 / (zeta * wn) #0.02 default
        damping = 2/timeconst #2 * x[5] * wn  #
        self.cntsolimp=x[0:5].copy()
        self.cntsolref=np.append(-stiffness,-damping) #x[5:7].copy()


# Reproduce normal contact force of MuJoCo
def mjforce(model,data):
    # Reproduce Mujoco model
    m=mj.mj_getTotalmass(model)
    # fmj=np.zeros([model.nbody])
    fmj=np.zeros([data.ncon])
    fsd=np.zeros([model.nbody])
    a0 = np.zeros([data.ncon])
    aref=np.zeros([data.ncon])
    alcp=np.zeros([data.ncon])
    A=np.zeros([data.ncon])
    R=np.zeros([data.ncon])
    D=np.zeros([data.ncon])
    def_mj=np.zeros([data.ncon])
    marg_mj=np.zeros([data.ncon])
    #efc_force=np.zeros([data.nefc])
    a0vec = np.zeros([data.nefc])
    ddq0 = data.qacc_smooth.copy()  ##Unconstrained acceleration in joint space
    mj.mj_mulJacVec(model, data, a0vec,data.qacc_smooth)  # Unconstrained acceleration in contact space #1/m*(-m*9.81) #J*data.qacc_smooth --Unconstrained acceleration

    for i in np.arange(0,data.ncon):
        efcid=data.contact[i].efc_address
        geomid=min(data.contact[i].geom) #Terrain id
        bodyid=model.geom_bodyid[max(data.contact[i].geom)] #Parent body of contact geom2
        delr=data.efc_pos[efcid]-data.efc_margin[efcid] #deformation
        rdot=data.efc_vel[efcid] #deformation rate
        print('def,rate of def',data.efc_pos[efcid],rdot)

        d_0=model.geom_solimp[geomid][0]
        d_width = model.geom_solimp[geomid][1]
        width=model.geom_solimp[geomid][2]
        midpt = model.geom_solimp[geomid][3]
        p = model.geom_solimp[geomid][4]
        if delr>0:
            delr=0
            rdot=0
        x=abs(delr)/width
        y=0
        if x>=1:
            y=1
        elif x<=midpt:
            a=(1/midpt)**(p-1)
            y=a*(x**p)
        elif x>midpt and x<1:
            b=1/(1-midpt)**(p-1)
            y=1-b*(1-x)**p
        #y=min(1,y)
        #print('d(r) =',1/(1+data.efc_R[0]/data.efc_diagApprox[0]))
        d=d_0+y*(d_width-d_0)
        k=-model.geom_solref[0][0]*d/(d_width**2)
        b=-model.geom_solref[0][1]/d_width
        #print(k/d,b)
        #print(data.efc_KBIP)
        zeta=(b*d_width)/(2*np.sqrt(k*(d_width**2)/d)) #b/(2*np.sqrt(k)) #

        aref[i] +=(-b*rdot-k*delr) #data.efc_aref[efcid] -- Reference acceleration
        a0[i]=a0vec[efcid]
        # print(aref)
        A[i]=data.efc_diagApprox[efcid] #1/(m/data.ncon)#data.efc_diagApprox[efcid] # #data.efc_diagApprox[efcid]
        R[i]=(1-d)/d*A[i]
        D[i] =1/(R[i])
        #print(D,data.efc_D)
        def_mj[i]=data.efc_pos[efcid]
        marg_mj[i]=data.efc_margin[efcid]

        # Spring force
        kn=1235#-model.geom_solref[0][0]/(d_width**2)#m*9.81/width/data.ncon
        cn=9.9#b#2*zeta*np.sqrt(kn*m)
        fsd[bodyid]=-kn*delr-cn*rdot


    #a0=a0vec[efcid] #-9.81
    #jar=(a0-aref) #data.efc_b[efcid] start deviation from ref accln
    #print((1 - d) * (-9.81) + d * aref)
    # Convex optimization Newton method
    # Minimize KE of contact to obtain yslack
    #Min(x+9.81).'M*(x+9.81) + s(J*x-aref)
    # def fct(ddqi): #Minimize-- constraint + Gauss
    #     Ma=np.zeros([model.nv])
    #     mj.mj_mulM(model,data,Ma,ddqi)
    #     #grad = Ma - data.qfrc_smooth
    #     #print(data.qacc_smooth)
    #     #print(Ma-data.qfrc_smooth-(ddqi-data.qacc_smooth))
    #     cost=1/2*np.matmul(np.transpose(Ma-data.qfrc_smooth),(ddqi-data.qacc_smooth))
    #     mj.mj_mulJacVec(model,data,a0vec,ddqi)
    #     s=0
    #     """
    #     # From Only Normal contacts
    #     a0 = np.zeros([data.ncon])
    #     for i in np.arange(0,data.ncon):
    #         bodyid = model.geom_bodyid[data.contact[i].geom2]
    #         efcid = data.contact[i].efc_address
    #         a0[i] =a0vec[efcid].copy()
    #         jar=(a0[i]-aref[i])
    #         if jar<0:
    #             s +=1/2*D[i]*jar*jar # For all constraints
    #     """
    #     # From MuJoCo's constraints
    #     a0 = np.zeros([data.nefc])
    #     for i in np.arange(0,data.ncon):
    #         efcid = data.contact[i].efc_address
    #         a0[i] =ddqi[2]#a0vec[efcid].copy()
    #         jar=(a0[i]-aref[i])
    #         if jar<0:
    #             s +=1/2*D[i]*jar*jar # For all constraints
    #
    #
    #     cost +=s
    #     return cost
    # ddqsol= minimize(fct, 0*ddq0)['x'] #data.qacc_smooth
    # # Error in reproduced accln
    # print('Sol for ddq is',ddqsol)
    # print('Error in ddqsol is',ddqsol-data.qacc)
    # mj.mj_mulJacVec(model, data, a0vec, ddqsol)
    """def sa(x):
        jar=(x-aref)
        return jar*m*jar
    vdot= fct(alcp) + sa(alcp)
    dsa=(sa(vdot)-sa(alcp))/(vdot-alcp)
    """
    #alcp=-9.81#data.qacc_smooth[2].copy()
    #print(yslack-aref)

    #jar=alcp-aref # efc_b= J*q_accsmooth-aref, # min deviation from ref accln
    #a1=a0+A*f
    for i in np.arange(0,data.ncon):
        #bodyid = model.geom_bodyid[data.contact[i].geom2]
        efcid = data.contact[i].efc_address
        # mj.mj_mulJacVec(model, data, a0vec, ddqsol)
        # alcp[i]=ddqsol[2] #a0vec[efcid] #data.qacc_warmstart[2] #-9.81
        # print('a0 =',alcp[i], (m * (-9.81) +  D[i]*aref[i])/(m +  D[i]))
        #jar=alcp[i]-aref[i]
        # if jar<0:
        fmj[i] = -1/(A[i]+R[i])*(a0[i]-aref[i]) #-D[i]*(jar)  #fmj=-1*D[i]*(jar)=-m*d*(jar<0)*(jar)
        # print('fmjforce =',fmj[i],-m*D[i]/(m+D[i])*(a0vec[efcid]-aref[i])) #for single contact pt
        #fmjsd=-1*D*(-aref) #fmj=-m*d*(jar)
        #fmjlcp=-1*D*(alcp) #fmj=-m*d*(jar)
        print(fmj)

    return fmj,fsd,def_mj,marg_mj


# Contact parameters identification given the position or force profile
def sysident(model,ttraj,rspl,rdottraj,fspl,nocp,simfreq):
    # MuJoCo data structures
    #model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model

    # function for error
    def fn_error(solimpzeta):
        data = mj.MjData(model)  # MuJoCo data
        solimp=solimpzeta[0:5]
        zeta=solimpzeta[5]
        zpln=0
        #nocp = 1  # No of contact points
        trn=trnparam( nocp, zeta, zpln)  # hard terrain parameters for left foot terrain
        trn.mjparam(model)
        # Change terrain solref of Kondo xml model
        model.geom_solref[0] = trn.solref[0]
        model.geom_solimp[0] = trn.solimp[0]
        #print('solref is',model.geom_solref[0])
        #print('solimp is',model.geom_solimp[0])
        ti = 0
        fn_err=0
        while ti < ttraj[-1]:
            while (data.time-ti)<(1/simfreq):
                mj.mj_step(model,data)
            ti=data.time

            # contact force
            fc = np.zeros([6])
            for i in np.arange(0, data.ncon):
                conid = data.contact[i].geom1
                fci = np.zeros([6])
                try:
                    mj.mj_contactForce(model, data, conid, fci)
                    fc = fc + fci
                except:
                    print('no contact')
            rcdes=np.zeros([3])
            for i in np.arange(0,3):
                rcdes[i]=rspl[i](ti)
            fcdes=np.zeros([6])
            for i in np.arange(0,3):
                fcdes[i]=fspl[i](ti)
            fn_err +=np.linalg.norm(data.qpos[0:3]-rcdes[0:3]) #np.linalg.norm(fc[0:3]-fcdes[0:3])  #
            """plt.figure(1)
            plt.plot(data.time, data.qpos[2], '.g')
            plt.figure(2)
            plt.plot(data.time, fc[0], '.g')
            """
        print('Error in sol =',fn_err)
        return fn_err

    # Actual solution
    init_sol=np.ones([6])
    init_sol[0:5]=1*model.geom_solimp[0]
    init_sol[5]=-model.geom_solref[0][1]/(2*np.sqrt(nocp*-model.geom_solref[0][0]))
    print('Act sol for ballsim is ',init_sol)
    fn_error(init_sol)
    # Sol. obtained from minimization of force error
    par_sol=np.array([9.20408857e-05, 9.00000000e-01, 8.00000000e-04, 5.00000000e-01, 2.00000000e+00, 0.32126470265858953])
    print('Sol 1 for ballsim is ',par_sol)
    fn_error(par_sol)
    # Sol. obtained from minimization of position error
    par_sol=np.array([4.00580157e-06, 6.98917856e-01, 7.38566972e-04, 4.25074547e-01, 1.98996017e+00, 10.0])
    print('Sol 2 for ballsim is ',par_sol)
    fn_error(par_sol)

    # Another Sol. obtained from minimization of force error with closer bounds
    par_sol = np.array([0.369757209, 0.835229815, 0.000776542288, 0.941126337, 1.72142685e+00, 0.3212647026585895])
    print(par_sol)
    #fn_error(par_sol)
    # Sol. obtained from minimization of force error for box
    par_sol = np.array([4.00580157e-06, 9.00000000e-01, 2.00000000e-04, 5.00000000e-01, 2.00000000e+00,  0.07139215614635253])
    print('Sol 1 for boxsim is ', par_sol)
    fn_error(par_sol)
    # Solve minimization
    bnds = ((0, 0.95), (0.5, 1), (0.0001, 0.001), (0.0, 1), (1, 2), (0.01,10))
    #par_sol=minimize(fn_error,par_sol, bounds=bnds).x
    # Solve minimization using differential evolution
    #par_sol= differential_evolution(fn_error, bounds=bnds)['x'] #data.qacc_smooth
    #print(par_sol)
    #fn_error(par_sol)

    return par_sol[0:5],par_sol[-1]

# Add robot xml to scene xml
def addrobot2scene(xml_path,robotpath):
    #xml_path = r"C:\Users\SG\OneDrive - IIT Kanpur\Documents\MATLAB Drive\Python\mujoco\kondo\scene_defT.xml" #xml file (assumes this is in the same folder as this file)

    # get the full path
    #dirname = os.getcwd() #os.path.dirname(__file__)
    #abspath = os.path.join(dirname + "/" + xml_path)
    #xml_path = abspath

    xmltree = ET.parse(xml_path)
    root = xmltree.getroot()
    # Change mass,pos, orientation and length of pendulum
    bodyeul=np.zeros([1,3])
    #model.geom_size[2,1]=l/2 # length of cylindrical rod
    for tag1 in root.findall("include"):
        tag1.attrib['file']=robotpath #' '.join(map(str, np.array([0,0,zpln]))) #change contact plane pos

    # xmltree.write('robotwithscene.xml')
    xml_str = ET.tostring(root)
    # ET.dump(root)
    # xml_path = 'robotwithscene.xml'
    return xml_str

class mydataparam:
    def __init__(self, d0, dwidth, width, midpt, power,nocp,zeta,zpln):
        self.solimp = [d0, dwidth, width, midpt, power]
        deln=width*nocp
        kn=9.81/deln
        stiffness=9.81*(1-dwidth)/deln
        wn=np.sqrt(nocp*stiffness)
        #timeconst = 1 / (zeta * wn) #0.02 default
        damping=2*zeta*wn/nocp #
        #damping=2/(dwidth*0.02)#timeconst
        self.solref=[-stiffness,-damping]
        self.zpln=zpln

def DepthvsForce(model,data,plotdata):
  # Find the height at which the vertical force becomes less than the weight, i.e. contact is initiated
  weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
  mujoco.mj_inverse(model, data)
  if data.ncon: #Contact already exists
    dz=0.000001
  else: #No contact exists
    dz=-0.000001
  while True:
    data.qpos[2] += dz
    mujoco.mj_inverse(model, data)
    # print(data.qpos[2],data.qfrc_inverse[2],weight)
    if (dz>0)*(data.ncon==0) or (dz<0)*(data.ncon>0):
      z_0=data.qpos[2]
      break
  #Plot height vs Vertical Force
  height_arr = np.linspace(z_0-0.01, z_0, 10001)
  vertical_forces = []
  for z in height_arr:
    data.qpos[2] = z
    mujoco.mj_inverse(model, data)
    #if z%0.0005==0: print(z,data.efc_KBIP, data.efc_diagApprox)
    vertical_forces.append(data.qfrc_inverse[2])

  height_offsets=height_arr-z_0
  vertical_forces=np.array(vertical_forces)

  # Find the height-offset at which the vertical force is smallest.
  idx = np.argmin(np.abs(vertical_forces))
  best_offset = height_offsets[idx]
  # Plot the relationship.
  if plotdata==1:
      print('weight=', weight)
      # setting font sizeto 30
      plt.rcParams.update({'font.size': 18})
      plt.rcParams["font.family"] = "serif"
      plt.rcParams["mathtext.fontset"] = "dejavuserif"
      plt.rcParams["axes.spines.right"] = "False"
      plt.rcParams["axes.spines.top"] = "False"
      #Plot
      fig=plt.figure(figsize=(8, 6))
      plt.plot(abs(height_offsets) * 1, vertical_forces, 'r-', linewidth=3)
      # Red vertical line at offset corresponding to smallest vertical force.
      plt.axvline(x=abs(best_offset) * 1, color='black', linestyle='-')
      # Green horizontal line at the humanoid's weight.
      weight = model.body_subtreemass[1] * np.linalg.norm(model.opt.gravity)
      plt.axhline(y=weight, color='black', linestyle='-')
      plt.xlabel('Deformation (m)')
      plt.ylabel('Vertical force on base (N)')
      plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
      # plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
      plt.minorticks_on()
      # ax.spines['right'].set_visible(False)
      # ax.spines['top'].set_visible(False)
      # plt.title(f'Min. vertical force at deformation {str(best_offset * 1000)[1:5]} mm.')
      #plt.show(block=False)
      plt.pause(1)

  return best_offset



def mydataplots(DesData,ActData,humn):
    # Saved data = [t, q, dq, rcom, drcom, oL, oR, rcop, fcl, fcr, tau, I * dq, WD]
    # Plots
    plt.rcParams.update({'font.size': 12})
    col = ['r', 'g', 'b', 'k', 'c', 'y']

    # plt.figure(1)  # Joint position     # Act vs Des q
    fig1, ax1 = plt.subplots(nrows=3, sharex=True)  # joint traj
    #fig1.suptitle('Joint angle traj')
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(DesData[0][1]))) #qdes
    Y2data = np.empty((0, len(ActData[0][1]))) #qact
    Y3data = np.empty((0, len(DesData[0][2]))) #dqdes
    Y4data = np.empty((0, len(ActData[0][2]))) #dqact

    for idata in DesData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[1]]), axis=0) #qdes
        Y3data=np.append(Y3data,np.array([idata[2]]), axis=0) #dqdes
    for idata in ActData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y2data=np.append(Y2data,np.array([idata[1]]), axis=0) #q
        Y4data=np.append(Y4data,np.array([idata[2]]), axis=0) #dq

    for i in humn.left_legjnts:  # model.nv):
        ax1[0].plot(Xdata, Y1data[:,i] * 180 / np.pi, '-', color=col[i - min(humn.left_legjnts)], label=f'th_des_{i - min(humn.left_legjnts) + 1}')
        ax1[0].plot(Xdata, Y2data[:,i] * 180 / np.pi, 'o', color=col[i - min(humn.left_legjnts)], label=f'th_act_{i - min(humn.left_legjnts) + 1}')
        ax1[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax1[0].lines) )
    for i in humn.right_legjnts:  # model.nv):
        ax1[1].plot(Xdata, Y1data[:,i] * 180 / np.pi, '-', color=col[i - min(humn.right_legjnts)], label=f'th_des_{i - min(humn.right_legjnts) + 1+len(humn.left_legjnts)}')
        ax1[1].plot(Xdata, Y2data[:,i] * 180 / np.pi, 'o', color=col[i - min(humn.right_legjnts)], label=f'th_act_{i - min(humn.right_legjnts) + 1+len(humn.left_legjnts)}')
        ax1[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax1[1].lines))
    for i in humn.ub_jnts:  # model.nv):
        ax1[2].plot(Xdata, Y1data[:,i] * 180 / np.pi, '-', color=col[(i - min(humn.ub_jnts))%6], label=f'th_des_{i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts)}')
        ax1[2].plot(Xdata, Y2data[:,i] * 180 / np.pi, 'o', color=col[(i - min(humn.ub_jnts))%6], label=f'th_act_{i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts)}')
        ax1[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax1[2].lines))

    ax1[2].set_xlabel('Time (s)')
    ax1[0].set_ylabel('Angle (deg)')
    ax1[0].grid(visible=None, which='major', axis='both')
    ax1[1].set_ylabel('Angle (deg)')
    ax1[1].grid(visible=None, which='major', axis='both')
    ax1[2].set_ylabel('Angle (deg)')
    ax1[2].grid(visible=None, which='major', axis='both')

    # plt.figure(11)  # Joint velocity     # Act vs Des dq/dt
    fig11, ax11 = plt.subplots(nrows=3, sharex=True)  # joint rates

    for i in humn.left_legjnts:  # model.nv):
        ax11[0].plot(Xdata, Y3data[:,i], '-', color=col[i - min(humn.left_legjnts)], label=f'dth_des_{i - min(humn.left_legjnts) + 1}')
        ax11[0].plot(Xdata, Y4data[:,i], 'o', color=col[i - min(humn.left_legjnts)], label=f'dth_act_{i - min(humn.left_legjnts) + 1}')
        ax11[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax11[0].lines) )
    for i in humn.right_legjnts:  # model.nv):
        ax11[1].plot(Xdata, Y3data[:,i], '-', color=col[i - min(humn.right_legjnts)], label=f'dth_des_{i - min(humn.right_legjnts) + 1+len(humn.left_legjnts)}')
        ax11[1].plot(Xdata, Y4data[:,i], 'o', color=col[i - min(humn.right_legjnts)], label=f'dth_act_{i - min(humn.right_legjnts) + 1+len(humn.left_legjnts)}')
        ax11[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax11[1].lines))
    for i in humn.ub_jnts:  # model.nv):
        ax11[2].plot(Xdata, Y3data[:,i], '-', color=col[(i - min(humn.ub_jnts))%6], label=f'dth_des_{i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts)}')
        ax11[2].plot(Xdata, Y4data[:,i], 'o', color=col[(i - min(humn.ub_jnts))%6], label=f'dth_act_{i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts)}')
        ax11[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax11[2].lines))

    ax11[2].set_xlabel('Time (s)')
    ax11[0].set_ylabel('Joint rate (rad/s)')
    ax11[0].grid(visible=None, which='major', axis='both')
    ax11[1].set_ylabel('Joint rate (rad/s)')
    ax11[1].grid(visible=None, which='major', axis='both')
    ax11[2].set_ylabel('Joint rate (rad/s)')
    ax11[2].grid(visible=None, which='major', axis='both')


    # plt.figure(2) # COM, COP position
    fig2, ax2 = plt.subplots(nrows=3, sharex=True)  # COM and COP Position with time
    #fig2.suptitle('COM and COP traj')

    Xdata = np.empty((0))
    Y1data = np.empty((0, 3))
    Y2data = np.empty((0, 3))
    Y3data = np.empty((0, 3))
    Y4data = np.empty((0, 3))
    for idata in DesData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[3]]),axis=0) #rcom
        Y2data=np.append(Y2data,np.array([idata[7]]),axis=0) #rcop

    for idata in ActData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y3data=np.append(Y3data,np.array([idata[3]]),axis=0) #rcom
        Y4data=np.append(Y4data,np.array([idata[7]]),axis=0) #rcop

    print('Tracking RMS error in COM_x = ', np.linalg.norm(Y1data[:,0]-Y3data[:,0])/np.sqrt(len(Y1data[:,0]))*1e3,'mm')
    print('Tracking RMS error in COM_y = ', np.linalg.norm(Y1data[:,1]-Y3data[:,1])/np.sqrt(len(Y1data[:,1]))*1e3,'mm')
    print('Tracking RMS error in COM_z = ', np.linalg.norm(Y1data[:,2]-Y3data[:,2])/np.sqrt(len(Y1data[:,2]))*1e3,'mm')
    # print('Tracking RMS error in ZMP_x = ', np.linalg.norm(Y2data[:,0]-Y4data[:,0])/np.sqrt(len(Y2data[:,0]))*1e3,'mm')
    # print('Tracking RMS error in ZMP_y = ', np.linalg.norm(Y2data[:,1]-Y4data[:,1])/np.sqrt(len(Y2data[:,1]))*1e3,'mm')
    # print('Tracking RMS error in ZMP_z = ', np.linalg.norm(Y2data[:,2]-Y4data[:,2])/np.sqrt(len(Y2data[:,2]))*1e3,'mm')

    for i in np.arange(0, 3):
        ax2[i].plot(Xdata, Y1data[:,i], '-r', label=f'COM_des')  # COM_des
        ax2[i].plot(Xdata, Y3data[:,i], '.r', label=f'COM_act')  # COM
        ax2[i].plot(Xdata, Y2data[:,i], '-g', label=f'COP_des')  # COP_des
        ax2[i].plot(Xdata, Y4data[:,i], '.g', label=f'COP_act')  # COP
        ax2[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax2[0].lines))
    # plt.figure()

    ax2[2].set_xlabel('Time (s)')
    ax2[0].set_ylabel('X (m)')
    ax2[0].grid(visible=None, which='major', axis='both')
    ax2[1].set_ylabel('Y (m)')
    ax2[1].grid(visible=None, which='major', axis='both')
    ax2[2].set_ylabel('Z (m)')
    ax2[2].grid(visible=None, which='major', axis='both')


    plt.figure()  # COM-COP
    # fig4 = plt.figure(4)  # COM and COP
    plt.plot(Y1data[:,0], Y1data[:,1], '-r', label=f'COM_des')
    plt.plot(Y3data[:,0], Y3data[:,1], '.r', label=f'COM_act')
    plt.plot(Y2data[:,0], Y2data[:,1], '-g', label=f'COP_des')
    plt.plot(Y4data[:,0], Y4data[:,1], '.g', label=f'COP_act')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 4)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.grid(visible=None, which='major', axis='both')


    # plt.figure() # Position with time
    fig3, ax3 = plt.subplots(nrows=4, sharex=True)  # Hip, Leg Position with time
    #fig3.suptitle('Hip and Foot traj')
    # Hip
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(DesData[0][1])))
    Y2data = np.empty((0, len(ActData[0][1])))
    for idata in DesData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[1]]), axis=0) #qdes
    for idata in ActData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y2data=np.append(Y2data,np.array([idata[1]]), axis=0) #q

    for i in np.arange(0, 3):
        ax3[i].plot(Xdata, Y1data[:,i], '-r', label=f'Hip_des')  # Hip-des
        ax3[i].plot(Xdata, Y2data[:,i], '.r', label=f'Hip_act')  # Hip-act
        ax3[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax3[0].lines) )


    # Xdata = np.empty((0))
    Y3data = np.empty((0, len(ActData[0][5])))
    Y4data = np.empty((0, len(ActData[0][6])))
    Y5data = np.empty((0, len(ActData[0][5])))
    Y6data = np.empty((0, len(ActData[0][6])))
    for idata in DesData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y3data=np.append(Y3data,np.array([idata[5]]), axis=0) #oLeft
        Y4data=np.append(Y4data,np.array([idata[6]]), axis=0) #oRight

    for idata in ActData:
        # Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y5data=np.append(Y5data,np.array([idata[5]]), axis=0) #oLeft
        Y6data=np.append(Y6data,np.array([idata[6]]), axis=0) #oRight

    for i in np.arange(0, 2):
        ax3[i].plot(Xdata, Y3data[:,i], '-g', label=f'Left-Foot_des')  # Left leg
        ax3[i].plot(Xdata, Y5data[:,i], '.g', label=f'Left-Foot_act')  # Left leg
        ax3[i].plot(Xdata, Y4data[:,i], '-b', label=f'Right-Foot_des')  # Right leg
        ax3[i].plot(Xdata, Y6data[:,i], '.b', label=f'Right-Foot_act')  # Right leg
        ax3[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax3[0].lines))
    i=2
    ax3[i+1].plot(Xdata, Y3data[:,i], '-g', label=f'Left-Foot_des')  # Left leg
    ax3[i+1].plot(Xdata, Y5data[:,i], '.g', label=f'Left-Foot_act')  # Left leg
    ax3[i+1].plot(Xdata, Y4data[:,i], '-b', label=f'Right-Foot_des')  # Right leg
    ax3[i+1].plot(Xdata, Y6data[:,i], '.b', label=f'Right-Foot_act')  # Right leg
    # ax3[i+1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax3[i+1].lines))

    # plt.figure()
    ax3[3].set_xlabel('Time (s)')
    ax3[0].set_ylabel('X (m)')
    ax3[0].grid(visible=None, which='major', axis='both')
    ax3[1].set_ylabel('Y (m)')
    ax3[1].grid(visible=None, which='major', axis='both')
    ax3[2].set_ylabel('Z (m)')
    ax3[2].grid(visible=None, which='major', axis='both')
    ax3[3].set_ylabel('Z (m)')
    ax3[3].grid(visible=None, which='major', axis='both')

    # plt.figure() # XY position
    fig33, ax33 = plt.subplots(nrows=2, sharex=True)  # HIP and Foot Traj XY
    ax33[0].plot(Y1data[:,0], Y1data[:,2], '-r', label=f'Hip_des')  # Hip-des
    ax33[0].plot(Y2data[:,0], Y2data[:,2], '.r', label=f'Hip_act')  # Hip-act
    ax33[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax33[0].lines) )

    ax33[1].plot(Y3data[:,0], Y3data[:,2], '-g', label=f'Left-Foot_des')  # Left leg
    ax33[1].plot(Y5data[:,0], Y5data[:,2], '.g', label=f'Left-Foot_act')  # Left leg
    ax33[1].plot(Y4data[:,0], Y4data[:,2], '-b', label=f'Right-Foot_des')  # Right leg
    ax33[1].plot(Y6data[:,0], Y6data[:,2], '.b', label=f'Right-Foot_act')  # Right leg
    ax33[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax33[1].lines))
    # plt.figure()
    ax33[1].set_xlabel('X (m)')
    ax33[0].set_ylabel('Z (m)')
    ax33[0].grid(visible=None, which='major', axis='both')
    ax33[1].set_ylabel('Z (m)')
    ax33[1].grid(visible=None, which='major', axis='both')

    plt.figure()  # Normal Contact force
    # fig5 = plt.figure(5)  # Contact force
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(ActData[0][8])))
    Y2data = np.empty((0, len(ActData[0][9])))
    for idata in ActData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[8]]), axis=0) #fcl
        Y2data=np.append(Y2data,np.array([idata[9]]), axis=0) #fcr
    plt.plot(Xdata, Y1data[:,0], '-g', label=f'Left foot')
    plt.plot(Xdata, Y2data[:,0], '-b', label=f'Right foot')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = 2)

    #plt.figure(5)
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.grid(visible=None, which='major', axis='both')


    # plt.figure(6) # Applied joint torque
    fig6, ax6 = plt.subplots(nrows=3, sharex=True)  # Torque
    # fig6.suptitle('Joint torque')
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(ActData[0][10])))
    for idata in ActData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[10]]), axis=0) #data.ctrl

    for i in humn.left_legjnts:
        # plt.plot(data.time, tauid[i], '.r')
        ax6[0].plot(Xdata, Y1data[:,i - 6], color=col[i - min(humn.left_legjnts)], label=f'tau_{i - min(humn.left_legjnts) + 1}')
        ax6[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax6[0].lines))
    for i in humn.right_legjnts:
        # plt.plot(data.time, tauid[i], '.r')
        ax6[1].plot(Xdata, Y1data[:,i - 6], color=col[i - min(humn.right_legjnts)], label=f'tau_{i - min(humn.right_legjnts) + 1+len(humn.left_legjnts)}')
        ax6[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax6[1].lines))
    for i in humn.ub_jnts:
        # plt.plot(data.time, tauid[i], '.r')
        ax6[2].plot(Xdata, Y1data[:,i - 6], color=col[(i - min(humn.ub_jnts))%6], label=f'tau_{i - min(humn.ub_jnts) + 1+len(humn.left_legjnts)+len(humn.right_legjnts)}')
        ax6[2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol = len(ax6[2].lines))

    #plt.figure(6)
    ax6[2].set_xlabel('Time (s)')
    ax6[0].set_ylabel('Torque (Nm)')
    ax6[0].grid(visible=None, which='major', axis='both')
    ax6[1].set_ylabel('Torque (Nm)')
    ax6[1].grid(visible=None, which='major', axis='both')
    ax6[2].set_ylabel('Torque (Nm)')
    ax6[2].grid(visible=None, which='major', axis='both')


    # plt.figure(7)
    fig7, ax7 = plt.subplots(nrows=3, sharex=True)  # Hip, Leg Position with time
    # fig7.suptitle('Angular momentum')
    Xdata = np.empty((0))
    Y1data = np.empty((0, len(ActData[0][11])))
    for idata in ActData:
        Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
        Y1data=np.append(Y1data,np.array([idata[11]]), axis=0) # Iw*dq

    for i in range(3):
        ax7[i].plot(Xdata, Y1data[:,i], '-r')  # Ang momentum
    # plt.figure(7)
    ax7[2].set_xlabel('Time (s)')
    ax7[0].set_ylabel('Lx (kg-m^2/s)')
    ax7[0].grid(visible=None, which='major', axis='both')
    ax7[1].set_ylabel('Ly (kg-m^2/s)')
    ax7[1].grid(visible=None, which='major', axis='both')
    ax7[2].set_ylabel('Lz (kg-m^2/s)')
    ax7[2].grid(visible=None, which='major', axis='both')

    plt.figure() #COT
    # fig8 = plt.figure(8)  # COT
    Xdata = np.empty((0))
    Y1data = np.empty((0))
    #Y2data = np.empty((0, 3))
    rcmX0=ActData[0][3][0]
    for idata in ActData:
        rcmX=idata[3][0]-rcmX0
        if rcmX>0:
            Xdata=np.append(Xdata,np.array([idata[0]]), axis=0) #time
            Y1data=np.append(Y1data,np.array([idata[12]/(humn.m*9.81*rcmX)]),axis=0) #WD/(m*g*rcomX)
            #Y2data=np.append(Y2data,np.array([idata[12]]),axis=0) #WD
        #Y3data=np.append(Y3data,np.array([idata[11]]),axis=0) #drcom
    plt.plot(Xdata, Y1data, '-r', label=f'COT')
    # plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('COT')
    plt.grid(visible=None, which='major', axis='both')

    plt.show()


# DAIR LAB's cube
import os.path as op

ROOT_DIR = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
def root_path(*path: str) -> str:
    return op.join(ROOT_DIR, *path)

DATA_DIR = root_path(r"ContactNets Cube")
OUT_DIR = root_path('ContactNets Cube\out')
LIB_DIR = root_path('ContactNets Cube\lib')
RESULTS_DIR = root_path('ContactNets Cube\results')

def out_path(*path: str) -> str:
    return op.join(OUT_DIR, *path)

def data_path(*path: str) -> str:
    return op.join(DATA_DIR, *path)

def results_path(*path: str) -> str:
    return op.join(RESULTS_DIR, *path)


def lib_path(*path: str) -> str:
    return op.join(LIB_DIR, *path)

PROCESSING_DIR = root_path('contactnets', 'utils', 'processing')
def processing_path(*path: str):
    return op.join(PROCESSING_DIR, *path)

import distutils.dir_util
"""
# Copy the tosses data and processing parameters into the working directory
distutils.dir_util.copy_tree(data_path('DAIRLab contact-nets main data-tosses_processed'),
                             out_path('data', 'all'))
distutils.dir_util.copy_tree(data_path('params_processed'),
                             out_path('params'))
"""
