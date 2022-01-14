from tkinter import *
from math import *
import numpy as np
import time

PI = 3.14159265

canvas_width = 500
canvas_height = 500

dt = 0.01
origin = np.array([250, 400])

m1 = 1
l1 = 1
m2 = 1
l2 = 0.7
gravity = 0.01
j1 = 1 
j2 = 1

K1 = 0
K2 = 0
K3 = 0
K4 = 0

runTime = 5
numIterations = int(runTime / dt)
lamb = 1
lambFactor = 10
lambMax = 10000

desiredPos = np.array((0.5, 0.5))
desiredState = np.zeros((2))

scaling = 100

#4x4 Matrix
Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

R = np.array([[0, 0], [0, 0]])

master = Tk()
master.title( "ILQR - 2 Link Planar Arm" )
canvas = Canvas(master, width=canvas_width, height=canvas_height)
canvas.pack(expand = YES, fill = BOTH)

def main():
    global dt 
    global l1, l2
    global origin
    global desiredState
    global canvas
    global K1, K2, K3, K4
    currentIteration = 0
    K1 = ((1/3. * m1 + m2) * l1**2. + 1/3. * m2 * l2**2.)
    K2 = m2 * l1 * l2
    K3 = 1/3. * m2 * l2**2.
    K4 = 1/2. * m2 * l1 * l2
    U = initControls(numIterations)
    # State = [theta1, theta2, theta1dot, theta2dot]
    # 4x1 Matrix
    X = np.zeros((numIterations, 4))
    X0 = np.array([0.524, 0.35, 0, 0])
    invKin =  inverseKinematics(l1, l2, desiredPos[0], desiredPos[1])
    desiredState = invKin[1]

    X[0] = X0

    testX0 = np.array([PI/2, PI/2])
    testX1 = np.array([3*PI/4, PI])
    anglediff = calcAngleDiffConstrained(testX1, testX0)
    a = 1

    #testAngle = (-10 * PI)
    #testAngle = constrainAngleBetween0and2PI(testAngle)

    #testAngle = (-10*PI - (PI/2))
    #testAngle = constrainAngleBetween0and2PI(testAngle)
    #a=1

    #while(currentIteration < numIterations):
    #    draw(canvas, [l1, l2], X[currentIteration], origin,desiredState)
    #    master.update_idletasks()
    #    master.update()
    #    X[currentIteration+1, :],_ = simulateDynamicsOneTimeStep(X[currentIteration, :], U[currentIteration,:])
    #    currentIteration = currentIteration + 1
    #    time.sleep(dt)
    
    
    U = ilqr(X0, U)

    firstTimeCalc = True

    while(1):

        if firstTimeCalc == True:
            draw(canvas, [l1, l2], X[currentIteration], origin,desiredState)
            master.update_idletasks()
            master.update()
            X[currentIteration+1, :],_ = simulateDynamicsOneTimeStep(X[currentIteration, :], U[currentIteration,:])
            currentIteration = currentIteration + 1
            if currentIteration >= numIterations - 1:
                currentIteration = 0
                firstTimeCalc = True
                time.sleep(2)

        else:
            draw(stateAngles[currentIteration])
            master.update_idletasks()
            master.update()
            currentIteration = currentIteration + 1
            if currentIteration >= numIterations - 1:
                currentIteration = 0
                time.sleep(2)
        time.sleep(dt)


def draw(canvas, armLengths, currentConfiguration, canvasOrigin, desiredState):

    global desiredPos
    global scaling
    goalColour = "#eb0e0e"
    pointSize = 5

    desiredStateScaled = np.zeros((2))
    desiredStateScaled[0] = (desiredPos[0] * scaling) + canvasOrigin[0]
    desiredStateScaled[1] = canvasOrigin[1] - (desiredPos[1] * scaling)

    canvas.delete('all')
    jointPos = forwardKinematics(l1 * scaling, currentConfiguration[0], l2 * scaling, currentConfiguration[1], canvasOrigin, True)
    drawArm(jointPos)

    canvas.create_oval( desiredStateScaled[0] - pointSize, desiredStateScaled[1] - pointSize, desiredStateScaled[0] + pointSize, desiredStateScaled[1] + pointSize, fill = goalColour )

def drawArm(jointPos):
    global origin
    armColour = "#34d8eb"
    endEffectorColour = "#000000"

    intermediatePointX = jointPos[0, 0]
    intermediatePointY = jointPos[0, 1]

    canvas.create_line(origin[0], origin[1], intermediatePointX, intermediatePointY, fill=armColour, width = 10)

    Xe = jointPos[1, 0]
    Ye = jointPos[1, 1]

    canvas.create_line(intermediatePointX, intermediatePointY, Xe, Ye, fill=armColour, width = 10)

    pointSize = 5
    canvas.create_oval( Xe - pointSize, Ye - pointSize, Xe + pointSize, Ye + pointSize, fill = endEffectorColour )
    canvas.create_oval( intermediatePointX - pointSize, intermediatePointY - pointSize, intermediatePointX + pointSize, intermediatePointY + pointSize, fill = endEffectorColour )


def forwardKinematics(l1, theta1, l2, theta2, origin, draw):
    jointPos = np.zeros((2, 2))

    intermediatePointX = origin[0] + (l1 * cos(theta1))
    if draw:
        intermediatePointY = origin[1] - (l1 * sin(theta1))
    else:
        intermediatePointY = origin[1] + (l1 * sin(theta1))

    Xe = intermediatePointX + (l2 * cos(theta1 + theta2))
    if draw:
        Ye = intermediatePointY - (l2 * sin(theta1 + theta2))
    else:
        Ye = intermediatePointY + (l2 * sin(theta1 + theta2))

    jointPos[0, 0] = intermediatePointX
    jointPos[0, 1] = intermediatePointY

    jointPos[1, 0] = Xe
    jointPos[1, 1] = Ye

    return jointPos

def inverseKinematics(l1, l2, Xe, Ye):
    currentTheta_r = np.zeros(2)
    validSol = True

    numerator = pow(Xe,2) + pow(Ye,2) - pow(l1, 2) - pow(l2, 2)
    denominator = 2 * l1 * l2 
    try:
        q2 = acos(numerator/denominator)
        numerator = l2 * sin(q2)
        denominator = l1 + l2*cos(q2)

        q1 = atan(Ye/Xe) - atan(numerator/denominator)

        currentTheta_r[0] = q1 
        currentTheta_r[1] = q2
    except:
        validSol = False

    return validSol, currentTheta_r


def initControls(tN):

    U = np.zeros((tN, 2))
    for i  in range(tN):
        U[i, 0] = 0
        U[i, 1] = 0

    return U

def forwardPass(X0, U):
    global l1, l2, dt
    origin = np.array([0, 0])
    tN = U.shape[0]
    numStates = X0.shape[0]
    X = np.zeros((tN, numStates))
    X[0,:] = X0
    cost = 0 

    for t in range(tN - 1):
        l,_,_,_,_,_ = immediateCost(X[t, :], U[t])
        X[t+1, :],_ = simulateDynamicsOneTimeStep(X[t, :], U[t,:])

        
        cost = cost + (l*dt)

    l_f,_,_ = terminalCost(X[-1])
    cost = cost + l_f


    return X, cost

def simulateDynamicsOneTimeStep(X, U):
    global dt
    global l1, l2, m1, m2, j1, j2 
    global gravity
    global K1, K2, K3, K4
    dof = U.shape[0]

    # torque = (M(theta) * theta dot dot) + C(theta, theta dot) + g(theta)
    # Initialise matrices
    M = np.zeros((2,2))
    C = np.zeros((2))
    g = np.zeros((2))
    angularAccel = np.zeros((2))

    # Setup all the equations of motion matrices
    M[0, 0] = (m1 * pow(l1,2)) + (m2 * pow(l1,2)) + (m2 * pow(l2,2)) + (2 * m2 * l1 * l2 * cos(X[1]))
    M[0, 1] = (m2 * pow(l2, 2)) + (m2 * l1 * l2 * cos(X[1]))
    M[1, 0] = (m2 * pow(l2, 2)) + (m2 * l1 * l2 * cos(X[1]))
    M[1, 1] = m2 * pow(l2, 2)

    C[0] = -m2 * l1 * l2 * sin(X[1]) * (2*X[2]*X[3] + pow(X[3], 2))
    C[1] = m2 * l1 * l2 * pow(X[2], 2) * sin(X[1])

    interm = (m2 * gravity * l2 * cos(X[0] + X[1]))
    g[0] = ((m1 + m2) * (l1 * gravity *cos(X[0]))) + interm
    g[1] = m2 * gravity * l2 * cos(X[0] + X[1])

    # Invert M matrix to enable us to rearrange equation to solve for angular acceleration
    MInv = np.linalg.inv(M)

    UConstrained = U.copy()
    for i in range(dof):
        if(UConstrained[i] > 100):
            UConstrained[i] = 100
        if(UConstrained[i] < -100):
            UConstrained[i] = -100

    
    angularAccel = MInv @ (UConstrained - C - g)

    Xnext = np.zeros((4))
    Xdot = np.zeros((4))
            
    # Integrate velocity with acceleration

    Xnext[2] = X[2] + (angularAccel[0] * dt)
    Xnext[3] = X[3] + (angularAccel[1] * dt)

    for i in range(2):
        if(Xnext[2 + i] >  100):
            Xnext[2 + i] = 100
        if(Xnext[2 + i] <  -100):
            Xnext[2 + i] = -100

    tempCopy = Xnext.copy()

    ## Integrate position vector with velocity
    Xnext[0] = X[0] + (tempCopy[2] * dt)
    Xnext[1] = X[1] + (tempCopy[3] * dt)

    # Constraina ngles between 0 and 2 pi
    #if(Xnext[0] > 2* PI):
    #    Xnext[0] = Xnext[0] - 2*PI
    #if(Xnext[0] < 0):
    #    Xnext[0] = Xnext[0] + 2*PI

    #if(Xnext[1] > 2* PI):
    #    Xnext[1] = Xnext[1] - 2*PI
    #if(Xnext[1] < 0):
    #    Xnext[1] = Xnext[1] + 2*PI

    Xdot = (Xnext - X) / dt

    return Xnext, Xdot

def immediateCost(X, U):
    global desiredState
    global desiredPos
    global l1, l2
    dof = U.shape[0]
    num_states = X.shape[0]

    #X_diff = np.zeros(2)
    #jointPos = forwardKinematics(l1, X[0], l2, X[1], [0, 0], False)
    #actualPos = jointPos[1,:]
    #X_diff[0] = (actualPos[0] - desiredPos[0])
    #X_diff[1] = (actualPos[1] - desiredPos[1])

    #X_diff = np.zeros(4)
    ## States 1 and 2 are difference from a desired joint angle
    #X_diff[0] = (X[0] - desiredState[0])
    #X_diff[1] = (X[1] - desiredState[1])

    #if(X_diff[0] > PI):
    #    X_diff[0] = (2 * PI) - X_diff[0]

    #if(X_diff[1] > PI):
    #    X_diff[1] = (2 * PI) - X_diff[1]

    #if(X_diff[0] < -PI):
    #    X_diff[0] = (-2 * PI) - X_diff[0]

    #if(X_diff[1] < -PI):
    #    X_diff[1] = (-2 * PI) - X_diff[1]

    ## States 3 and 4 are velocity and we want these to end at 0
    #X_diff[2] = X[2]
    #X_diff[3] = X[3]

    ## Instant state cost = 0.5X_t*Q*X + 0.5U_t*R*U
    #l = (0.5 * np.transpose(X_diff) @ Q @ X_diff) + (0.5 * np.transpose(U) @ R @ U)

    #if(l > 400):
    #    a = 1
    
    #l_x = Q @ X_diff
    #l_xx = Q
    #l_u = R @ U
    #l_uu = R
    #l_ux = np.zeros((dof, num_states))

    l = (0.5 * np.transpose(U) @ R @ U)
    l_x = np.zeros(num_states)
    l_xx = np.zeros((num_states, num_states))
    l_u = R @ U
    l_uu = R
    l_ux = np.zeros((dof, num_states))


    return l, l_x, l_xx, l_u, l_uu, l_ux

def terminalCost(X):
    global desiredState
    
    num_states = X.shape[0]


    #X_diff = np.zeros(4)
    ## States 1 and 2 are difference from a desired joint angle
    #X_diff[0] = (X[0] - desiredState[0])
    #X_diff[1] = (X[1] - desiredState[1])
    ## States 3 and 4 are velocity and we want these to end at 0
    #X_diff[2] = X[2]
    #X_diff[3] = X[3]

    #num_states = X.shape[0]
    #l_x = np.zeros((num_states))
    #l_xx = np.zeros((num_states, num_states))

    #l =  50 * (0.5 * np.transpose(X_diff) @ Q @ X_diff)
    #l_x = 25 * Q @ X_diff
    #l_xx = 25 * Q

    #calculate current end effector position 
    
    l = calcTerminalCost(X)
    l_x = np.zeros((4))
    l_xx = np.zeros((4, 4))
    

    eps = 1e-1
    #add, add, minus, minus
    
    l_x = calcFirstOrderCostChange(X, eps)

    for i in range(num_states):

        incX = X.copy()
        incX2 = X.copy()
        decX = X.copy()
        decX2 = X.copy()

        incX[i] += eps 
        incX2[i] += (2 * eps)
        decX[i] -= eps 
        decX2[i] -= (2 * eps)

        incXCost = calcFirstOrderCostChange(incX, eps)
        decXCost = calcFirstOrderCostChange(decX, eps)

        l_xx[:,i] = (incXCost - decXCost) / (2 * eps)



    # Final cost only requires these three values
    return l, l_x, l_xx

def calcFirstOrderCostChange(X, eps):
    l_x = np.zeros((4))
    num_states = X.shape[0]

    for i in range(num_states):
        incX = X.copy()
        decX = X.copy()

        incX[i] += eps 
        decX[i] -= eps 

        incXCost = calcTerminalCost(incX)
        decXCost = calcTerminalCost(decX)

        l_x[i] = (incXCost - decXCost) / (2 * eps)


    return l_x

def calcTerminalCost(X):
    global desiredPos
    global l1, l2

    posFactor = 10
    velFactor = 0.1

    endEffectorPos = np.zeros((2))
    xyErr = np.zeros((2))
    velErr = np.zeros((2))
    endEffectorPos[0] = (l1 * cos(X[0])) + (l2*cos(X[0] + X[1]))
    endEffectorPos[1] = (l1 * sin(X[0])) + (l2*sin(X[0] + X[1]))

    xyErr[0] = endEffectorPos[0] - desiredPos[0]
    xyErr[1] = endEffectorPos[1] - desiredPos[1]

    #xyErr = endEffectorPos - desiredPos[0:1]
    velErr = X[2:4]
    l = (posFactor * np.sum(xyErr**2)) + (velFactor * np.sum(velErr**2))

    return l

def finiteDifferencing(X, U):

    dof = U.shape[0]
    num_states = X.shape[0]

    A = np.zeros((num_states, num_states))
    B = np.zeros((num_states, dof))

    eps = 1e-5
    for ii in range(num_states):
        inc_x = X.copy()
        inc_x[ii] += eps 
        _,stateInc = simulateDynamicsOneTimeStep(inc_x, U.copy())

        dec_x = X.copy()
        dec_x[ii] -= eps
        _,stateDec = simulateDynamicsOneTimeStep(dec_x, U.copy())

        A[:, ii] = ((stateInc - stateDec) / (2 * eps)) #/ (dt*dt)


    for ii in range(dof):
        inc_u = U.copy()
        inc_u[ii] += eps
        _,stateInc = simulateDynamicsOneTimeStep(X.copy(), inc_u)

        dec_u = U.copy()
        dec_u[ii] -= eps
        _,stateDec = simulateDynamicsOneTimeStep(X.copy(), dec_u)

        B[:, ii] = ((stateInc - stateDec) / (2 * eps)) #/ (dt*dt)


    #print(A)
    #print(B)
    return A, B

def ilqr(X0, U):
    maxIterations = 1000
    global lamb
    global lambFactor
    global lambMax
    global dt
    overOldCostLast = False
    origin = np.array([0,0])
    epsConverge = 0.00001

    tN = U.shape[0] # number of time steps
    dof = 2 # number of degrees of freedom of plant 
    num_states = 4 # number of states (position and velocity)

    X, oldCost = forwardPass(X0, U)
    for i in range(maxIterations):

        # Initiialise all relevant partial differentation matrtices for all time steps 
        f_x = np.zeros((tN, num_states, num_states)) # df / dx
        f_u = np.zeros((tN, num_states, dof)) # df / du
        # for storing quadratized cost function 
        l = np.zeros((tN,1)) # immediate state cost 
        l_x = np.zeros((tN, num_states)) # dl / dx
        l_xx = np.zeros((tN, num_states, num_states)) # d^2 l / dx^2
        # POTENTIAL ISSUE HERE, changed size of array with extra 1 at the end
        l_u = np.zeros((tN, dof)) # dl / du
        l_uu = np.zeros((tN, dof, dof)) # d^2 l / du^2
        l_ux = np.zeros((tN, dof, num_states)) # d^2 l / du / dx
        # for everything except final state

        #Forwards pass
        for t in range(tN - 1):
            # x(t+1) = f(x(t), u(t)) = x(t) + (dx(t) * dt)
            # x(t). = AX(t) + BU(t)
            # Calculate A and B via finitie differencing
            # Calculate f_x, f_u using A and B(linearied approximation to dynamics with respect to state and controls) 
            A, B = finiteDifferencing(X[t], U[t])
            f_x[t] = np.eye(num_states) + A * dt
            #f_x[t] = A * dt
            f_u[t] = B * dt

            # calculate l, l_x, l_xx, l_u, l_uu, l_ux
            # Multiply them by dt (time step)
            l[t],l_x[t],l_xx[t],l_u[t],l_uu[t],l_ux[t] = immediateCost(X[t, :], U[t])
            l[t] *= dt
            l_x[t] *= dt
            l_xx[t] *= dt
            l_u[t] *= dt
            l_uu[t] *= dt
            l_ux[t] *= dt

            # Calculate terminal cost (maybe heavy penalties for nopt reaching goal state
        l[-1], l_x[-1], l_xx[-1] = terminalCost(X[-1])

        V = l[-1].copy() # value function
        V_x = l_x[-1].copy() # dV / dx
        V_xx = l_xx[-1].copy() # d^2 V / dx^2
        k = np.zeros((tN, dof)) # feedforward modification
        K = np.zeros((tN, dof, num_states)) # feedback gain

         # Time to optimise our control sequence 
        #Set Value function equal to terminal cost function
        # initialise other matrices, V_x, V_xx, k , K
        for t in range(tN-2, -1, -1):
            #NOTE: we're working backwards, so V_x = V_x[t+1] = V'_x

            # 4a) Q_x = l_x + np.dot(f_x^T, V'_x)
            Q_x = l_x[t] +  f_x[t].T @ V_x
            Q_x = l_x[t] + np.dot(f_x[t].T, V_x) 
            # 4b) Q_u = l_u + np.dot(f_u^T, V'_x)
            Q_u = l_u[t] + np.dot(f_u[t].T, V_x)
            Q_u = l_u[t] + f_u[t].T @ V_x

            # NOTE: last term for Q_xx, Q_uu, and Q_ux is vector / tensor product
            # but also note f_xx = f_uu = f_ux = 0 so they're all 0 anyways.
                
            # 4c) Q_xx = l_xx + np.dot(f_x^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_xx)
            Q_xx = l_xx[t] + np.dot(f_x[t].T, np.dot(V_xx, f_x[t])) 
            # 4d) Q_ux = l_ux + np.dot(f_u^T, np.dot(V'_xx, f_x)) + np.einsum(V'_x, f_ux)
            Q_ux = l_ux[t] + np.dot(f_u[t].T, np.dot(V_xx, f_x[t]))
            # 4e) Q_uu = l_uu + np.dot(f_u^T, np.dot(V'_xx, f_u)) + np.einsum(V'_x, f_uu)

            Q_uu = l_uu[t] + np.dot(f_u[t].T, np.dot(V_xx, f_u[t]))

            # Calculate Q_uu^-1 with regularization term set by 
            # Levenberg-Marquardt heuristic (at end of this loop)
            Q_uu_evals, Q_uu_evecs = np.linalg.eig(Q_uu)
            Q_uu_evals[Q_uu_evals < 0] = 0.0
            Q_uu_evals += lamb
            Q_uu_inv = np.dot(Q_uu_evecs, 
                    np.dot(np.diag(1.0/Q_uu_evals), Q_uu_evecs.T))

            # 5b) k = -np.dot(Q_uu^-1, Q_u)
            k[t] = -np.dot(Q_uu_inv, Q_u)
            # 5b) K = -np.dot(Q_uu^-1, Q_ux)
            K[t] = -np.dot(Q_uu_inv, Q_ux)

            # 6a) DV = -.5 np.dot(k^T, np.dot(Q_uu, k))
            # 6b) V_x = Q_x - np.dot(K^T, np.dot(Q_uu, k))
            V_x = Q_x - np.dot(K[t].T, np.dot(Q_uu, k[t]))
            # 6c) V_xx = Q_xx - np.dot(-K^T, np.dot(Q_uu, K))
            V_xx = Q_xx - np.dot(K[t].T, np.dot(Q_uu, K[t]))

        Unew = np.zeros((tN, dof))
        # calculate the optimal change to the control trajectory
        xnew = X[0].copy() # 7a)

        for t in range(tN - 1): 
            # use feedforward (k) and feedback (K) gain matrices 
            # calculated from our value function approximation
            # to take a stab at the optimal control signal

            #testX = X[t].copy
            #testXnew = 
            if(t == 60):
                a = 1
            angleDiff = calcAngleDiffConstrained(X[t].copy(), xnew.copy())
            stateFeedback = np.zeros((4))
            stateFeedback[0] = angleDiff[0]
            stateFeedback[1] = angleDiff[1]
            stateFeedback[2] = xnew[2]
            stateFeedback[3] = xnew[3]

            #stateFeedback = np.concatenate(angleDiff, xnew[2:4])
            intermediate = np.dot(K[t], stateFeedback)

            # Add an alpha term between 0 and 1 for lower case k
            Unew[t] = U[t] + k[t] + intermediate # 7b)
            # given this u, find our next state
            xnew,_ = simulateDynamicsOneTimeStep(xnew, Unew[t])
            if t > 100:
                a = 1
            #print("made it past time: " + str(t))

        # evaluate the new trajectory 
        Xnew, newCost = forwardPass(X0, Unew)
        print("---------------------------------------------")
        print("terminal cost is")
        print(terminalCost(X[-1]))
        print("terminal state is")
        print(X[-1])
        print("terminal diff is")
        print(diff(X[-1]))
        print("old cost ")
        print(oldCost)
        print("new cost")
        print(newCost)
        print("current lambda")
        print(lamb)
        #print("lamda factor")
        #print(lambFactor)
        print("--------------------------------------------------")

        # Levenberg-Marquardt heuristic
        if newCost < oldCost: 
            # decrease lambda (get closer to Newton's method)
            if lamb > 1e-30:
                lamb /= lambFactor
            if(overOldCostLast == True):
                pass
                #lambFactor -= 0.1
            overOldCostLast = False

            X = np.copy(Xnew) # update trajectory 
            U = np.copy(Unew) # update control signal
            

            # check to see if update is small enough to exit
            if i > 300 and ((abs(oldCost-newCost)/newCost) < epsConverge):
                print("Converged at iteration = %d; Cost = %.4f;"%(i,newCost) + 
                        " logLambda = %.1f"%np.log(lamb))
                break
            oldCost = np.copy(newCost)

        else: 
            # increase lambda (get closer to gradient descent)
            lamb *= lambFactor
            if(overOldCostLast == False):
                pass
                #lambFactor -= 0.1
            overOldCostLast = True
            # print("cost: %.4f, increasing lambda to %.4f")%(cost, lamb)
            if lamb > lambMax: 
                print("lambda > max_lambda at iteration = %d;"%i + 
                    " Cost = %.4f; logLambda = %.1f"%(oldCost, 
                                                        np.log(lamb)))
                break
    return U

def calcAngleDiffConstrained(X, Xnew):
    angleDiff = np.zeros((2))

    X[0] = constrainAngleBetween0and2PI(X[0])
    X[1] = constrainAngleBetween0and2PI(X[1])
    Xnew[0] = constrainAngleBetween0and2PI(Xnew[0])
    Xnew[1] = constrainAngleBetween0and2PI(Xnew[1])

    angleDiff[0] = Xnew[0] - X[0]
    if((Xnew[0] < PI) and (X[0] > PI)):
        angleDiff[0] = angleDiff[0]  * -1



    angleDiff[1] = Xnew[1] - X[1]
    if((Xnew[1] < PI) and (X[1] > PI)):
        angleDiff[1] = angleDiff[1]  * -1

    if(angleDiff[0] > PI):
        angleDiff[0] = angleDiff[0] - (2 * PI)

    if(angleDiff[0] < -PI):
        angleDiff[0] = angleDiff[0] + (2 * PI)


    if(angleDiff[1] > PI):
        angleDiff[1] = angleDiff[1] - (2 * PI)

    if(angleDiff[1] < -PI):
        angleDiff[1] = angleDiff[1] + (2 * PI)

    return angleDiff

def constrainAngleBetween0and2PI(angle):

    angleSub = angle - (2 * PI)
    if angleSub > 0:
        numPiSubtract = floor(angle / (2 * PI))
        angle -= (numPiSubtract * 2 * PI)

    angleAdd = angle + (2 * PI)
    if angleAdd < (2 * PI):
        numPiAdd = abs(ceil(angle / (2 * PI))) + 1
        angle += (numPiAdd * 2 * PI)

    return angle


def diff(X):
    global desiredState

    X_diff = np.zeros(2)
    X_diff[0] = (X[0] - desiredState[0])
    X_diff[1] = (X[1] - desiredState[1])

    return X_diff


main()