import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from collections import deque

def sample_arm_uniform(rewardProbabilities, arm):
    return int(float(rewardProbabilities[arm]) > np.random.rand())

def sample_arm(rewardProbabilities, arm):
    if arm < rewardProbabilities[0]:
        return int(float(rewardProbabilities[1]) > np.random.rand())
    else:
        return int(float(rewardProbabilities[2]) > np.random.rand())
    


def ts(gainedRewardArray, alpha,beta,armToChoose, rewardProbabilities, avgReward, numSelections, gainedReward, roundNum):
    sampled_means = np.random.beta(alpha, beta)
    selected_arms = np.argsort(sampled_means)[-armToChoose:]
    numSelections[selected_arms] = numSelections[selected_arms] + 1
    for j in selected_arms:
        reward = sample_arm(rewardProbabilities, j)
        alpha[j] = alpha[j] + reward
        beta[j] = beta[j] + 1 - reward
        avgReward[j] = avgReward[j] + (reward)/numSelections[j]
        gainedReward = gainedReward + reward
        gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward
    return gainedRewardArray, alpha, beta, avgReward, numSelections, gainedReward

def ts_delayed(gainedRewardArray, alphad, betad, numSelections, avgReward, numOfArms, armToChoose, roundNum, gainedReward, rewardProbabilities, fifo, fifoSize, delays, delayTime):
    if((np.size(np.argwhere(numSelections < delayTime))) > 0):
        #selections  = np.random.choice(numOfArms, armToChoose, replace=False)
        selections  = (np.arange(armToChoose) + roundNum % numOfArms) % numOfArms -1
        numSelections[selections] = numSelections[selections] + 1
        for j in selections:    
            loc = np.argwhere(delays[j] == -1)
            loc = loc[0][0]
            delays[j, loc] = delayTime

    else:
        sampled_means = np.random.beta(alphad, betad)
        selections = np.argsort(sampled_means)[-armToChoose:]
        for j in selections:
            tmpIdx = np.random.randint(0, fifoSize)
            reward = fifo[j, tmpIdx]
            alphad[j] = alphad[j] + reward
            betad[j] = betad[j] + 1 - reward
            loc = np.argwhere(delays[j] == -1)
            loc = loc[0][0]
            delays[j, loc] = delayTime
            avgReward[j] = avgReward[j] + (reward)/numSelections[j]
            
    if ((np.size(np.argwhere(delays == 0))) > 0):
        for m in np.argwhere(delays==0):
            
            tmp_queue = deque(fifo[m[0],:].ravel(), maxlen=fifoSize)
            reward = sample_arm(rewardProbabilities, m[0])
            tmp_queue.appendleft(reward)
            fifo[m[0],:] = np.array(tmp_queue)
            delays[tuple(m)] = -1
            gainedReward = gainedReward + reward
            gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward
            

    return gainedRewardArray, alphad, betad, numSelections, avgReward, gainedReward, fifo, delays

def ts_delayed_pure(gainedRewardArray, alphad, betad, numSelections, avgReward, numOfArms, armToChoose, roundNum, gainedReward, rewardProbabilities, fifo, fifoSize, delays, delayTime):
    if((np.size(np.argwhere(numSelections < delayTime))) > 0):
        #selections  = np.random.choice(numOfArms, armToChoose, replace=False)
        selections  = (np.arange(armToChoose) + roundNum % numOfArms) % numOfArms -1
        numSelections[selections] = numSelections[selections] + 1
        for j in selections:    
            loc = np.argwhere(delays[j] == -1)
            loc = loc[0][0]
            delays[j, loc] = delayTime

    else:
        sampled_means = np.random.beta(alphad, betad)
        selections = np.argsort(sampled_means)[-armToChoose:]
        for j in selections:
            loc = np.argwhere(delays[j] == -1)
            loc = loc[0][0]
            delays[j, loc] = delayTime
            
            
    if ((np.size(np.argwhere(delays == 0))) > 0):
        for m in np.argwhere(delays==0):
            
            tmp_queue = deque(fifo[m[0],:].ravel(), maxlen=fifoSize)
            reward = sample_arm(rewardProbabilities, m[0])

            alphad[j] = alphad[j] + reward
            betad[j] = betad[j] + 1 - reward
            tmp_queue.appendleft(reward)
            fifo[m[0],:] = np.array(tmp_queue)
            delays[tuple(m)] = -1
            gainedReward = gainedReward + reward
            gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward
            avgReward[j] = avgReward[j] + (reward)/numSelections[j]
            

    return gainedRewardArray, alphad, betad, numSelections, avgReward, gainedReward, fifo, delays



def ucb(gainedRewardArray, numSelections, avgReward, numOfArms, armToChoose, roundNum, gainedReward, rewardProbilities):

    if((np.size(np.argwhere(numSelections == 0))) > 0):
        #selections  = np.random.choice(numOfArms, armToChoose, replace=False)
        selections  = (np.arange(armToChoose) + roundNum % numOfArms) % numOfArms -1
    else:
        UCB = avgReward + np.sqrt(2*np.log(roundNum)/numSelections)
        selections = np.argpartition(UCB, -armToChoose)[-armToChoose:]
    
    for j in selections:
        reward = sample_arm(rewardProbilities, j)
        numSelections[j] = numSelections[j] + 1
        avgReward[j] = avgReward[j] + (reward)/numSelections[j]
        gainedReward = gainedReward + reward
        gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward

    return gainedRewardArray, numSelections, avgReward, gainedReward

def ucb_delayed_pure(gainedRewardArray, numSelections, avgReward, numOfArms, armToChoose, roundNum, gainedReward, rewardProbabilities, fifo, fifoSize, delays, delayTime):

    if((np.size(np.argwhere(numSelections < delayTime))) > 0):
        #selections  = np.random.choice(numOfArms, armToChoose, replace=False)
        selections  = (np.arange(armToChoose) + roundNum % numOfArms) % numOfArms -1
        numSelections[selections] = numSelections[selections] + 1
        for j in selections:    
            loc = np.argwhere(delays[j] == -1)
            loc = loc[0][0]
            delays[j, loc] = delayTime
    else:
        UCB = avgReward + np.sqrt(2*np.log(roundNum)/numSelections)
        selections = np.argpartition(UCB, -armToChoose)[-armToChoose:]
        numSelections[selections] = numSelections[selections] + 1
        
        for j in selections:
            tmpIdx = np.random.randint(0, fifoSize)
            reward = fifo[j, tmpIdx]
            loc = np.argwhere(delays[j] == -1)
            loc = loc[0][0]
            delays[j, loc] = delayTime
            
    if ((np.size(np.argwhere(delays == 0))) > 0):
        for m in np.argwhere(delays==0):
            reward = sample_arm(rewardProbabilities, m[0])
            delays[tuple(m)] = -1
            gainedReward = gainedReward + reward
            avgReward[j] = avgReward[j] + (reward)/numSelections[j]
            gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward
            

    return gainedRewardArray, numSelections, avgReward, gainedReward, delays, fifo

def ucb_delayed(gainedRewardArray, numSelections, avgReward, numOfArms, armToChoose, roundNum, gainedReward, rewardProbabilities, fifo, fifoSize, delays, delayTime):

    if((np.size(np.argwhere(numSelections < delayTime))) > 0):
        #selections  = np.random.choice(numOfArms, armToChoose, replace=False)
        selections  = (np.arange(armToChoose) + roundNum % numOfArms) % numOfArms -1
        numSelections[selections] = numSelections[selections] + 1
        for j in selections:    
            loc = np.argwhere(delays[j] == -1)
            loc = loc[0][0]
            delays[j, loc] = delayTime
    else:
        UCB = avgReward + np.sqrt(2*np.log(roundNum)/numSelections)
        selections = np.argpartition(UCB, -armToChoose)[-armToChoose:]
        numSelections[selections] = numSelections[selections] + 1
        
        for j in selections:
            tmpIdx = np.random.randint(0, fifoSize)
            reward = fifo[j, tmpIdx]
            loc = np.argwhere(delays[j] == -1)
            loc = loc[0][0]
            delays[j, loc] = delayTime
            avgReward[j] = avgReward[j] + (reward)/numSelections[j]
            
    if ((np.size(np.argwhere(delays == 0))) > 0):
        for m in np.argwhere(delays==0):
            
            tmp_queue = deque(fifo[m[0],:].ravel(), maxlen=fifoSize)
            reward = sample_arm(rewardProbabilities, m[0])
            tmp_queue.appendleft(reward)
            fifo[m[0],:] = np.array(tmp_queue)
            delays[tuple(m)] = -1
            gainedReward = gainedReward + reward
            gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward
            

    return gainedRewardArray, numSelections, avgReward, gainedReward, delays, fifo

def optimal(gainedRewardArray, armToChoose, rewardProbabilities, gainedReward, avgReward, numSelections, roundNum):
    selections  = np.arange(armToChoose)
    for j in selections:
        reward = sample_arm(rewardProbabilities, j)
        gainedReward = gainedReward + reward
        avgReward[j] = avgReward[j] + (reward)/numSelections[j]
        numSelections[j] = numSelections[j] + 1
        gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward
    return gainedRewardArray, gainedReward, avgReward, numSelections

def roundRobin(gainedRewardArray, armToChoose, numOfArms, rewardProbabilities, gainedReward, avgReward, numSelections, roundNum):
    selections  = (np.arange(armToChoose) + roundNum % numOfArms) % numOfArms
    for j in selections:
        reward = sample_arm(rewardProbabilities, j)
        gainedReward = gainedReward + reward
        avgReward[j] = avgReward[j] + (reward)/numSelections[j]
        numSelections[j] = numSelections[j] + 1
        gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward
    return gainedRewardArray, gainedReward, avgReward, numSelections

def roundRobin_delayed(gainedRewardArray, armToChoose, numOfArms, rewardProbabilities, gainedReward, avgReward, numSelections, roundNum, delays, delayTime):
    selections  = (np.arange(armToChoose) + roundNum % numOfArms) % numOfArms
    for j in selections:
        
        loc = np.argwhere(delays[j] == -1)
        loc = loc[0][0]
        delays[j, loc] = delayTime  
        numSelections[j] = numSelections[j] + 1
    
    
    if ((np.size(np.argwhere(delays == 0))) > 0):
        for m in np.argwhere(delays==0):
            reward = sample_arm(rewardProbabilities, m[0])
            delays[tuple(m)] = -1
            gainedReward = gainedReward + reward
            avgReward[j] = avgReward[j] + (reward)/numSelections[j]
            gainedRewardArray[roundNum] = gainedRewardArray[roundNum] + reward

    return gainedRewardArray, gainedReward, avgReward, numSelections, delays

class Algos(Enum):
    UCB = 0
    UCB_DELAYED_5 = 1
    UCB_DELAYED_10 = 2
    UCB_DELAYED_20 = 3
    UCB_DELAYED_PURE_5 = 4
    UCB_DELAYED_PURE_10 = 5
    UCB_DELAYED_PURE_20 = 6
    TS = 7
    TS_DELAYED_5 = 8
    TS_DELAYED_10 = 9
    TS_DELAYED_20 = 10
    TS_DELAYED_PURE_5 = 11
    TS_DELAYED_PURE_10 = 12
    TS_DELAYED_PURE_20 = 13
    OPTIMAL = 14
    ROUND_ROBIN = 15
    ROUND_ROBIN_DELAYED_5 = 16
    ROUND_ROBIN_DELAYED_10 = 17
    ROUND_ROBIN_DELAYED_20 = 18

    UCB_DELAYED_VARY = 19
    UCB_DELAYED_PURE_VARY = 20
    TS_DELAYED_VARY = 21
    TS_DELAYED_PURE_VARY = 22
    ROUND_ROBIN_DELAYED_VARY = 23


MAX_DELAY = 20

armNums = [500, 1000, 3000, 5000]

ucb_vary = np.zeros(len(armNums))
ucb_pure_vary = np.zeros(len(armNums))
ts_vary = np.zeros(len(armNums))
ts_vary = np.zeros(len(armNums))
ts_pure_vary = np.zeros(len(armNums))
round_robin_vary = np.zeros(len(armNums))
optimall = np.zeros(len(armNums))


for kk in range (0, len(armNums)):

    numOfArms = armNums[kk]
    armToChoose = 10
    numOfRounds = 10000

    algoNum = len(Algos)
    fifoSize = 10
    fifo = np.zeros((algoNum, numOfArms, fifoSize))
    
    rewardProbabilities = np.zeros(3)
    rewardProbabilities[0] = int(numOfArms/5)
    rewardProbabilities[1] = 0.9
    rewardProbabilities[2] = 0.3


    avgRewardArray = np.zeros((algoNum, numOfRounds))
    gainedRewardArray = np.zeros((algoNum, numOfRounds))
    numSelections = np.zeros((algoNum, numOfArms))
    avgReward = np.zeros((algoNum, numOfArms))
    delays = np.zeros((algoNum, numOfArms, MAX_DELAY+60)) -1

    gainedReward = np.zeros(algoNum)

    alpha = np.ones(numOfArms)
    alphad = np.ones(numOfArms)

    beta = np.ones(numOfArms)
    betad = np.ones(numOfArms)

    alphad2 = np.ones(numOfArms)
    betad2 = np.ones(numOfArms)

    alphad3 = np.ones(numOfArms)
    betad3= np.ones(numOfArms)

    alphad4 = np.ones(numOfArms)
    betad4= np.ones(numOfArms)

    alphad5 = np.ones(numOfArms)
    betad5 = np.ones(numOfArms)

    alphad6 = np.ones(numOfArms)
    betad6 = np.ones(numOfArms)

    alphad7 = np.ones(numOfArms)
    betad7 = np.ones(numOfArms)

    alphad8 = np.ones(numOfArms)
    betad8 = np.ones(numOfArms)

    delayMean = 10
    delay_vals = np.random.poisson(delayMean, numOfRounds)

    for i in range(1,numOfRounds):

        # Optimal
        gainedRewardArray[Algos.OPTIMAL.value,:], gainedReward[Algos.OPTIMAL.value], avgReward[Algos.OPTIMAL.value,:], numSelections[Algos.OPTIMAL.value,:] = optimal(gainedRewardArray[Algos.OPTIMAL.value,:], armToChoose, rewardProbabilities, gainedReward[Algos.OPTIMAL.value], avgReward[Algos.OPTIMAL.value,:], numSelections[Algos.OPTIMAL.value,:], i)
        avgRewardArray[Algos.OPTIMAL.value,i] = gainedReward[Algos.OPTIMAL.value] / i

        # UCB
        #gainedRewardArray[Algos.UCB.value,:], numSelections[Algos.UCB.value,:],  avgReward[Algos.UCB.value,:], gainedReward[Algos.UCB.value]= ucb(gainedRewardArray[Algos.UCB.value,:], numSelections[Algos.UCB.value, :], avgReward[Algos.UCB.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB.value], rewardProbabilities)
        #avgRewardArray[Algos.UCB.value,i] = gainedReward[Algos.UCB.value] / i
        
        # UCB Delayed PURE 5
        #gainedRewardArray[Algos.UCB_DELAYED_PURE_5.value,:], numSelections[Algos.UCB_DELAYED_PURE_5.value,:],  avgReward[Algos.UCB_DELAYED_PURE_5.value,:], gainedReward[Algos.UCB_DELAYED_PURE_5.value], delays[Algos.UCB_DELAYED_PURE_5.value,:], fifo[Algos.UCB_DELAYED_PURE_5.value, :, :] = ucb_delayed_pure(gainedRewardArray[Algos.UCB_DELAYED_PURE_5.value,:], numSelections[Algos.UCB_DELAYED_PURE_5.value, :], avgReward[Algos.UCB_DELAYED_PURE_5.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB_DELAYED_PURE_5.value], rewardProbabilities, fifo[Algos.UCB_DELAYED_PURE_5.value,:,:], fifoSize, delays[Algos.UCB_DELAYED_PURE_5.value,:], 5)
        #avgRewardArray[Algos.UCB_DELAYED_PURE_5.value,i] = gainedReward[Algos.UCB_DELAYED_PURE_5.value] / i
        # UCB Delayed PURE 10
        #gainedRewardArray[Algos.UCB_DELAYED_PURE_10.value,:], numSelections[Algos.UCB_DELAYED_PURE_10.value,:],  avgReward[Algos.UCB_DELAYED_PURE_10.value,:], gainedReward[Algos.UCB_DELAYED_PURE_10.value], delays[Algos.UCB_DELAYED_PURE_10.value,:], fifo[Algos.UCB_DELAYED_PURE_10.value, :, :] = ucb_delayed_pure(gainedRewardArray[Algos.UCB_DELAYED_PURE_10.value,:], numSelections[Algos.UCB_DELAYED_PURE_10.value, :], avgReward[Algos.UCB_DELAYED_PURE_10.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB_DELAYED_PURE_10.value], rewardProbabilities, fifo[Algos.UCB_DELAYED_PURE_10.value,:,:], fifoSize, delays[Algos.UCB_DELAYED_PURE_10.value,:], 10)
        #avgRewardArray[Algos.UCB_DELAYED_PURE_10.value,i] = gainedReward[Algos.UCB_DELAYED_PURE_10.value] / i
        # UCB Delayed PURE 20
        #gainedRewardArray[Algos.UCB_DELAYED_PURE_20.value,:], numSelections[Algos.UCB_DELAYED_PURE_20.value,:],  avgReward[Algos.UCB_DELAYED_PURE_20.value,:], gainedReward[Algos.UCB_DELAYED_PURE_20.value], delays[Algos.UCB_DELAYED_PURE_20.value,:], fifo[Algos.UCB_DELAYED_PURE_20.value, :, :] = ucb_delayed_pure(gainedRewardArray[Algos.UCB_DELAYED_PURE_20.value,:], numSelections[Algos.UCB_DELAYED_PURE_20.value, :], avgReward[Algos.UCB_DELAYED_PURE_20.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB_DELAYED_PURE_20.value], rewardProbabilities, fifo[Algos.UCB_DELAYED_PURE_20.value,:,:], fifoSize, delays[Algos.UCB_DELAYED_PURE_20.value,:], 5)
        #avgRewardArray[Algos.UCB_DELAYED_PURE_20.value,i] = gainedReward[Algos.UCB_DELAYED_PURE_20.value] / i
        
        # UCB Delayed 5
        #gainedRewardArray[Algos.UCB_DELAYED_5.value,:], numSelections[Algos.UCB_DELAYED_5.value,:],  avgReward[Algos.UCB_DELAYED_5.value,:], gainedReward[Algos.UCB_DELAYED_5.value], delays[Algos.UCB_DELAYED_5.value,:], fifo[Algos.UCB_DELAYED_5.value, :, :] = ucb_delayed(gainedRewardArray[Algos.UCB_DELAYED_5.value,:], numSelections[Algos.UCB_DELAYED_5.value, :], avgReward[Algos.UCB_DELAYED_5.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB_DELAYED_5.value], rewardProbabilities, fifo[Algos.UCB_DELAYED_5.value,:,:], fifoSize, delays[Algos.UCB_DELAYED_5.value,:], 5)
        #avgRewardArray[Algos.UCB_DELAYED_5.value,i] = gainedReward[Algos.UCB_DELAYED_5.value] / i
        # UCB Delayed 10
        #gainedRewardArray[Algos.UCB_DELAYED_10.value,:], numSelections[Algos.UCB_DELAYED_10.value,:],  avgReward[Algos.UCB_DELAYED_10.value,:], gainedReward[Algos.UCB_DELAYED_10.value], delays[Algos.UCB_DELAYED_10.value,:], fifo[Algos.UCB_DELAYED_10.value, :, :] = ucb_delayed(gainedRewardArray[Algos.UCB_DELAYED_10.value,:], numSelections[Algos.UCB_DELAYED_10.value, :], avgReward[Algos.UCB_DELAYED_10.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB_DELAYED_10.value], rewardProbabilities, fifo[Algos.UCB_DELAYED_10.value,:,:], fifoSize, delays[Algos.UCB_DELAYED_10.value,:], 10)
        #avgRewardArray[Algos.UCB_DELAYED_10.value,i] = gainedReward[Algos.UCB_DELAYED_10.value] / i
        # UCB Delayed 20
        #gainedRewardArray[Algos.UCB_DELAYED_20.value,:], numSelections[Algos.UCB_DELAYED_20.value,:],  avgReward[Algos.UCB_DELAYED_20.value,:], gainedReward[Algos.UCB_DELAYED_20.value], delays[Algos.UCB_DELAYED_20.value,:], fifo[Algos.UCB_DELAYED_20.value, :, :] = ucb_delayed(gainedRewardArray[Algos.UCB_DELAYED_20.value,:], numSelections[Algos.UCB_DELAYED_20.value, :], avgReward[Algos.UCB_DELAYED_20.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB_DELAYED_20.value], rewardProbabilities, fifo[Algos.UCB_DELAYED_20.value,:,:], fifoSize, delays[Algos.UCB_DELAYED_20.value,:], 20)
        #avgRewardArray[Algos.UCB_DELAYED_20.value,i] = gainedReward[Algos.UCB_DELAYED_20.value] / i
        
        # TS
        #gainedRewardArray[Algos.TS.value,:], alpha, beta, avgReward[Algos.TS.value,:], numSelections[Algos.TS.value,:], gainedReward[Algos.TS.value] = ts(gainedRewardArray[Algos.TS.value,:], alpha, beta, armToChoose, rewardProbabilities, avgReward[Algos.TS.value,:], numSelections[Algos.TS.value,:], gainedReward[Algos.TS.value], i)
        #avgRewardArray[Algos.TS.value,i] = gainedReward[Algos.TS.value] / i

        # TS Delayed 5
        #gainedRewardArray[Algos.TS_DELAYED_5.value,:], alphad, betad, numSelections[Algos.TS_DELAYED_5.value,:], avgReward[Algos.TS_DELAYED_5.value,:], gainedReward[Algos.TS_DELAYED_5.value], fifo[Algos.TS_DELAYED_5.value, :, :], delays[Algos.TS_DELAYED_5.value,:] = ts_delayed(gainedRewardArray[Algos.TS_DELAYED_5.value,:], alphad, betad, numSelections[Algos.TS_DELAYED_5.value,:], avgReward[Algos.TS_DELAYED_5.value,:], numOfArms, armToChoose, i, gainedReward[Algos.TS_DELAYED_5.value], rewardProbabilities, fifo[Algos.TS_DELAYED_5.value,:,:], fifoSize, delays[Algos.TS_DELAYED_5.value,:], 5)
        #avgRewardArray[Algos.TS_DELAYED_5.value,i] = gainedReward[Algos.TS_DELAYED_5.value] / i
        # TS Delayed 10
        #gainedRewardArray[Algos.TS_DELAYED_10.value,:], alphad2, betad2, numSelections[Algos.TS_DELAYED_10.value,:], avgReward[Algos.TS_DELAYED_10.value,:], gainedReward[Algos.TS_DELAYED_10.value], fifo[Algos.TS_DELAYED_10.value, :, :], delays[Algos.TS_DELAYED_10.value,:] = ts_delayed(gainedRewardArray[Algos.TS_DELAYED_10.value,:], alphad2, betad2, numSelections[Algos.TS_DELAYED_10.value,:], avgReward[Algos.TS_DELAYED_10.value,:], numOfArms, armToChoose, i, gainedReward[Algos.TS_DELAYED_10.value], rewardProbabilities, fifo[Algos.TS_DELAYED_10.value,:,:], fifoSize, delays[Algos.TS_DELAYED_10.value,:], 10)
        #avgRewardArray[Algos.TS_DELAYED_10.value,i] = gainedReward[Algos.TS_DELAYED_10.value] / i
        # TS Delayed 20
        #gainedRewardArray[Algos.TS_DELAYED_20.value,:], alphad3, betad3, numSelections[Algos.TS_DELAYED_20.value,:], avgReward[Algos.TS_DELAYED_20.value,:], gainedReward[Algos.TS_DELAYED_20.value], fifo[Algos.TS_DELAYED_20.value, :, :], delays[Algos.TS_DELAYED_20.value,:] = ts_delayed(gainedRewardArray[Algos.TS_DELAYED_20.value,:], alphad3, betad3, numSelections[Algos.TS_DELAYED_20.value,:], avgReward[Algos.TS_DELAYED_20.value,:], numOfArms, armToChoose, i, gainedReward[Algos.TS_DELAYED_20.value], rewardProbabilities, fifo[Algos.TS_DELAYED_20.value,:,:], fifoSize, delays[Algos.TS_DELAYED_20.value,:], 20)
        #avgRewardArray[Algos.TS_DELAYED_20.value,i] = gainedReward[Algos.TS_DELAYED_10.value] / i
        
        # TS Delayed Pure  5
        #gainedRewardArray[Algos.TS_DELAYED_PURE_5.value,:], alphad4, betad4, numSelections[Algos.TS_DELAYED_PURE_5.value,:], avgReward[Algos.TS_DELAYED_PURE_5.value,:], gainedReward[Algos.TS_DELAYED_PURE_5.value], fifo[Algos.TS_DELAYED_PURE_5.value, :, :], delays[Algos.TS_DELAYED_PURE_5.value,:] = ts_delayed_pure(gainedRewardArray[Algos.TS_DELAYED_PURE_5.value,:], alphad4, betad4, numSelections[Algos.TS_DELAYED_PURE_5.value,:], avgReward[Algos.TS_DELAYED_PURE_5.value,:], numOfArms, armToChoose, i, gainedReward[Algos.TS_DELAYED_PURE_5.value], rewardProbabilities, fifo[Algos.TS_DELAYED_PURE_5.value,:,:], fifoSize, delays[Algos.TS_DELAYED_PURE_5.value,:], 5)
        #avgRewardArray[Algos.TS_DELAYED_PURE_5.value,i] = gainedReward[Algos.TS_DELAYED_PURE_5.value] / i
        # TS Delayed Pure  10
        #gainedRewardArray[Algos.TS_DELAYED_PURE_10.value,:], alphad5, betad5, numSelections[Algos.TS_DELAYED_PURE_10.value,:], avgReward[Algos.TS_DELAYED_PURE_10.value,:], gainedReward[Algos.TS_DELAYED_PURE_10.value], fifo[Algos.TS_DELAYED_PURE_10.value, :, :], delays[Algos.TS_DELAYED_PURE_10.value,:] = ts_delayed_pure(gainedRewardArray[Algos.TS_DELAYED_PURE_10.value,:], alphad5, betad5, numSelections[Algos.TS_DELAYED_PURE_10.value,:], avgReward[Algos.TS_DELAYED_PURE_10.value,:], numOfArms, armToChoose, i, gainedReward[Algos.TS_DELAYED_PURE_10.value], rewardProbabilities, fifo[Algos.TS_DELAYED_PURE_10.value,:,:], fifoSize, delays[Algos.TS_DELAYED_PURE_10.value,:], 10)
        #avgRewardArray[Algos.TS_DELAYED_PURE_10.value,i] = gainedReward[Algos.TS_DELAYED_PURE_10.value] / i
        # TS Delayed Pure  20
        #gainedRewardArray[Algos.TS_DELAYED_PURE_20.value,:], alphad6, betad6, numSelections[Algos.TS_DELAYED_PURE_20.value,:], avgReward[Algos.TS_DELAYED_PURE_20.value,:], gainedReward[Algos.TS_DELAYED_PURE_20.value], fifo[Algos.TS_DELAYED_PURE_20.value, :, :], delays[Algos.TS_DELAYED_PURE_20.value,:] = ts_delayed_pure(gainedRewardArray[Algos.TS_DELAYED_PURE_20.value,:], alphad6, betad6, numSelections[Algos.TS_DELAYED_PURE_20.value,:], avgReward[Algos.TS_DELAYED_PURE_20.value,:], numOfArms, armToChoose, i, gainedReward[Algos.TS_DELAYED_PURE_20.value], rewardProbabilities, fifo[Algos.TS_DELAYED_PURE_20.value,:,:], fifoSize, delays[Algos.TS_DELAYED_PURE_20.value,:], 20)
        #avgRewardArray[Algos.TS_DELAYED_PURE_20.value,i] = gainedReward[Algos.TS_DELAYED_PURE_20.value] / i
        
        # Round Robin
        #gainedRewardArray[Algos.ROUND_ROBIN.value,:], gainedReward[Algos.ROUND_ROBIN.value], avgReward[Algos.ROUND_ROBIN.value,:], numSelections[Algos.ROUND_ROBIN.value,:] = roundRobin(gainedRewardArray[Algos.ROUND_ROBIN.value,:], armToChoose, numOfArms, rewardProbabilities, gainedReward[Algos.ROUND_ROBIN.value], avgReward[Algos.ROUND_ROBIN.value,:], numSelections[Algos.ROUND_ROBIN.value,:], i)
        #avgRewardArray[Algos.ROUND_ROBIN.value,i] = gainedReward[Algos.ROUND_ROBIN.value] / i
        
        # Round Robin Delayed 5
        #gainedRewardArray[Algos.ROUND_ROBIN_DELAYED_5.value,:], gainedReward[Algos.ROUND_ROBIN_DELAYED_5.value], avgReward[Algos.ROUND_ROBIN_DELAYED_5.value,:], numSelections[Algos.ROUND_ROBIN_DELAYED_5.value,:], delays[Algos.ROUND_ROBIN_DELAYED_5.value,:] = roundRobin_delayed(gainedRewardArray[Algos.ROUND_ROBIN_DELAYED_5.value,:], armToChoose, numOfArms, rewardProbabilities, gainedReward[Algos.ROUND_ROBIN_DELAYED_5.value], avgReward[Algos.ROUND_ROBIN_DELAYED_5.value,:], numSelections[Algos.ROUND_ROBIN_DELAYED_5.value,:], i, delays[Algos.ROUND_ROBIN_DELAYED_5.value,:], 10)
        #avgRewardArray[Algos.ROUND_ROBIN_DELAYED_5.value,i] = gainedReward[Algos.ROUND_ROBIN_DELAYED_5.value] / i
        # Round Robin Delayed 10
        #gainedRewardArray[Algos.ROUND_ROBIN_DELAYED_10.value,:], gainedReward[Algos.ROUND_ROBIN_DELAYED_10.value], avgReward[Algos.ROUND_ROBIN_DELAYED_10.value,:], numSelections[Algos.ROUND_ROBIN_DELAYED_10.value,:], delays[Algos.ROUND_ROBIN_DELAYED_10.value,:] = roundRobin_delayed(gainedRewardArray[Algos.ROUND_ROBIN_DELAYED_10.value,:], armToChoose, numOfArms, rewardProbabilities, gainedReward[Algos.ROUND_ROBIN_DELAYED_10.value], avgReward[Algos.ROUND_ROBIN_DELAYED_10.value,:], numSelections[Algos.ROUND_ROBIN_DELAYED_10.value,:], i, delays[Algos.ROUND_ROBIN_DELAYED_10.value,:], 10)
        #avgRewardArray[Algos.ROUND_ROBIN_DELAYED_10.value,i] = gainedReward[Algos.ROUND_ROBIN_DELAYED_10.value] / i
        # Round Robin Delayed 20
        #gainedRewardArray[Algos.ROUND_ROBIN_DELAYED_20.value,:], gainedReward[Algos.ROUND_ROBIN_DELAYED_20.value], avgReward[Algos.ROUND_ROBIN_DELAYED_20.value,:], numSelections[Algos.ROUND_ROBIN_DELAYED_20.value,:], delays[Algos.ROUND_ROBIN_DELAYED_20.value,:] = roundRobin_delayed(gainedRewardArray[Algos.ROUND_ROBIN_DELAYED_20.value,:], armToChoose, numOfArms, rewardProbabilities, gainedReward[Algos.ROUND_ROBIN_DELAYED_20.value], avgReward[Algos.ROUND_ROBIN_DELAYED_20.value,:], numSelections[Algos.ROUND_ROBIN_DELAYED_20.value,:], i, delays[Algos.ROUND_ROBIN_DELAYED_20.value,:], 20)
        #avgRewardArray[Algos.ROUND_ROBIN_DELAYED_20.value,i] = gainedReward[Algos.ROUND_ROBIN_DELAYED_20.value] / i

        # UCB Delayed Vary
        gainedRewardArray[Algos.UCB_DELAYED_VARY.value,:], numSelections[Algos.UCB_DELAYED_VARY.value,:],  avgReward[Algos.UCB_DELAYED_VARY.value,:], gainedReward[Algos.UCB_DELAYED_VARY.value], delays[Algos.UCB_DELAYED_VARY.value,:], fifo[Algos.UCB_DELAYED_VARY.value, :, :] = ucb_delayed(gainedRewardArray[Algos.UCB_DELAYED_VARY.value,:], numSelections[Algos.UCB_DELAYED_VARY.value, :], avgReward[Algos.UCB_DELAYED_VARY.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB_DELAYED_VARY.value], rewardProbabilities, fifo[Algos.UCB_DELAYED_VARY.value,:,:], fifoSize, delays[Algos.UCB_DELAYED_VARY.value,:], delay_vals[i])
        avgRewardArray[Algos.UCB_DELAYED_VARY.value,i] = gainedReward[Algos.UCB_DELAYED_VARY.value] / i
        # UCB Delayed Pure Vary
        gainedRewardArray[Algos.UCB_DELAYED_PURE_VARY.value,:], numSelections[Algos.UCB_DELAYED_PURE_VARY.value,:],  avgReward[Algos.UCB_DELAYED_PURE_VARY.value,:], gainedReward[Algos.UCB_DELAYED_PURE_VARY.value], delays[Algos.UCB_DELAYED_PURE_VARY.value,:], fifo[Algos.UCB_DELAYED_PURE_VARY.value, :, :] = ucb_delayed_pure(gainedRewardArray[Algos.UCB_DELAYED_PURE_VARY.value,:], numSelections[Algos.UCB_DELAYED_PURE_VARY.value, :], avgReward[Algos.UCB_DELAYED_PURE_VARY.value, :], numOfArms, armToChoose, i, gainedReward[Algos.UCB_DELAYED_PURE_VARY.value], rewardProbabilities, fifo[Algos.UCB_DELAYED_PURE_VARY.value,:,:], fifoSize, delays[Algos.UCB_DELAYED_PURE_VARY.value,:], delay_vals[i])
        avgRewardArray[Algos.UCB_DELAYED_PURE_VARY.value,i] = gainedReward[Algos.UCB_DELAYED_PURE_VARY.value] / i
        # TS Delayed Vary
        gainedRewardArray[Algos.TS_DELAYED_VARY.value,:], alphad7, betad7, numSelections[Algos.TS_DELAYED_VARY.value,:], avgReward[Algos.TS_DELAYED_VARY.value,:], gainedReward[Algos.TS_DELAYED_VARY.value], fifo[Algos.TS_DELAYED_VARY.value, :, :], delays[Algos.TS_DELAYED_VARY.value,:] = ts_delayed(gainedRewardArray[Algos.TS_DELAYED_VARY.value,:], alphad7, betad7, numSelections[Algos.TS_DELAYED_VARY.value,:], avgReward[Algos.TS_DELAYED_VARY.value,:], numOfArms, armToChoose, i, gainedReward[Algos.TS_DELAYED_VARY.value], rewardProbabilities, fifo[Algos.TS_DELAYED_VARY.value,:,:], fifoSize, delays[Algos.TS_DELAYED_VARY.value,:], delay_vals[i])
        avgRewardArray[Algos.TS_DELAYED_VARY.value,i] = gainedReward[Algos.TS_DELAYED_VARY.value] / i
        # TS Delayed Pure Vary
        gainedRewardArray[Algos.TS_DELAYED_PURE_VARY.value,:], alphad8, betad8, numSelections[Algos.TS_DELAYED_PURE_VARY.value,:], avgReward[Algos.TS_DELAYED_PURE_VARY.value,:], gainedReward[Algos.TS_DELAYED_PURE_VARY.value], fifo[Algos.TS_DELAYED_PURE_VARY.value, :, :], delays[Algos.TS_DELAYED_PURE_VARY.value,:] = ts_delayed_pure(gainedRewardArray[Algos.TS_DELAYED_PURE_VARY.value,:], alphad8, betad8, numSelections[Algos.TS_DELAYED_PURE_VARY.value,:], avgReward[Algos.TS_DELAYED_PURE_VARY.value,:], numOfArms, armToChoose, i, gainedReward[Algos.TS_DELAYED_PURE_VARY.value], rewardProbabilities, fifo[Algos.TS_DELAYED_PURE_VARY.value,:,:], fifoSize, delays[Algos.TS_DELAYED_PURE_VARY.value,:], delay_vals[i])
        avgRewardArray[Algos.TS_DELAYED_PURE_VARY.value,i] = gainedReward[Algos.TS_DELAYED_PURE_VARY.value] / i
        # Round Robin Delayed Vary
        gainedRewardArray[Algos.ROUND_ROBIN_DELAYED_VARY.value,:], gainedReward[Algos.ROUND_ROBIN_DELAYED_VARY.value], avgReward[Algos.ROUND_ROBIN_DELAYED_VARY.value,:], numSelections[Algos.ROUND_ROBIN_DELAYED_VARY.value,:], delays[Algos.ROUND_ROBIN_DELAYED_VARY.value,:] = roundRobin_delayed(gainedRewardArray[Algos.ROUND_ROBIN_DELAYED_VARY.value,:], armToChoose, numOfArms, rewardProbabilities, gainedReward[Algos.ROUND_ROBIN_DELAYED_VARY.value], avgReward[Algos.ROUND_ROBIN_DELAYED_VARY.value,:], numSelections[Algos.ROUND_ROBIN_DELAYED_VARY.value,:], i, delays[Algos.ROUND_ROBIN_DELAYED_VARY.value,:], delay_vals[i])
        avgRewardArray[Algos.ROUND_ROBIN_DELAYED_VARY.value,i] = gainedReward[Algos.ROUND_ROBIN_DELAYED_VARY.value] / i


        idxs = np.argwhere(delays > 0)
        for idx in idxs:
            delays[tuple(idx)] = delays[tuple(idx)] - 1



    # mean reward per round vs each case with delay

    #labels = ["PURE UCB", "REWARD QUEUE UCB", "PURE TS", "REWARD QUEUE TS", "OPTIMAL", "ROUND ROBIN"]

    #pure_ucb = np.array([np.mean(gainedRewardArray[0,:]), np.mean(gainedRewardArray[4,:]), np.mean(gainedRewardArray[5,:]), np.mean(gainedRewardArray[6,:])])
    #reward_queue_ucb = np.array([np.mean(gainedRewardArray[1,:]), np.mean(gainedRewardArray[2,:]), np.mean(gainedRewardArray[3,:])])
    #pure_ts = np.array([np.mean(gainedRewardArray[7,:]), np.mean(gainedRewardArray[11,:]), np.mean(gainedRewardArray[12,:]), np.mean(gainedRewardArray[13,:])])
    #reward_queue_ts = np.array([np.mean(gainedRewardArray[8,:]), np.mean(gainedRewardArray[9,:]), np.mean(gainedRewardArray[10,:])])
    #optimal_case = np.array([np.mean(gainedRewardArray[14,:])])
    #round_robin = np.array([np.mean(gainedRewardArray[15,:]), np.mean(gainedRewardArray[16,:]), np.mean(gainedRewardArray[17,:]), np.mean(gainedRewardArray[18,:])])

    ucb_vary[kk] = np.mean(gainedRewardArray[19,:])
    ucb_pure_vary[kk] = np.mean(gainedRewardArray[20,:])
    ts_vary[kk] = np.mean(gainedRewardArray[21,:])
    ts_pure_vary[kk] = np.mean(gainedRewardArray[22,:])
    round_robin_vary[kk] = np.mean(gainedRewardArray[23,:])
    optimall[kk] = np.mean(gainedRewardArray[14,:])


#pure_ucb_std = np.array([np.std(gainedRewardArray[0,:]), np.std(gainedRewardArray[4,:]), np.std(gainedRewardArray[5,:]), np.std(gainedRewardArray[6,:])])
#reward_queue_ucb_std = np.array([np.std(gainedRewardArray[1,:]), np.std(gainedRewardArray[2,:]), np.std(gainedRewardArray[3,:])])
#pure_ts_std = np.array([np.std(gainedRewardArray[7,:]), np.std(gainedRewardArray[11,:]), np.std(gainedRewardArray[12,:]), np.std(gainedRewardArray[13,:])])
#reward_queue_ts_std = np.array([np.std(gainedRewardArray[8,:]), np.std(gainedRewardArray[9,:]), np.std(gainedRewardArray[10,:])])
#optimal_case_std = np.array([np.std(gainedRewardArray[14,:])])
#round_robin_std = np.array([np.std(gainedRewardArray[15,:]), np.std(gainedRewardArray[16,:]), np.std(gainedRewardArray[17,:]), np.std(gainedRewardArray[18,:])])

#x = np.array([0, 5, 10 ,20])


ax = plt.subplot()
plt.plot(armNums, ucb_vary, marker='^')
plt.plot(armNums, ucb_pure_vary, marker='*')
plt.plot(armNums, ts_vary, marker='+')
plt.plot(armNums, ts_pure_vary, marker='v')
plt.plot(armNums, optimall, marker='d')
plt.plot(armNums, round_robin_vary, marker='o')

plt.legend(["UCB with Reward Queue", "Pure UCB", "TS with reward queue", "Pure TS", "Optimal", "RoundRobin"], bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(y = np.mean(gainedRewardArray[Algos.OPTIMAL.value,:]), color = 'r', linestyle = '--')
plt.grid()
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.xlabel("Number of arms")
plt.ylabel("Mean reward per round")
plt.title("Mean reward per round vs number of arms [N = 20,50,100,200,500 , K = 10]")
plt.tight_layout()
plt.show()

"""
# line graph
ax = plt.subplot()
plt.plot(x, pure_ucb, marker='^')
plt.plot(x[1:], reward_queue_ucb, marker='*')
plt.plot(x, pure_ts, marker='+')
plt.plot(x[1:], reward_queue_ts, marker='v')
plt.plot(x[0], optimal_case, marker='d')
plt.plot(x, round_robin, marker='o')

plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(y = np.mean(gainedRewardArray[Algos.OPTIMAL.value,:]), color = 'r', linestyle = '--')
plt.grid()
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.xlabel("Delay")
plt.ylabel("Mean reward per round with std")
plt.title("Mean reward per round with std vs delay [N = 100, K = 3]")
plt.tight_layout()
plt.show()
"""


"""
# error bar with std
ax = plt.subplot()
plt.errorbar(x, pure_ucb, pure_ucb_std, marker='^', capsize=3)
plt.errorbar(x[1:], reward_queue_ucb, reward_queue_ucb_std, marker='*', capsize=3)
plt.errorbar(x, pure_ts, pure_ts_std, marker='+', capsize=3)
plt.errorbar(x[1:], reward_queue_ts, reward_queue_ts_std, marker='v', capsize=3)
plt.errorbar(x[0], optimal_case, optimal_case_std, marker='d', capsize=3)
plt.errorbar(x, round_robin, round_robin_std, marker='o', capsize=3)

plt.legend(labels, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(y = np.mean(gainedRewardArray[Algos.OPTIMAL.value,:]), color = 'r', linestyle = '--')
plt.grid()
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.xlabel("Delay")
plt.ylabel("Mean reward per round with std")
plt.title("Mean reward per round with std vs delay [N = 100, K = 3]")
plt.tight_layout()
plt.show()
"""

"""
# mean reward per round vs delay
labels = ["PURE UCB without delay", "UCB with reward queue delay = 5", "UCB with reward queue delay = 10", "UCB with reward queue delay = 20", "UCB with reward queue delay = 50", "PURE TS without delay", "TS with reward queue d = 5", "TS with reward queue d = 10", "Optimal", "Round Robin without delay", "Delayed Round Robin, d = 5", "PURE Delayed UCB delay = 5"]
legends = [labels[0], labels[5], labels[8], labels[9], labels[1], labels[6], labels[10], labels[2], labels[7], labels[11], labels[3], labels[4]]
xax = np.array([0, 5, 10, 20, 50])
ax = plt.subplot()
plt.errorbar(xax[0], np.mean(gainedRewardArray[0,:]), np.std(gainedRewardArray[0,:]), linestyle='None', marker='s', capsize=3)
plt.errorbar(xax[0], np.mean(gainedRewardArray[5,:]), np.std(gainedRewardArray[5,:]), linestyle='None', marker='+', capsize=3)
plt.errorbar(xax[0], np.mean(gainedRewardArray[8,:]), np.std(gainedRewardArray[8,:]), linestyle='None', marker='*', capsize=3)
plt.errorbar(xax[0], np.mean(gainedRewardArray[9,:]), np.std(gainedRewardArray[9,:]), linestyle='None', marker='^', capsize=3)
plt.errorbar(xax[1], np.mean(gainedRewardArray[1,:]), np.std(gainedRewardArray[1,:]), linestyle='None', marker='s', capsize=3)
plt.errorbar(xax[1], np.mean(gainedRewardArray[6,:]), np.std(gainedRewardArray[6,:]), linestyle='None', marker='+', capsize=3)
plt.errorbar(xax[1], np.mean(gainedRewardArray[10,:]), np.std(gainedRewardArray[10,:]), linestyle='None', marker='*', capsize=3)
plt.errorbar(xax[1], np.mean(gainedRewardArray[11,:]), np.std(gainedRewardArray[11,:]), linestyle='None', marker='^', capsize=3)
plt.errorbar(xax[2], np.mean(gainedRewardArray[2,:]), np.std(gainedRewardArray[2,:]), linestyle='None', marker='*', capsize=3)
plt.errorbar(xax[2], np.mean(gainedRewardArray[7,:]), np.std(gainedRewardArray[7,:]), linestyle='None', marker='+', capsize=3)
plt.errorbar(xax[3], np.mean(gainedRewardArray[3,:]), np.std(gainedRewardArray[3,:]), linestyle='None', marker='*', capsize=3)
plt.errorbar(xax[4], np.mean(gainedRewardArray[4,:]), np.std(gainedRewardArray[4,:]), linestyle='None', marker='*', capsize=3)

plt.legend(legends, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.axhline(y = np.mean(gainedRewardArray[Algos.OPTIMAL.value,:]), color = 'r', linestyle = '--')
plt.grid()
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.xlabel("Delay")
plt.ylabel("Mean reward per round with std")
plt.title("Mean reward per round with std vs delay [N = 100, K = 3]")
plt.tight_layout()
plt.show()
"""

"""
# mean reward per round vs cases
labels = ["PURE UCB without delay", "UCB with reward queue delay = 5", "UCB with reward queue delay = 10", "UCB with reward queue delay = 20", "UCB with reward queue delay = 50", "PURE TS without delay", "TS with reward queue d = 5", "TS with reward queue d = 10", "Optimal", "Round Robin without delay", "Delayed Round Robin, d = 5", "PURE Delayed UCB delay = 5"]
ax = plt.subplot()
plt.errorbar(labels, np.mean(gainedRewardArray, axis=1), np.std(gainedRewardArray, axis=1), linestyle='None', marker='^', capsize=3)
plt.axhline(y = np.mean(gainedRewardArray[Algos.OPTIMAL.value,:]), color = 'r', linestyle = '--')
plt.grid()
plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
plt.xlabel("Cases")
plt.ylabel("Mean reward per round with std")
plt.title("Mean reward per round with std [N = 100, K = 3]")
plt.tight_layout()
plt.show()
"""


""" 
# straight line graph mean reward vs rounds
plt.plot(avgRewardArray[Algos.UCB.value,:], label="PURE UCB without delay")
plt.plot(avgRewardArray[Algos.UCB_DELAYED_5.value,:], label="UCB with reward queue delay = 5")
plt.plot(avgRewardArray[Algos.UCB_DELAYED_10.value,:], label="UCB with reward queue delay = 10")
plt.plot(avgRewardArray[Algos.UCB_DELAYED_20.value,:], label="UCB with reward queue delay = 20")
plt.plot(avgRewardArray[Algos.UCB_DELAYED_50.value,:], label="UCB with reward queue delay = 50")
plt.plot(avgRewardArray[Algos.UCB_DELAYED_PURE_5.value,:], label="PURE Delayed UCB delay = 5")
plt.plot(avgRewardArray[Algos.TS.value,:], label="PURE TS without delay")
plt.plot(avgRewardArray[Algos.TS_DELAYED_5.value,:], label="TS with reward queue d = 5")
plt.plot(avgRewardArray[Algos.TS_DELAYED_10.value,:], label="TS with reward queue d = 10")
plt.plot(avgRewardArray[Algos.OPTIMAL.value,:], label="Optimal")
plt.plot(avgRewardArray[Algos.ROUND_ROBIN.value,:], label="Round Robin without delay")
plt.plot(avgRewardArray[Algos.ROUND_ROBIN_DELAYED_5.value,:], label="Delayed Round Robin, d = 5")

plt.title("Average Reward vs Rounds [N = 100, K = 3]")
plt.xlabel("Rounds")
plt.ylabel("Average Reward")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""


