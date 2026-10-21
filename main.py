#!/bin/python3
import numpy as np
import plotly.graph_objects as go

def batched_arnoldi(workMatrix,projectiveVectors,fractionalConvergence=10**-6,requiredConvNum=None,maxIter=40):
    propO1=projectiveVectors
    nVectors = projectiveVectors.shape[1]
    egvlPrev=np.zeros(nVectors)
    egvcPrev=np.zeros([nVectors]*2)
    if requiredConvNum is None:
        requiredConvNum = nVectors*8//10
    converged=False
    for i in np.arange(maxIter):
        propOrth,_ = np.linalg.qr(propO1)#tfp.math.gram_schmidt(propO1)
        #np.matmul(np.conj(propOrth.numpy().T),propOrth)
        propO1= np.matmul(workMatrix,propOrth)
        c1=np.einsum('ji,jk->ki',propO1,np.conj(propOrth))
            
        egvl,egvc=np.linalg.eig(c1)
        rank2=np.argsort(np.abs(egvl))[::-1]
        egvl=egvl[rank2]
        egvc=egvc[:,rank2]
        diff = np.max(np.abs(egvl-egvlPrev)[:requiredConvNum])
        if diff<fractionalConvergence:
            print(f'converged in {i} iterations')
            converged=True
            break
        egvlPrev=egvl
        egvcPrev=egvc
    egvcProj=np.matmul(propOrth+propO1,egvc)
    #egvcProj=np.matmul(workMatrix,egvcProj)
    egvcProj = egvcProj/np.sqrt(np.einsum('ij,ij->j',egvcProj,np.conj(egvcProj)) )
    return egvl,egvcProj,converged


def generate_new_vectors(mSize,nTestVec):
    return np.random.rand(mSize,nTestVec)+np.random.rand(mSize,nTestVec)*1.j

def resolve_largest_egvl(workMatrix,requiredConvNum=20, testVecNum=30):
    currentVectors = generate_new_vectors(mSize,testVecNum)            
    offsets=np.zeros_like(workMatrix)
    numResid=1
    
    while numResid>0:
        egvl,egvc,cnv=batched_arnoldi(workMatrix,currentVectors,requiredConvNum=requiredConvNum,fractionalConvergence=5*10**-7)
        if not cnv:
            currentVectors=egvc
            print('slow convergence')
        else:
            numResid=np.sum(np.abs(egvl)>0.1)
            
            rank=np.argsort(np.abs(egvl))
            egvcSubset= egvc[:,:requiredConvNum]
            egvlSubset= egvl[:requiredConvNum]
            t=np.einsum('ji,jk',egvcSubset[:,:requiredConvNum],egvcSubset[:,:requiredConvNum])
            tInv=np.linalg.inv(t)
            offset=np.matmul(egvcSubset,egvlSubset[:,np.newaxis]*np.matmul(tInv,egvcSubset.T))
            
            
            break
    return offset, egvlSubset
    
mSize = 300
def generate_test_matrix(mSize,plot=True):
    eigenValues=np.exp(np.random.rand(mSize)*2*np.pi*1.j)*np.exp(4*np.random.rand(mSize)**2-2)
    eigenValues = eigenValues[np.argsort(np.abs(eigenValues))]
    eigenVectorsR = np.random.rand(mSize,mSize)*np.exp(np.random.rand(mSize,mSize)*2*np.pi*1.j)
    eigenVectorsR /= np.sum(eigenVectorsR*np.conj(eigenVectorsR),axis=1)[:,np.newaxis]
    eigenVectorsL = np.linalg.inv(eigenVectorsR)
    testInput=np.matmul(eigenVectorsR,eigenValues[:,np.newaxis]*eigenVectorsL)
    
    #test that random eigenvectors are not parallel
    assert np.max(np.abs(np.matmul(eigenVectorsL,eigenVectorsR)-np.eye(mSize)))<10**11
    #test that right eigenvectors behave as intended
    assert np.max(np.abs(np.matmul(testInput,eigenVectorsR)/eigenValues[np.newaxis]-eigenVectorsR))<10**-7
    #test that lapack agrees with our construction
    egvc,_ = np.linalg.eig(testInput)
    egvcS=egvc[np.argsort(np.abs(egvc))]
    assert np.max(np.abs(egvcS-eigenValues))<10**-12
    
    if plot:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=eigenValues.real,y=eigenValues.imag,mode='markers'))
        fig.show()
    return testInput,eigenValues





mSize=300
workMatrix,eigenValues=generate_test_matrix(mSize,False)
numRemoved=0


egvls=[]
convNumber=15
numRemoved=0
egvlRemoved=[]
for i in np.arange(5):
    offset,egvl=resolve_largest_egvl(workMatrix,convNumber,30)
    egvls.append(egvl)
    workMatrix-=offset #remove the components of workMatrix corresponding to the converged egvl
    numRemoved+=convNumber
    egvlRemoved.append(egvl)


fig=go.Figure()
fig.add_trace(go.Scatter(x=eigenValues.real,y=eigenValues.imag,mode='markers',name="true eigenvalues"))
for i,egvlR in enumerate(egvlRemoved):
    fig.add_trace(go.Scatter(x=egvlR.real,y=egvlR.imag,mode='markers', name=f"eigenvalue set {i}"))
fig.show()


fig=go.Figure()
#fig.add_trace(go.Scatter(x=eigenValues.real,y=eigenValues.imag,mode='markers',name="true eigenvalues"))
for i,egvlR in enumerate(egvlRemoved):
    findMatch = np.searchsorted(np.abs(eigenValues),np.abs(egvlR)-0.00001)
    error=egvlR-eigenValues[findMatch]
    fig.add_trace(go.Scatter(x=error.real,y=error.imag,mode='markers', name=f"error, eigenvalue set {i}"))

fig.show()

