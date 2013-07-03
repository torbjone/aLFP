:Comment : Added passive current and modified restingstate by Torbjorn Ness 2013
:Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON	{
	SUFFIX Ih_reduced
	NONSPECIFIC_CURRENT i
	RANGE gIhbar, gIh, ihcn, gl, el, il, i, vss
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gIhbar = 0.00001	(S/cm2) 
	ehcn =  -45.0 		(mV)	
	gl = 0.001 		(S/cm2) <0,1e9>
	vss  			(mV)
	el      	   	(mV)
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)
	i 	(mA/cm2)
	il 	(mA/cm2)
	gIh	(S/cm2)
	mInf
	mTau
	mAlpha
	mBeta
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	gIh = gIhbar*m
	ihcn = gIh*(v-ehcn)
	il = gl*(v - el) : Passive leak current
	i = ihcn + il
}

DERIVATIVE states	{
	rates()
	m' = (mInf-m)/mTau
}

INITIAL{
	rates()
	m = mInf
	el 	= (gl*vss + gIhbar*mInf*(vss-ehcn))/gl : such that vss is steady state
}

PROCEDURE rates(){
	UNITSOFF
        if(v == -154.9){
            v = v + 0.0001
        }
		mAlpha =  0.001*6.43*(v+154.9)/(exp((v+154.9)/11.9)-1)
		mBeta  =  0.001*193*exp(v/33.1)
		mInf = mAlpha/(mAlpha + mBeta)
		mTau = 1/(mAlpha + mBeta)
	UNITSON
}
