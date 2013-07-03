:Comment : Linearized by Torbjorn Ness 2013
:Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006

NEURON	{
	SUFFIX Ih_linearized
	NONSPECIFIC_CURRENT i
	RANGE gIhbar, ihcn, vss, gl, el, il, i
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gIhbar = 0.00001	(S/cm2) 
	ehcn =  -45.0 		(mV)
	vss     		(mV)
	gl = 0.001 		(S/cm2) <0,1e9>
	el   	 		(mV)
}

ASSIGNED	{
	v	(mV)
	ihcn	(mA/cm2)
	mInf
	i 	(mA/cm2)
	il 	(mA/cm2)
	mTau
	mAlpha
	a1
	a2
	a3
	b1
	b2
	b3
	foo
	dminf
	mBeta
}

STATE	{ 
	m
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	ihcn  = gIhbar*(mInf*(v-ehcn) + m*(vss - ehcn))
	il    = gl*(v - el) : Passive leak current
	i     = ihcn + il
}

DERIVATIVE states	{
	m' = (dminf*(v - vss) -m)/mTau
}

INITIAL{
	a1	= 0.001*6.43
	a2 	= 154.9
	a3 	= 11.9
	b1 	= 0.001*193
	b2 	= 33.1
	mAlpha 	= a1*(vss+a2)/(exp((vss+a2)/a3)-1)
	mBeta  	=  b1*exp(vss/b2)
	mInf 	= mAlpha/(mAlpha + mBeta)
	foo 	= mAlpha/(a3*a1) * exp((vss + a2)/a3) + (vss + a2)/b2 - 1
	dminf 	= - mBeta/mAlpha / (1 + mBeta/mAlpha)^2 /(vss + a2) * foo
	mTau 	= 1/(mAlpha + mBeta)
	m 	= mInf : SURE ABOUT THIS ONE? Not as in qa.mod
	el 	= (gl*vss + gIhbar*mInf*(vss-ehcn))/gl : such that vss is steady state
}
