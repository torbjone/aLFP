: quasi-active current
: linear current with voltage-dependent dynamics and activation time constant
: Michiel Remme, 2013

NEURON	{
	SUFFIX qa
	NONSPECIFIC_CURRENT i
	RANGE mtau, vss, i, ipas, iqa
	RANGE gbarm, mhalf, mk, Em
	RANGE gl, El
	RANGE phi
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	vss 	= -60 (mV)
	mtau 	= 40 (ms) : activation time constant of linearized current at vstar
	mhalf 	= -82 : half-active
	mk 	= -7 : Slope
	Em 	= -43 : Reversal potential
	gbarm   = 0.00427 (S/cm2)
	gl 	= 0.0001 (S/cm2)
	El
	phi
}

ASSIGNED	{
	v		(mV)
	i		(mA/cm2)
	ipas 	(mA/cm2)
	iqa		(mA/cm2)
	dminf
	mss
}

STATE	{ 
	w 		(mV)
}

INITIAL{
	mss    	= 1/(1+exp(-(vss-mhalf)/mk)) : m steady state
	dminf 	= mss*(1-mss)/mk
	w 	= dminf*(v-vss)
	El 	= (gl*vss + gbarm*mss*(vss-Em))/gl : such that vss is steady state
	phi 	= (gbarm*(Em-vss)*dminf)/(gl+gbarm*mss) : feedback
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	ipas 	= gl*(v-El)
	iqa 	= gbarm*(mss*(v-Em) + w*(vss-Em))
	i	= ipas + iqa
}

DERIVATIVE states	{
	w' = (dminf*(v-vss)-w)/mtau
}
