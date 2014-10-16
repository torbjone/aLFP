TITLE leak + quasi-active current 
: Michiel Remme, 2013

NEURON	{
	SUFFIX QA_old
	NONSPECIFIC_CURRENT i
	RANGE gm, em : total membrane conductance and weighted reversal potential (=resting potential)
	RANGE phi, taum
	RANGE i, im, iqa
}

UNITS	{
	(S) 	= (siemens)
	(mV) 	= (millivolt)
	(mA) 	= (milliamp)
}

PARAMETER	{
	gm		= 0.0001    (S/cm2)
	em      = -60       (mV)
	phi 	= 0
    taum    = 1         (ms)
}

ASSIGNED	{
	v		(mV)
	i       (mA/cm2)
	im      (mA/cm2) : total passive current (includes ohmic component of active current)
	iqa     (mA/cm2)
}

STATE	{ 
	m	: linear gating variable
}

INITIAL  {
	m = v-em
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	im 		= gm*(v-em)
	iqa 	= -gm*phi*m
	i	 	= im + iqa
}

DERIVATIVE states	{
    m' = (v - em - m)/taum
}
