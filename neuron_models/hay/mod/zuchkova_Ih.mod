:Comment : Implemented by Torbjorn Ness
:Reference : :		Zuchkova et al. 2013

NEURON	{
	SUFFIX Ih_z
	NONSPECIFIC_CURRENT ih
	RANGE gIhbar, ih
}

UNITS	{
	(S) = (siemens)
	(mV) = (millivolt)
	(mA) = (milliamp)
}

PARAMETER	{
	gIhbar = 0.00001 (S/cm2) 
	eh =  -43.0 (mV)
    fTau = 40 (ms)
    sTau = 300 (ms)
}

ASSIGNED	{
	v	(mV)
	ih	(mA/cm2)
    hInf
}

STATE	{ 
	f
    s
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	ih = gIhbar*(0.8*f + 0.2*s) * (v-eh)
}

DERIVATIVE states	{
	rates()
	f' = (hInf-f)/fTau
	s' = (hInf-s)/sTau
}

INITIAL{
	rates()
	s = hInf
	f = hInf
}

PROCEDURE rates(){
	UNITSOFF
		hInf = 1. / (1 + exp((v + 82.)/7))
	UNITSON
}
