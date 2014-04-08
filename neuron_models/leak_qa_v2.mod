TITLE leak + quasi-active current 
: Michiel Remme, 2013
: Modified by Ness 2014

NEURON	{
	SUFFIX QA
	NONSPECIFIC_CURRENT i
	RANGE g_pas, mu, g_w, i, V_r, tau_w
}

UNITS	{
	(S) 	= (siemens)
	(mV) 	= (millivolt)
	(mA) 	= (milliamp)
}

PARAMETER	{
	g_pas	= 0.0001    (S/cm2)
	:e_w     = -60       (mV)
    V_r     = -80 (mV)
    mu  	= 0
    tau_w    = 1         (ms)
    gamma_R
    g_w     = 0.0001 (S/cm2)
}

ASSIGNED	{
	v		(mV)
	i       (mA/cm2)
}

STATE	{ 
	m	: linear gating variable
}

INITIAL  {
    :e_leak = V_r - (gamma_R - 1) * e_w
    gamma_R = (1 + g_w / g_pas)
    m = 0
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	i = g_pas * (gamma_R * v + m * mu - V_r * gamma_R)
}

DERIVATIVE states	{
    m' = (v - V_r - m)/tau_w
}
