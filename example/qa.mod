TITLE leak + quasi-active current 
: Michiel Remme, 2013
: Modified by Ness 2014

NEURON	{
	SUFFIX QA
	NONSPECIFIC_CURRENT i
	RANGE g_pas, mu, g_w_bar, i, V_r, tau_w, w_inf
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
    g_w_bar     = 0.0001 (S/cm2)
    w_inf   = 0.5
}

ASSIGNED	{
	v		(mV)
	i       (mA/cm2)
}

STATE	{ 
	m	: linear gating variable
}

INITIAL  {
    gamma_R = (1 + g_w_bar * w_inf / g_pas)
    m = 0
}

BREAKPOINT	{
	SOLVE states METHOD cnexp
	i = g_pas * (gamma_R * (v - V_r) + m * mu)
}

DERIVATIVE states	{
    m' = (v - V_r - m)/tau_w
}
