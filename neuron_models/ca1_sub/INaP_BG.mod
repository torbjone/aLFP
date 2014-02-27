TITLE Borg-Graham type I_NaP channel
UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
    e_rev = 30 (mV)
	celsius (degC)
	gnabar = .01 (mho/cm2)
    vhalfn = -47.   (mV)
    zeta = 6.5    (1)
    gamma = 0.5   (1)
    tau0 = 1. (ms)
    K = 0.0 (1/ms)
    R = 8315.
    F = 9.648e4
}


NEURON {
	SUFFIX INaP_BK
    USEION na WRITE ina
    RANGE gnabar
}

STATE {
	n
}

INITIAL {
        rates(v)
        n=ninf
}

ASSIGNED {
	ina (mA/cm2)
    gna
    ninf
    taun
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gna = gnabar*n
	ina = gna * (v - e_rev)
}


FUNCTION alpn(v(mV)) {
  alpn = exp(zeta * gamma * (v-vhalfn) * F / (R * (273.16+celsius)))
}

FUNCTION betn(v(mV)) {
  betn = exp(-zeta * (1 - gamma) * (v-vhalfn) * F / (R * (273.16+celsius)))
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n)/taun
}

PROCEDURE rates(v (mV)) { :callable from hoc
    LOCAL a, b
    a = alpn(v)
    b = betn(v)
    ninf = a /(a + b)
    taun = tau0
}

