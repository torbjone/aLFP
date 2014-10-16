TITLE Borg-Graham type Ih channel
UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
    e_rev = -40 (mV)
	celsius (degC)
	gbar = .01 (mho/cm2)
    vhalfn = -90.   (mV)
    zeta = -3.1    (1)
    gamma = 0.5   (1)
    tau0 = 4. (ms)
    K = 0.006 (1/ms)
    R = 8315
    F = 9.648e4
}


NEURON {
	SUFFIX debug_BG
    NONSPECIFIC_CURRENT ih
    RANGE gbar, gh, ih, n, taun, ninf, alpn, betn
}

STATE {
	n
}

INITIAL {
        rates(v)
        n=ninf
}

ASSIGNED {
	ih (mA/cm2)
    gh
    ninf
    taun
    alpn
    betn
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gh = gbar*n
	ih = gh * (v - e_rev)
}


DERIVATIVE states {
    rates(v)
    n' = (ninf - n)/taun :This was oposite sign in supl. material, but that doesn't make sense right?
}

PROCEDURE rates(v (mV)) {
    alpn = K * exp(zeta * gamma * (v-vhalfn) * F / (R * (273.16+celsius)))
    betn = K * exp(-zeta * (1 - gamma) * (v-vhalfn) * F / (R * (273.16+celsius)))
    ninf = alpn /(alpn + betn)
    taun = 1 / (alpn + betn) + tau0
}

