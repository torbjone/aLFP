TITLE Borg-Graham type Ih channel
UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	v (mV)
    eh = -40 (mV)
	celsius (degC)
	ghbar = .01 (mho/cm2)
    vhalfn = -90.   (mV)
    z = -3.1    (1)
    gamma = 0.5   (1)
    tau0 = 4 (ms)
    K = 0.006 (1/ms)
    R = 8315
    F = 9.648e4
}


NEURON {
	SUFFIX Ih_BK_dist
    NONSPECIFIC_CURRENT ih
    RANGE ghbar
    GLOBAL ninf, taun
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
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	gh = ghbar*n
	ih = gh * (v - eh)
}


FUNCTION alpn(v(mV)) {
  alpn = K * exp(z * gamma * (v-vhalfn) * F / (R * (273.16+celsius)))
}

FUNCTION betn(v(mV)) {
  betn = K * exp(-z * (1 - gamma) * (v-vhalfn) * F / (R * (273.16+celsius)))
}

DERIVATIVE states {
    rates(v)
    n' = (n - ninf)/taun
}

PROCEDURE rates(v (mV)) { :callable from hoc
    LOCAL a, q10, b
    :q10=3^((celsius-30)/10)
    a = alpn(v)
    b = betn(v)
    ninf = a /(a + b)
    taun = 1 / (a + b) + tau0
}

